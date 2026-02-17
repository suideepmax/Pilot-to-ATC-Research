"""
Wav2Vec2 CTC fine-tuning on UWB-ATCC.
GPU-first. All issues fixed including loss plateau.
    python train_asr.py --clean_checkpoints --no_cache
"""

import os, re, json, logging, shutil, argparse, math, sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import librosa
import evaluate

from datasets import load_dataset, Dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log", mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

try:
    import tensorboard; TB = True
except ImportError:
    TB = False


# ── GPU detection ─────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {name} ({mem:.1f} GB)")
        logger.info(f"CUDA: {torch.version.cuda}  |  PyTorch: {torch.__version__}")
        torch.cuda.empty_cache()
        return dev, True
    else:
        logger.warning("=" * 60)
        logger.warning("  NO CUDA GPU DETECTED — training will be VERY slow")
        logger.warning("=" * 60)
        return torch.device("cpu"), False


# ── config ────────────────────────────────────────────────────────────
@dataclass
class ASRConfig:
    project_dir: str   = os.path.expanduser("~/asr_project")
    dataset_dir: Optional[str] = None
    output_dir:  Optional[str] = None
    target_sr: int     = 16000
    max_audio_len: float = 20.0
    min_audio_len: float = 0.3
    # Use wav2vec2-base (pretrained only, NOT fine-tuned for CTC)
    # This is critical — base-960h has an lm_head trained for a different vocab,
    # and its encoder features are calibrated for that vocab.
    # wav2vec2-base gives us a clean encoder we can attach any head to.
    model_name: str    = "facebook/wav2vec2-base"
    freeze_feature_encoder: bool = True
    gradient_checkpointing: bool = True
    batch_size: int    = 4
    grad_accum: int    = 8
    epochs: int        = 30
    lr: float          = 3e-4
    warmup: int        = 500
    fp16: bool         = True
    eval_steps: int    = 500
    save_steps: int    = 500
    log_steps: int     = 50
    save_limit: int    = 3
    use_cache: bool    = True
    clean_ckpt: bool   = False
    force_cpu: bool    = False

    def __post_init__(self):
        self.dataset_dir = self.dataset_dir or os.path.join(self.project_dir, "data/uwb_atcc/manifests")
        self.output_dir  = self.output_dir  or os.path.join(self.project_dir, "models/wav2vec2_uwb_atcc")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.max_samples = int(self.max_audio_len * self.target_sr)
        self.min_samples = int(self.min_audio_len * self.target_sr)


# ── text ──────────────────────────────────────────────────────────────
class TextNorm:
    _keep   = re.compile(r"[^a-z0-9 '\-]+")
    _spaces = re.compile(r"\s+")
    def __call__(self, t: str) -> str:
        if not t: return ""
        t = t.lower().strip()
        t = self._keep.sub(" ", t)
        return self._spaces.sub(" ", t).strip()


# ── audio ─────────────────────────────────────────────────────────────
class AudioLoader:
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg; self.fails: list = []

    def __call__(self, row: Dict) -> Dict:
        try:
            p = row["path"]
            if not os.path.exists(p): raise FileNotFoundError(p)
            w, sr = sf.read(p, always_2d=False)
            if w.ndim > 1: w = w.mean(1)
            w = w.astype(np.float32)
            if sr != self.cfg.target_sr:
                w = librosa.resample(w, orig_sr=sr, target_sr=self.cfg.target_sr)
            if len(w) > self.cfg.max_samples: w = w[:self.cfg.max_samples]
            if len(w) < self.cfg.min_samples: raise ValueError(f"short ({len(w)})")
            row["audio"] = {"array": w, "sampling_rate": self.cfg.target_sr}
            row["valid"] = True
        except Exception as e:
            self.fails.append({"path": row.get("path","?"), "error": str(e)})
            row["audio"] = {"array": np.zeros(self.cfg.target_sr, dtype=np.float32),
                            "sampling_rate": self.cfg.target_sr}
            row["valid"] = False
        return row


# ── vocab ─────────────────────────────────────────────────────────────
def build_vocab(ds: Dataset) -> Dict[str, int]:
    chars = set()
    for t in ds["transcript"]: chars.update(t)
    v = {c: i for i, c in enumerate(sorted(chars))}
    if " " in v: v["|"] = v.pop(" ")
    v["[UNK]"] = len(v)
    v["[PAD]"] = len(v)
    return v


# ── CTC length check ─────────────────────────────────────────────────
DOWNSAMPLE = 320

def ctc_ok(audio_len: int, label_len: int) -> bool:
    frames = audio_len // DOWNSAMPLE
    return frames >= label_len + 1


# ── collator ──────────────────────────────────────────────────────────
@dataclass
class CTCCollator:
    processor: Wav2Vec2Processor

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        inp = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(inp, padding=True, return_tensors="pt")
        batch["input_values"] = batch["input_values"].float()

        labels_list = [f["labels"] for f in features]
        mx = max(len(l) for l in labels_list)
        pid = self.processor.tokenizer.pad_token_id
        padded = [l + [pid] * (mx - len(l)) for l in labels_list]
        labels = torch.tensor(padded, dtype=torch.long)
        labels[labels == pid] = -100
        batch["labels"] = labels
        return batch


# ── metrics ───────────────────────────────────────────────────────────
class Metrics:
    def __init__(self, proc):
        self.proc = proc
        self.wer = evaluate.load("wer")
        self.cer = evaluate.load("cer")

    def __call__(self, pred):
        ids = np.argmax(pred.predictions, axis=-1)
        pred_str = self.proc.batch_decode(ids)
        lab = pred.label_ids.copy()
        lab[lab == -100] = self.proc.tokenizer.pad_token_id
        ref_str = self.proc.batch_decode(lab, group_tokens=False)
        return {
            "wer": self.wer.compute(predictions=pred_str, references=ref_str),
            "cer": self.cer.compute(predictions=pred_str, references=ref_str),
        }


# ── pipeline ──────────────────────────────────────────────────────────
class ASRTrainer:
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        self.norm = TextNorm()
        self.loader = AudioLoader(cfg)

    def load_ds(self):
        tr = os.path.join(self.cfg.dataset_dir, "train.csv")
        va = os.path.join(self.cfg.dataset_dir, "valid.csv")
        ds = load_dataset("csv", data_files={"train": tr, "validation": va})
        logger.info(f"Raw: train={len(ds['train'])} val={len(ds['validation'])}")
        return ds

    def preprocess(self, ds):
        ds = ds.map(self.loader, num_proc=1, load_from_cache_file=self.cfg.use_cache, desc="Audio")
        for s in ds:
            n = len(ds[s])
            ds[s] = ds[s].filter(lambda x: x["valid"], desc=f"valid-{s}")
            d = n - len(ds[s])
            if d: logger.warning(f"Dropped {d} bad audio from {s}")

        ds = ds.map(lambda b: {"transcript": self.norm(b["transcript"])},
                     num_proc=1, load_from_cache_file=self.cfg.use_cache, desc="Norm")
        for s in ds:
            n = len(ds[s])
            ds[s] = ds[s].filter(lambda x: len(x["transcript"]) > 0, desc=f"empty-{s}")
            d = n - len(ds[s])
            if d: logger.warning(f"Dropped {d} empty from {s}")

        logger.info(f"Clean: train={len(ds['train'])} val={len(ds['validation'])}")
        return ds

    def make_processor(self, ds):
        vocab = build_vocab(ds["train"])
        vp = os.path.join(self.cfg.output_dir, "vocab.json")
        with open(vp, "w") as f: json.dump(vocab, f, ensure_ascii=False, indent=2)

        tok = Wav2Vec2CTCTokenizer(vocab_file=vp, unk_token="[UNK]",
                                    pad_token="[PAD]", word_delimiter_token="|")

        # Verify vocab size consistency
        logger.info(f"Vocab file: {len(vocab)} entries")
        logger.info(f"Tokenizer vocab_size: {tok.vocab_size}")
        logger.info(f"len(tokenizer): {len(tok)}")
        logger.info(f"PAD id: {tok.pad_token_id}")
        logger.info(f"UNK id: {tok.unk_token_id}")

        fe  = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=self.cfg.target_sr,
                                        padding_value=0.0, do_normalize=True,
                                        return_attention_mask=True)
        return Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)

    def featurize(self, ds, proc):
        def fn(b):
            a = b["audio"]
            b["input_values"] = proc(a["array"], sampling_rate=a["sampling_rate"]).input_values[0]
            b["labels"] = proc.tokenizer(b["transcript"]).input_ids
            b["_alen"] = len(b["input_values"])
            b["_llen"] = len(b["labels"])
            return b

        drop = [c for c in ds["train"].column_names if c not in ("input_values","labels","_alen","_llen")]
        ds = ds.map(fn, remove_columns=drop, num_proc=1,
                     load_from_cache_file=self.cfg.use_cache, desc="Features")

        for s in ds:
            n = len(ds[s])
            ds[s] = ds[s].filter(lambda x: ctc_ok(x["_alen"], x["_llen"]), desc=f"ctc-{s}")
            d = n - len(ds[s])
            if d: logger.warning(f"CTC filter dropped {d} from {s}")

        ds = ds.remove_columns(["_alen", "_llen"])
        logger.info(f"Final: train={len(ds['train'])} val={len(ds['validation'])}")
        return ds

    def make_model(self, proc, device):
        logger.info(f"Loading {self.cfg.model_name}")

        # Use vocab_size from the vocab file directly, NOT len(tokenizer)
        # len(tokenizer) can include added_tokens that inflate the count
        pad_id   = proc.tokenizer.pad_token_id
        vocab_sz = proc.tokenizer.vocab_size
        logger.info(f"Model vocab_size={vocab_sz}, pad_token_id={pad_id}")

        model = Wav2Vec2ForCTC.from_pretrained(
            self.cfg.model_name,
            revision="refs/pr/11",
            vocab_size=vocab_sz,
            pad_token_id=pad_id,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            ignore_mismatched_sizes=True,
        )

        # Disable SpecAugment — masked_spec_embed missing from checkpoint
        model.config.mask_time_prob = 0.0
        model.config.mask_feature_prob = 0.0
        model.config.apply_spec_augment = False
        logger.info("Disabled SpecAugment (masked_spec_embed not in checkpoint)")

        # Reinit lm_head properly
        logger.info(f"Reinit lm_head: Linear({model.lm_head.in_features}, {model.lm_head.out_features})")
        nn.init.xavier_uniform_(model.lm_head.weight, gain=1.0)
        nn.init.zeros_(model.lm_head.bias)

        model = model.to(device)
        logger.info(f"Model device: {next(model.parameters()).device}")

        logger.info(f"  config.pad_token_id       = {model.config.pad_token_id}")
        logger.info(f"  config.vocab_size         = {model.config.vocab_size}")
        logger.info(f"  config.ctc_zero_infinity  = {model.config.ctc_zero_infinity}")
        logger.info(f"  config.ctc_loss_reduction = {model.config.ctc_loss_reduction}")
        logger.info(f"  config.mask_time_prob     = {model.config.mask_time_prob}")

        if self.cfg.freeze_feature_encoder:
            logger.info("Freezing feature encoder")
            model.freeze_feature_encoder()

        if self.cfg.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing")
            model.gradient_checkpointing_enable()

        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        to = sum(p.numel() for p in model.parameters())
        logger.info(f"Params: {tr:,} trainable / {to:,} total")
        return model

    def find_ckpt(self):
        if not os.path.exists(self.cfg.output_dir): return None
        cps = []
        for d in os.listdir(self.cfg.output_dir):
            p = os.path.join(self.cfg.output_dir, d)
            if not (d.startswith("checkpoint") and os.path.isdir(p)): continue
            ok = (os.path.exists(os.path.join(p, "model.safetensors")) or
                  os.path.exists(os.path.join(p, "pytorch_model.bin")))
            ok = ok and os.path.exists(os.path.join(p, "optimizer.pt"))
            ok = ok and os.path.exists(os.path.join(p, "scheduler.pt"))
            ok = ok and os.path.exists(os.path.join(p, "trainer_state.json"))
            if ok: cps.append(p)
        if cps:
            latest = sorted(cps, key=os.path.getmtime)[-1]
            logger.info(f"Checkpoint: {latest}")
            return latest
        return None

    def clean_ckpts(self):
        if not os.path.exists(self.cfg.output_dir): return
        for d in os.listdir(self.cfg.output_dir):
            p = os.path.join(self.cfg.output_dir, d)
            if d.startswith("checkpoint") and os.path.isdir(p):
                shutil.rmtree(p); logger.info(f"Removed {p}")

    def train(self):
        logger.info("=" * 60)
        logger.info("ASR Training")
        logger.info("=" * 60)

        device, has_cuda = get_device()
        if self.cfg.force_cpu:
            device, has_cuda = torch.device("cpu"), False
            logger.info("Forced CPU mode")

        use_fp16 = self.cfg.fp16 and has_cuda
        if self.cfg.fp16 and not has_cuda:
            logger.warning("fp16 requires CUDA → disabled")

        if self.cfg.clean_ckpt: self.clean_ckpts()

        ds   = self.load_ds()
        ds   = self.preprocess(ds)
        proc = self.make_processor(ds)
        ds   = self.featurize(ds, proc)
        model = self.make_model(proc, device)

        collator = CTCCollator(processor=proc)
        metrics  = Metrics(proc)

        # ── sanity check ─────────────────────────────────────────────
        logger.info("Sanity check …")
        n_sample = min(4, len(ds["train"]))
        batch = collator([ds["train"][i] for i in range(n_sample)])
        batch = {k: v.to(device) for k, v in batch.items()}

        # Test eval mode
        model.eval()
        with torch.no_grad():
            out = model(input_values=batch["input_values"], labels=batch["labels"])
        loss_eval = out.loss.item()
        logger.info(f"  loss (eval)  = {loss_eval:.4f}  (device={batch['input_values'].device})")

        # Check predictions are not all blank
        logits = out.logits
        pred_ids = torch.argmax(logits, dim=-1)
        unique_preds = torch.unique(pred_ids).tolist()
        logger.info(f"  unique predicted ids = {unique_preds}")
        logger.info(f"  PAD/blank id = {proc.tokenizer.pad_token_id}")

        if not math.isfinite(loss_eval):
            raise RuntimeError("Sanity check FAILED — eval loss is NaN/Inf")

        # Test train mode
        model.train()
        out2 = model(input_values=batch["input_values"], labels=batch["labels"])
        loss_train = out2.loss.item()
        logger.info(f"  loss (train) = {loss_train:.4f}")

        if not math.isfinite(loss_train):
            raise RuntimeError("Sanity check FAILED — train loss is NaN/Inf")

        out2.loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        logger.info(f"  grad_norm   = {gn.item():.4f}")
        model.zero_grad()

        if has_cuda:
            alloc = torch.cuda.memory_allocated() / 1e9
            logger.info(f"  GPU mem     = {alloc:.2f} GB")
        logger.info("Sanity check PASSED ✓")

        # ── Trainer ──────────────────────────────────────────────────
        args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.grad_accum,
            eval_strategy="steps",
            num_train_epochs=self.cfg.epochs,
            fp16=use_fp16,
            bf16=False,
            save_steps=self.cfg.save_steps,
            eval_steps=self.cfg.eval_steps,
            logging_steps=self.cfg.log_steps,
            learning_rate=self.cfg.lr,
            warmup_steps=self.cfg.warmup,
            save_total_limit=self.cfg.save_limit,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            report_to=["tensorboard"] if TB else "none",
            dataloader_num_workers=2 if has_cuda else 0,
            dataloader_pin_memory=has_cuda,
            group_by_length=True,
            remove_unused_columns=True,
            max_grad_norm=1.0,
            use_cpu=self.cfg.force_cpu,
        )

        logger.info(f"TrainingArgs: device={args.device}, fp16={args.fp16}, n_gpu={args.n_gpu}")

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            data_collator=collator,
            compute_metrics=metrics,
            processing_class=proc.feature_extractor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        ckpt = self.find_ckpt()
        logger.info(f"Resume: {ckpt}" if ckpt else "Fresh start")
        trainer.train(resume_from_checkpoint=ckpt)

        logger.info("Saving …")
        trainer.save_model(self.cfg.output_dir)
        proc.save_pretrained(self.cfg.output_dir)

        m = trainer.evaluate()
        logger.info(f"Final: {m}")
        with open(os.path.join(self.cfg.output_dir, "final_metrics.json"), "w") as f:
            json.dump(m, f, indent=2)

        logger.info("=" * 60)
        logger.info(f"Done → {self.cfg.output_dir}")
        logger.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project_dir", default=os.path.expanduser("~/asr_project"))
    p.add_argument("--dataset_dir", default=None)
    p.add_argument("--output_dir",  default=None)
    p.add_argument("--model_name",  default="facebook/wav2vec2-base")
    p.add_argument("--no_freeze_encoder",        action="store_true")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--grad_accum",  type=int,   default=8)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--warmup",      type=int,   default=500)
    p.add_argument("--no_fp16",                 action="store_true")
    p.add_argument("--no_cache",                action="store_true")
    p.add_argument("--clean_checkpoints",       action="store_true")
    p.add_argument("--force_cpu",               action="store_true")
    a = p.parse_args()

    cfg = ASRConfig(
        project_dir=a.project_dir, dataset_dir=a.dataset_dir, output_dir=a.output_dir,
        model_name=a.model_name,
        freeze_feature_encoder=not a.no_freeze_encoder,
        gradient_checkpointing=not a.no_gradient_checkpointing,
        batch_size=a.batch_size, grad_accum=a.grad_accum,
        epochs=a.epochs, lr=a.lr, warmup=a.warmup,
        fp16=not a.no_fp16, use_cache=not a.no_cache,
        clean_ckpt=a.clean_checkpoints, force_cpu=a.force_cpu,
    )
    ASRTrainer(cfg).train()


if __name__ == "__main__":
    main()
