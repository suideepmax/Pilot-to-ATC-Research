"""
Wav2Vec2 CTC fine-tuning — Stage 2
Fine-tunes FROM Stage 1 (wav2vec2-base on UWB-ATCC) with regularization.
Optimized for LOW RAM (8GB system) + 4GB VRAM.

    python train_asr_stage2.py
"""

import os, re, json, logging, shutil, argparse, math, gc
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    handlers=[logging.FileHandler("training_stage2.log", mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

try:
    import tensorboard; TB = True
except ImportError:
    TB = False


# ── RAM logging ───────────────────────────────────────────────────────
def log_ram():
    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info(f"RAM: {mem.used / 1e9:.1f} / {mem.total / 1e9:.1f} GB ({mem.percent}%)")
    except ImportError:
        pass


# ── GPU detection ─────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"GPU: {name} ({mem:.1f} GB)")
        logger.info(f"CUDA: {torch.version.cuda}  |  PyTorch: {torch.__version__}")
        torch.cuda.empty_cache()
        gc.collect()
        return dev, True
    else:
        logger.warning("=" * 60)
        logger.warning("  NO CUDA GPU DETECTED — training will be VERY slow")
        logger.warning("=" * 60)
        return torch.device("cpu"), False


# ── config ────────────────────────────────────────────────────────────@dataclass
class ASRConfig:
    project_dir: str    = os.path.expanduser("~/asr_project")
    dataset_dir: Optional[str] = None
    stage1_model: Optional[str] = None
    output_dir: Optional[str]   = None

    target_sr: int      = 16000
    max_audio_len: float = 15.0
    min_audio_len: float = 0.3

    freeze_feature_encoder: bool = True
    gradient_checkpointing: bool = True

    # SpecAugment
    mask_time_prob: float    = 0.05
    mask_time_length: int    = 10
    mask_feature_prob: float = 0.0
    mask_feature_length: int = 10

    # Dropout
    attention_dropout: float = 0.1
    hidden_dropout: float    = 0.1
    feat_proj_dropout: float = 0.0
    layerdrop: float         = 0.05

    # Training
    batch_size: int     = 2
    grad_accum: int     = 16
    epochs: int         = 20
    lr: float           = 5e-5
    warmup: int         = 300
    weight_decay: float = 0.005
    fp16: bool          = True
    eval_steps: int     = 500
    save_steps: int     = 500
    log_steps: int      = 50
    save_limit: int     = 2
    patience: int       = 8
    use_cache: bool     = True
    force_cpu: bool     = False

    def __post_init__(self):
        self.dataset_dir  = self.dataset_dir  or os.path.join(self.project_dir, "data/uwb_atcc/manifests")
        self.stage1_model = self.stage1_model or os.path.join(self.project_dir, "models/wav2vec2_uwb_atcc")
        self.output_dir   = self.output_dir   or os.path.join(self.project_dir, "models/wav2vec2_uwb_atcc_v2")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.max_samples = int(self.max_audio_len * self.target_sr)
        self.min_samples = int(self.min_audio_len * self.target_sr)


# ── text normalizer (same as Stage 1) ────────────────────────────────class TextNorm:
    _keep   = re.compile(r"[^a-z0-9 '\-]+")
    _spaces = re.compile(r"\s+")
    def __call__(self, t: str) -> str:
        if not t: return ""
        t = t.lower().strip()
        t = self._keep.sub(" ", t)
        return self._spaces.sub(" ", t).strip()


# ── On-the-fly audio loading ─────────────────────────────────────────def load_and_process_audio(path, target_sr, max_samples, min_samples):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    w, sr = sf.read(path, always_2d=False)
    if w.ndim > 1:
        w = w.mean(1)
    w = w.astype(np.float32)
    if sr != target_sr:
        w = librosa.resample(w, orig_sr=sr, target_sr=target_sr)
    if len(w) > max_samples:
        w = w[:max_samples]
    if len(w) < min_samples:
        return None
    return w


# ── CTC length check (same as Stage 1) ───────────────────────────────DOWNSAMPLE = 320

def ctc_ok(audio_len: int, label_len: int) -> bool:
    return audio_len // DOWNSAMPLE >= label_len + 1


# ── collator (same as Stage 1) ───────────────────────────────────────@dataclass
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


# ── metrics (same as Stage 1) ────────────────────────────────────────class Metrics:
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


# ── Stage 2 pipeline ─────────────────────────────────────────────────class ASRTrainerStage2:
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        self.norm = TextNorm()

    def load_ds(self):
        tr = os.path.join(self.cfg.dataset_dir, "train.csv")
        va = os.path.join(self.cfg.dataset_dir, "valid.csv")
        ds = load_dataset("csv", data_files={"train": tr, "validation": va})
        logger.info(f"Raw: train={len(ds['train'])} val={len(ds['validation'])}")
        return ds

    def preprocess_text(self, ds):
        ds = ds.map(
            lambda b: {"transcript": self.norm(b["transcript"])},
            num_proc=1,
            load_from_cache_file=self.cfg.use_cache,
            desc="Norm",
        )
        for s in ds:
            n = len(ds[s])
            ds[s] = ds[s].filter(lambda x: len(x["transcript"]) > 0, desc=f"empty-{s}")
            d = n - len(ds[s])
            if d: logger.warning(f"Dropped {d} empty from {s}")

        logger.info(f"After text norm: train={len(ds['train'])} val={len(ds['validation'])}")
        log_ram()
        return ds

    def make_processor(self, ds):
        stage1_vocab = os.path.join(self.cfg.stage1_model, "vocab.json")
        vp = os.path.join(self.cfg.output_dir, "vocab.json")
        if os.path.exists(stage1_vocab):
            logger.info(f"Using Stage 1 vocab: {stage1_vocab}")
            shutil.copy2(stage1_vocab, vp)
        else:
            raise FileNotFoundError(f"Stage 1 vocab not found: {stage1_vocab}")

        tok = Wav2Vec2CTCTokenizer(vocab_file=vp, unk_token="[UNK ",
                                    pad_token="[PAD]", word_delimiter_token="|")
        fe  = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=self.cfg.target_sr,
                                        padding_value=0.0, do_normalize=True,
                                        return_attention_mask=True)
        proc = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)

        logger.info(f"Vocab file entries: {json.load(open(vp)).__len__()}")
        logger.info(f"Tokenizer vocab_size: {tok.vocab_size}")
        logger.info(f"len(tokenizer): {len(tok)}")
        logger.info(f"PAD id: {tok.pad_token_id}")
        logger.info(f"UNK id: {tok.unk_token_id}")
        return proc

    def featurize(self, ds, proc):
        cfg = self.cfg

        def process_single(example):
            path = example["path"]
            transcript = example["transcript"]

            try:
                audio = load_and_process_audio(
                    path, cfg.target_sr, cfg.max_samples, cfg.min_samples
                )
            except Exception:
                audio = None

            if audio is None:
                example["input_values"] = [0.0]
                example["labels"] = []
                example["_valid"] = False
                return example

            inputs = proc(audio, sampling_rate=cfg.target_sr)
            example["input_values"] = inputs.input_values[0]
            example["labels"] = proc.tokenizer(transcript).input_ids

            alen = len(example["input_values"])
            llen = len(example["labels"])
            example["_valid"] = ctc_ok(alen, llen) and llen > 0

            return example

        drop_cols = [c for c in ds["train"].column_names
                     if c not in ("input_values", "labels", "_valid")]

        for split in ds:
            logger.info(f"Featurizing {split}...")
            ds[split] = ds[split].map(
                process_single,
                remove_columns=drop_cols,
                num_proc=1,
                load_from_cache_file=self.cfg.use_cache,
                writer_batch_size=500,
                desc=f"Features-{split}",
            )
            gc.collect()
            log_ram()

        for split in ds:
            n = len(ds[split])
            ds[split] = ds[split].filter(lambda x: x["_valid"], desc=f"valid-{split}")
            d = n - len(ds[split])
            if d: logger.warning(f"Dropped {d} invalid from {split}")

        ds = ds.remove_columns(["_valid"])
        logger.info(f"Final: train={len(ds['train'])} val={len(ds['validation'])}")
        log_ram()
        gc.collect()
        return ds

    def make_model(self, proc, device):
        logger.info(f"Loading Stage 1 model: {self.cfg.stage1_model}")
        log_ram()

        pad_id   = proc.tokenizer.pad_token_id
        vocab_sz = proc.tokenizer.vocab_size
        logger.info(f"Model vocab_size={vocab_sz}, pad_token_id={pad_id}")

        model = Wav2Vec2ForCTC.from_pretrained(
            self.cfg.stage1_model,
            vocab_size=vocab_sz,
            pad_token_id=pad_id,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
        )

        model.config.mask_time_prob = self.cfg.mask_time_prob
        model.config.mask_time_length = self.cfg.mask_time_length
        model.config.mask_feature_prob = self.cfg.mask_feature_prob
        model.config.mask_feature_length = self.cfg.mask_feature_length
        model.config.apply_spec_augment = self.cfg.mask_time_prob > 0

        if self.cfg.mask_time_prob > 0:
            if hasattr(model.wav2vec2, 'masked_spec_embed') and \
               model.wav2vec2.masked_spec_embed is not None:
                nn.init.zeros_(model.wav2vec2.masked_spec_embed)
                model.wav2vec2.masked_spec_embed.requires_grad = True
                logger.info("masked_spec_embed exists — reset to zeros (learnable)")
            else:
                hidden_size = model.config.hidden_size
                model.wav2vec2.masked_spec_embed = nn.Parameter(
                    torch.zeros(hidden_size, dtype=torch.float32)
                )
                logger.info(f"Created masked_spec_embed (hidden_size={hidden_size}) — "
                            f"Stage 1 did not have it")

        logger.info(f"SpecAugment: time_prob={self.cfg.mask_time_prob}, "
                     f"feature_prob={self.cfg.mask_feature_prob}")

        model.config.attention_dropout = self.cfg.attention_dropout
        model.config.hidden_dropout = self.cfg.hidden_dropout
        model.config.feat_proj_dropout = self.cfg.feat_proj_dropout
        model.config.layerdrop = self.cfg.layerdrop
        logger.info(f"Dropout: attn={self.cfg.attention_dropout}, "
                     f"hidden={self.cfg.hidden_dropout}, "
                     f"feat_proj={self.cfg.feat_proj_dropout}, "
                     f"layerdrop={self.cfg.layerdrop}")

        gc.collect()
        model = model.to(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"Model device: {next(model.parameters()).device}")
        log_ram()

        logger.info(f"  config.pad_token_id       = {model.config.pad_token_id}")
        logger.info(f"  config.vocab_size         = {model.config.vocab_size}")
        logger.info(f"  config.ctc_zero_infinity  = {model.config.ctc_zero_infinity}")
        logger.info(f"  config.ctc_loss_reduction = {model.config.ctc_loss_reduction}")
        logger.info(f"  config.mask_time_prob     = {model.config.mask_time_prob}")
        logger.info(f"  config.mask_feature_prob  = {model.config.mask_feature_prob}")

        if self.cfg.freeze_feature_encoder:
            model.freeze_feature_encoder()
            logger.info("Feature encoder frozen")

        if self.cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        to = sum(p.numel() for p in model.parameters())
        logger.info(f"Params: {tr:,} trainable / {to:,} total")

        if device.type == "cuda":
            logger.info(f"GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        return model

    def find_ckpt(self):
        if not os.path.exists(self.cfg.output_dir):
            return None
        cps = []
        for d in os.listdir(self.cfg.output_dir):
            p = os.path.join(self.cfg.output_dir, d)
            if not (d.startswith("checkpoint") and os.path.isdir(p)):
                continue
            ok = (os.path.exists(os.path.join(p, "model.safetensors")) or
                  os.path.exists(os.path.join(p, "pytorch_model.bin")))
            ok = ok and os.path.exists(os.path.join(p, "optimizer.pt"))
            ok = ok and os.path.exists(os.path.join(p, "scheduler.pt"))
            ok = ok and os.path.exists(os.path.join(p, "trainer_state.json"))
            if ok:
                cps.append(p)
        if cps:
            latest = sorted(cps, key=os.path.getmtime)[-1]
            logger.info(f"Checkpoint: {latest}")
            return latest
        return None

    def train(self):
        logger.info("=" * 60)
        logger.info("ASR Training — Stage 2")
        logger.info("=" * 60)
        log_ram()

        if not os.path.exists(self.cfg.stage1_model):
            raise FileNotFoundError(f"Stage 1 not found: {self.cfg.stage1_model}")

        device, has_cuda = get_device()
        if self.cfg.force_cpu:
            device, has_cuda = torch.device("cpu"), False
            logger.info("Forced CPU mode")

        use_fp16 = self.cfg.fp16 and has_cuda
        if self.cfg.fp16 and not has_cuda:
            logger.warning("fp16 requires CUDA — disabled")

        if has_cuda:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        ds   = self.load_ds()
        ds   = self.preprocess_text(ds)
        proc = self.make_processor(ds)

        gc.collect()
        log_ram()

        ds = self.featurize(ds, proc)

        gc.collect()
        log_ram()

        model = self.make_model(proc, device)

        collator = CTCCollator(processor=proc)
        metrics  = Metrics(proc)

        logger.info("Sanity check …")
        n_sample = min(4, len(ds["train"]))
        batch = collator([ds["train"][i] for i in range(n_sample)])
        batch = {k: v.to(device) for k, v in batch.items()}

        model.eval()
        with torch.no_grad():
            out = model(input_values=batch["input_values"], labels=batch["labels"])\n
        loss_eval = out.loss.item()
        logger.info(f"  loss (eval)  = {loss_eval:.4f}  (device={batch['input_values'].device})")

        logits = out.logits
        pred_ids = torch.argmax(logits, dim=-1)
        unique_preds = torch.unique(pred_ids).tolist()
        logger.info(f"  unique predicted ids = {unique_preds}")
        logger.info(f"  PAD/blank id = {proc.tokenizer.pad_token_id}")

        pred_str = proc.batch_decode(pred_ids)
        lab = batch["labels"].clone()
        lab[lab == -100] = proc.tokenizer.pad_token_id
        ref_str = proc.batch_decode(lab, group_tokens=False)
        logger.info(f"  REF:  '{ref_str[0][:80]}'")
        logger.info(f"  PRED: '{pred_str[0][:80]}'")

        if not math.isfinite(loss_eval):
            logger.error(f"  logits: min={logits.min().item():.4f} max={logits.max().item():.4f} "
                         f"nan={torch.isnan(logits).any().item()}")
            logger.error(f"  input:  {batch['input_values'].shape}  "
                         f"min={batch['input_values'].min().item():.4f} "
                         f"max={batch['input_values'].max().item():.4f} "
                         f"nan={torch.isnan(batch['input_values']).any().item()}")
            logger.error(f"  labels: {batch['labels'].shape}")

            with torch.no_grad():
                enc_out = model.wav2vec2(input_values=batch["input_values"])
                hidden = enc_out[0]
                logger.error(f"  encoder output: min={hidden.min().item():.4f} "
                             f"max={hidden.max().item():.4f} "
                             f"nan={torch.isnan(hidden).any().item()}")
                logits_raw = model.lm_head(model.dropout(hidden))
                logger.error(f"  lm_head output: min={logits_raw.min().item():.4f} "
                             f"max={logits_raw.max().item():.4f} "
                             f"nan={torch.isnan(logits_raw).any().item()}")
            raise RuntimeError("Sanity check FAILED — eval loss is NaN/Inf")

        model.train()
        out2 = model(input_values=batch["input_values"], labels=batch["labels"])\n        loss_train = out2.loss.item()
        logger.info(f"  loss (train) = {loss_train:.4f}")

        if not math.isfinite(loss_train):
            logger.error("  Train-mode loss is NaN but eval was fine.")
            logger.error("  This means SpecAugment or dropout is causing issues.")
            logger.error("  Try reducing mask_time_prob or setting it to 0.")
            raise RuntimeError("Sanity check FAILED in train mode")

        out2.loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        logger.info(f"  grad_norm   = {gn.item():.4f}")
        model.zero_grad(set_to_none=True)

        if has_cuda:
            alloc = torch.cuda.memory_allocated() / 1e9
            logger.info(f"  GPU mem     = {alloc:.2f} GB")

        del batch, out, out2, pred_ids, logits
        if has_cuda: torch.cuda.empty_cache()
        gc.collect()
        logger.info("Sanity check PASSED ✓")
        log_ram()

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
            weight_decay=self.cfg.weight_decay,
            save_total_limit=self.cfg.save_limit,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            report_to=["tensorboard"] if TB else "none",
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            dataloader_persistent_workers=False,
            group_by_length=True,
            remove_unused_columns=True,
            max_grad_norm=1.0,
            use_cpu=self.cfg.force_cpu,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            eval_accumulation_steps=4,
        )

        logger.info(f"TrainingArgs: device={args.device}, fp16={args.fp16}, "
                     f"batch={self.cfg.batch_size}x{self.cfg.grad_accum}, "
                     f"lr={args.learning_rate}, n_gpu={args.n_gpu}")
        log_ram()

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            data_collator=collator,
            compute_metrics=metrics,
            processing_class=proc.feature_extractor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.cfg.patience)],
        )

        ckpt = self.find_ckpt()
        logger.info(f"Resume: {ckpt}" if ckpt else "Fresh start")

        logger.info("Starting Stage 2 training")
        trainer.train(resume_from_checkpoint=ckpt)

        logger.info("Saving …")
        trainer.save_model(self.cfg.output_dir)
        proc.save_pretrained(self.cfg.output_dir)

        m = trainer.evaluate()
        logger.info(f"Final: {m}")
        with open(os.path.join(self.cfg.output_dir, "final_metrics.json"), "w") as f:
            json.dump(m, f, indent=2)

        s1_path = os.path.join(self.cfg.stage1_model, "final_metrics.json")
        if os.path.exists(s1_path):
            with open(s1_path) as f:
                s1 = json.load(f)
            logger.info("=" * 60)
            logger.info("COMPARISON: Stage 1 vs Stage 2")
            logger.info(f"  Stage 1 WER: {s1.get('eval_wer', 0):.4f}   "
                         f"Stage 2 WER: {m['eval_wer']:.4f}")
            logger.info(f"  Stage 1 CER: {s1.get('eval_cer', 0):.4f}   "
                         f"Stage 2 CER: {m['eval_cer']:.4f}")
            if s1.get('eval_wer', 0) > 0:
                imp = (s1['eval_wer'] - m['eval_wer']) / s1['eval_wer'] * 100
                logger.info(f"  WER change: {imp:+.1f}%")
            logger.info("=" * 60)

        logger.info(f"Done → {self.cfg.output_dir}")


def main():
    p = argparse.ArgumentParser(description="Wav2Vec2 CTC fine-tuning — Stage 2")
    p.add_argument("--project_dir",  default=os.path.expanduser("~/asr_project"))
    p.add_argument("--dataset_dir",  default=None)
    p.add_argument("--stage1_model", default=None)
    p.add_argument("--output_dir",   default=None)
    p.add_argument("--batch_size",   type=int,   default=2)
    p.add_argument("--grad_accum",   type=int,   default=16)
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--warmup",       type=int,   default=300)
    p.add_argument("--weight_decay", type=float, default=0.005)
    p.add_argument("--patience",     type=int,   default=8)
    p.add_argument("--eval_steps",   type=int,   default=500)
    p.add_argument("--save_steps",   type=int,   default=500)
    p.add_argument("--no_fp16",      action="store_true")
    p.add_argument("--no_cache",     action="store_true")
    p.add_argument("--force_cpu",    action="store_true")
    a = p.parse_args()

    cfg = ASRConfig(
        project_dir=a.project_dir,
        dataset_dir=a.dataset_dir,
        stage1_model=a.stage1_model,
        output_dir=a.output_dir,
        batch_size=a.batch_size,
        grad_accum=a.grad_accum,
        epochs=a.epochs,
        lr=a.lr,
        warmup=a.warmup,
        weight_decay=a.weight_decay,
        patience=a.patience,
        eval_steps=a.eval_steps,
        save_steps=a.save_steps,
        fp16=not a.no_fp16,
        use_cache=not a.no_cache,
        force_cpu=a.force_cpu,
    )
    ASRTrainerStage2(cfg).train()


if __name__ == "__main__":
    main()