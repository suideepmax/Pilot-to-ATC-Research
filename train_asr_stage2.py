"""
Wav2Vec2 CTC Stage 2 fine-tuning on UWB-ATCC.
Loads Stage 1 model and applies additional regularization with SpecAugment.
    python train_asr_stage2.py --no_cache
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
    get_cosine_schedule_with_warmup,
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

try:
    import psutil; HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ── RAM monitoring ────────────────────────────────────────────────────
def log_ram_usage(msg=""):
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        used_gb = mem.used / 1e9
        total_gb = mem.total / 1e9
        pct = mem.percent
        logger.info(f"RAM {msg}: {used_gb:.2f}/{total_gb:.2f} GB ({pct:.1f}%)")


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
    stage1_model: Optional[str] = None
    output_dir:  Optional[str] = None
    target_sr: int     = 16000
    max_audio_len: float = 20.0
    min_audio_len: float = 0.3
    freeze_feature_encoder: bool = True
    gradient_checkpointing: bool = True
    batch_size: int    = 2
    grad_accum: int    = 16
    epochs: int        = 20
    lr: float          = 5e-5
    warmup: int        = 500
    weight_decay: float = 0.005
    patience: int      = 8
    fp16: bool         = True
    eval_steps: int    = 500
    save_steps: int    = 500
    log_steps: int     = 50
    save_limit: int    = 3
    use_cache: bool    = True
    force_cpu: bool    = False
    # SpecAugment for Stage 2
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    # Dropout for Stage 2
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layerdrop: float = 0.05

    def __post_init__(self):
        self.dataset_dir = self.dataset_dir or os.path.join(self.project_dir, "data/uwb_atcc/manifests")
        self.stage1_model = self.stage1_model or os.path.join(self.project_dir, "models/wav2vec2_uwb_atcc")
        self.output_dir  = self.output_dir  or os.path.join(self.project_dir, "models/wav2vec2_uwb_atcc_v2")
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


# ── audio (RAM-optimized) ─────────────────────────────────────────────
class AudioLoader:
    """RAM-optimized loader that processes files one at a time during featurization."""
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        self.fails: list = []

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
        log_ram_usage("before preprocessing")
        
        # Normalize text first (no audio loading)
        ds = ds.map(lambda b: {"transcript": self.norm(b["transcript"])},
                     num_proc=1, load_from_cache_file=self.cfg.use_cache, desc="Norm")
        
        for s in ds:
            n = len(ds[s])
            ds[s] = ds[s].filter(lambda x: len(x["transcript"]) > 0, desc=f"empty-{s}")
            d = n - len(ds[s])
            if d: logger.warning(f"Dropped {d} empty from {s}")

        logger.info(f"Clean text: train={len(ds['train'])} val={len(ds['validation'])}")
        log_ram_usage("after text norm")
        return ds

    def load_stage1_processor(self):
        """Load processor from Stage 1 model (reuses vocab.json)"""
        logger.info(f"Loading processor from {self.cfg.stage1_model}")
        proc = Wav2Vec2Processor.from_pretrained(self.cfg.stage1_model)
        
        # Verify vocab consistency
        logger.info(f"Tokenizer vocab_size: {proc.tokenizer.vocab_size}")
        logger.info(f"len(tokenizer): {len(proc.tokenizer)}")
        logger.info(f"PAD id: {proc.tokenizer.pad_token_id}")
        logger.info(f"UNK id: {proc.tokenizer.unk_token_id}")
        
        return proc

    def featurize(self, ds, proc):
        """RAM-optimized: load audio one-at-a-time during featurization with frequent disk writes"""
        log_ram_usage("before featurization")
        
        def fn(b):
            # Load audio on-the-fly (not pre-loaded)
            a = self.loader(b)
            if not a["valid"]:
                # Return invalid sample (will be filtered later)
                b["input_values"] = np.zeros(self.cfg.min_samples, dtype=np.float32)
                b["labels"] = []
                b["_alen"] = 0
                b["_llen"] = 0
                b["valid"] = False
            else:
                audio = a["audio"]
                b["input_values"] = proc(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
                b["labels"] = proc.tokenizer(b["transcript"]).input_ids
                b["_alen"] = len(b["input_values"])
                b["_llen"] = len(b["labels"])
                b["valid"] = True
            return b

        # Process with writer_batch_size to flush to disk frequently (RAM saving)
        drop = [c for c in ds["train"].column_names if c not in ("input_values","labels","_alen","_llen","valid")]
        ds = ds.map(fn, remove_columns=drop, num_proc=1,
                     load_from_cache_file=self.cfg.use_cache, 
                     writer_batch_size=500,
                     desc="Features")
        
        log_ram_usage("after featurization")

        # Filter invalid samples
        for s in ds:
            n = len(ds[s])
            ds[s] = ds[s].filter(lambda x: x["valid"], desc=f"valid-{s}")
            d = n - len(ds[s])
            if d: logger.warning(f"Dropped {d} bad audio from {s}")

        # CTC filter
        for s in ds:
            n = len(ds[s])
            ds[s] = ds[s].filter(lambda x: ctc_ok(x["_alen"], x["_llen"]), desc=f"ctc-{s}")
            d = n - len(ds[s])
            if d: logger.warning(f"CTC filter dropped {d} from {s}")

        ds = ds.remove_columns(["_alen", "_llen", "valid"])
        logger.info(f"Final: train={len(ds['train'])} val={len(ds['validation'])}")
        log_ram_usage("after filtering")
        return ds

    def load_stage1_model(self, proc, device):
        """Load Stage 1 model and apply Stage 2 configuration"""
        logger.info(f"Loading Stage 1 model from {self.cfg.stage1_model}")
        
        model = Wav2Vec2ForCTC.from_pretrained(self.cfg.stage1_model)
        
        # Apply Stage 2 dropout configuration
        logger.info("Applying Stage 2 regularization config:")
        model.config.attention_dropout = self.cfg.attention_dropout
        model.config.hidden_dropout = self.cfg.hidden_dropout
        model.config.layerdrop = self.cfg.layerdrop
        logger.info(f"  attention_dropout = {self.cfg.attention_dropout}")
        logger.info(f"  hidden_dropout = {self.cfg.hidden_dropout}")
        logger.info(f"  layerdrop = {self.cfg.layerdrop}")
        
        # Enable SpecAugment for Stage 2
        model.config.mask_time_prob = self.cfg.mask_time_prob
        model.config.mask_time_length = self.cfg.mask_time_length
        model.config.mask_feature_prob = 0.0
        model.config.apply_spec_augment = True
        logger.info(f"  mask_time_prob = {self.cfg.mask_time_prob}")
        logger.info(f"  mask_time_length = {self.cfg.mask_time_length}")
        
        # CRITICAL FIX: Create masked_spec_embed if missing
        if self.cfg.mask_time_prob > 0:
            if hasattr(model.wav2vec2, 'masked_spec_embed') and \
               model.wav2vec2.masked_spec_embed is not None:
                logger.info("masked_spec_embed exists — reinitializing to zeros")
                nn.init.zeros_(model.wav2vec2.masked_spec_embed)
                model.wav2vec2.masked_spec_embed.requires_grad = True
            else:
                hidden_size = model.config.hidden_size
                logger.info(f"masked_spec_embed missing — creating new parameter (size={hidden_size})")
                model.wav2vec2.masked_spec_embed = nn.Parameter(
                    torch.zeros(hidden_size, dtype=torch.float32)
                )
        
        model = model.to(device)
        logger.info(f"Model device: {next(model.parameters()).device}")
        
        logger.info(f"  config.pad_token_id       = {model.config.pad_token_id}")
        logger.info(f"  config.vocab_size         = {model.config.vocab_size}")
        logger.info(f"  config.ctc_zero_infinity  = {model.config.ctc_zero_infinity}")
        logger.info(f"  config.ctc_loss_reduction = {model.config.ctc_loss_reduction}")
        
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

    def evaluate_stage1(self, proc, device):
        """Evaluate Stage 1 model for comparison"""
        logger.info("=" * 60)
        logger.info("Evaluating Stage 1 model for comparison")
        logger.info("=" * 60)
        
        try:
            # Load dataset and featurize
            ds = self.load_ds()
            ds = self.preprocess(ds)
            ds = self.featurize(ds, proc)
            
            # Load Stage 1 model (without Stage 2 modifications)
            model_s1 = Wav2Vec2ForCTC.from_pretrained(self.cfg.stage1_model)
            model_s1 = model_s1.to(device)
            
            collator = CTCCollator(processor=proc)
            metrics = Metrics(proc)
            
            # Create trainer for evaluation only
            args = TrainingArguments(
                output_dir=self.cfg.output_dir,
                per_device_eval_batch_size=self.cfg.batch_size,
                use_cpu=self.cfg.force_cpu,
                dataloader_num_workers=0,
                eval_accumulation_steps=4,
            )
            
            trainer = Trainer(
                model=model_s1,
                args=args,
                eval_dataset=ds["validation"],
                data_collator=collator,
                compute_metrics=metrics,
                processing_class=proc.feature_extractor,
            )
            
            m = trainer.evaluate()
            logger.info(f"Stage 1 metrics: WER={m['eval_wer']:.4f}, CER={m['eval_cer']:.4f}")
            
            return m
        except Exception as e:
            logger.warning(f"Failed to evaluate Stage 1 model: {e}")
            return None

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

    def train(self):
        logger.info("=" * 60)
        logger.info("ASR Training — Stage 2")
        logger.info("=" * 60)
        log_ram_usage("at start")

        device, has_cuda = get_device()
        if self.cfg.force_cpu:
            device, has_cuda = torch.device("cpu"), False
            logger.info("Forced CPU mode")

        use_fp16 = self.cfg.fp16 and has_cuda
        if self.cfg.fp16 and not has_cuda:
            logger.warning("fp16 requires CUDA → disabled")

        ds   = self.load_ds()
        ds   = self.preprocess(ds)
        proc = self.load_stage1_processor()
        ds   = self.featurize(ds, proc)
        model = self.load_stage1_model(proc, device)

        collator = CTCCollator(processor=proc)
        metrics  = Metrics(proc)

        # ── sanity check ─────────────────────────────────────────────
        logger.info("Sanity check …")
        n_sample = min(4, len(ds["train"]))
        batch = collator([ds["train"][i] for i in range(n_sample)])
        batch = {k: v.to(device) for k, v in batch.items()}

        # Test eval mode (SpecAugment disabled in eval)
        model.eval()
        with torch.no_grad():
            try:
                out = model(input_values=batch["input_values"], labels=batch["labels"])
                loss_eval = out.loss.item()
                logger.info(f"  loss (eval)  = {loss_eval:.4f}  (device={batch['input_values'].device})")
                
                # Check predictions
                logits = out.logits
                pred_ids = torch.argmax(logits, dim=-1)
                unique_preds = torch.unique(pred_ids).tolist()
                logger.info(f"  unique predicted ids = {unique_preds}")
                logger.info(f"  PAD/blank id = {proc.tokenizer.pad_token_id}")
                
                if not math.isfinite(loss_eval):
                    raise RuntimeError("Sanity check FAILED — eval loss is NaN/Inf")
            except Exception as e:
                logger.error(f"Sanity check FAILED in eval mode: {e}")
                logger.error(f"Model config: mask_time_prob={model.config.mask_time_prob}")
                logger.error(f"Has masked_spec_embed: {hasattr(model.wav2vec2, 'masked_spec_embed')}")
                if hasattr(model.wav2vec2, 'masked_spec_embed'):
                    logger.error(f"masked_spec_embed is None: {model.wav2vec2.masked_spec_embed is None}")
                raise

        # Test train mode (SpecAugment enabled in train)
        model.train()
        try:
            out2 = model(input_values=batch["input_values"], labels=batch["labels"])
            loss_train = out2.loss.item()
            logger.info(f"  loss (train) = {loss_train:.4f}")
            
            if not math.isfinite(loss_train):
                raise RuntimeError("Sanity check FAILED — train loss is NaN/Inf")
            
            out2.loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            logger.info(f"  grad_norm   = {gn.item():.4f}")
            model.zero_grad()
        except Exception as e:
            logger.error(f"Sanity check FAILED in train mode: {e}")
            logger.error(f"Model config: mask_time_prob={model.config.mask_time_prob}")
            logger.error(f"Has masked_spec_embed: {hasattr(model.wav2vec2, 'masked_spec_embed')}")
            if hasattr(model.wav2vec2, 'masked_spec_embed'):
                logger.error(f"masked_spec_embed is None: {model.wav2vec2.masked_spec_embed is None}")
                logger.error(f"masked_spec_embed shape: {model.wav2vec2.masked_spec_embed.shape if model.wav2vec2.masked_spec_embed is not None else 'N/A'}")
            raise

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
            weight_decay=self.cfg.weight_decay,
            save_total_limit=self.cfg.save_limit,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            report_to=["tensorboard"] if TB else "none",
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            group_by_length=True,
            remove_unused_columns=True,
            max_grad_norm=1.0,
            use_cpu=self.cfg.force_cpu,
            eval_accumulation_steps=4,
            lr_scheduler_type="cosine",
        )

        logger.info(f"TrainingArgs: device={args.device}, fp16={args.fp16}, n_gpu={args.n_gpu}")
        logger.info(f"LR scheduler: cosine, weight_decay={self.cfg.weight_decay}")

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
        trainer.train(resume_from_checkpoint=ckpt)

        logger.info("Saving …")
        trainer.save_model(self.cfg.output_dir)
        proc.save_pretrained(self.cfg.output_dir)

        m_stage2 = trainer.evaluate()
        logger.info(f"Stage 2 Final: {m_stage2}")
        with open(os.path.join(self.cfg.output_dir, "final_metrics.json"), "w") as f:
            json.dump(m_stage2, f, indent=2)

        # ── Stage 1 vs Stage 2 comparison ────────────────────────────
        logger.info("=" * 60)
        logger.info("Stage 1 vs Stage 2 Comparison")
        logger.info("=" * 60)
        
        # Try to load Stage 1 metrics from saved file
        s1_metrics_path = os.path.join(self.cfg.stage1_model, "final_metrics.json")
        if os.path.exists(s1_metrics_path):
            with open(s1_metrics_path, "r") as f:
                m_stage1 = json.load(f)
            
            wer_s1 = m_stage1.get("eval_wer", 0)
            cer_s1 = m_stage1.get("eval_cer", 0)
            wer_s2 = m_stage2.get("eval_wer", 0)
            cer_s2 = m_stage2.get("eval_cer", 0)
            
            wer_change = ((wer_s2 - wer_s1) / wer_s1 * 100) if wer_s1 > 0 else 0
            cer_change = ((cer_s2 - cer_s1) / cer_s1 * 100) if cer_s1 > 0 else 0
            
            logger.info(f"Stage 1 WER: {wer_s1:.4f} → Stage 2 WER: {wer_s2:.4f} ({wer_change:+.1f}%)")
            logger.info(f"Stage 1 CER: {cer_s1:.4f} → Stage 2 CER: {cer_s2:.4f} ({cer_change:+.1f}%)")
            
            comparison = {
                "stage1": {"wer": wer_s1, "cer": cer_s1},
                "stage2": {"wer": wer_s2, "cer": cer_s2},
                "improvement": {
                    "wer_change_pct": wer_change,
                    "cer_change_pct": cer_change,
                }
            }
            with open(os.path.join(self.cfg.output_dir, "stage_comparison.json"), "w") as f:
                json.dump(comparison, f, indent=2)
        else:
            logger.warning(f"Stage 1 metrics not found at {s1_metrics_path}")
            logger.info(f"Stage 2 WER: {m_stage2.get('eval_wer', 0):.4f}")
            logger.info(f"Stage 2 CER: {m_stage2.get('eval_cer', 0):.4f}")

        logger.info("=" * 60)
        logger.info(f"Done → {self.cfg.output_dir}")
        logger.info("=" * 60)
        log_ram_usage("at end")


# ── CLI ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project_dir", default=os.path.expanduser("~/asr_project"))
    p.add_argument("--dataset_dir", default=None)
    p.add_argument("--stage1_model", default=None)
    p.add_argument("--output_dir",  default=None)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--grad_accum",  type=int,   default=16)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--warmup",      type=int,   default=500)
    p.add_argument("--weight_decay", type=float, default=0.005)
    p.add_argument("--patience",    type=int,   default=8)
    p.add_argument("--eval_steps",  type=int,   default=500)
    p.add_argument("--save_steps",  type=int,   default=500)
    p.add_argument("--no_fp16",                 action="store_true")
    p.add_argument("--no_cache",                action="store_true")
    p.add_argument("--force_cpu",               action="store_true")
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
    ASRTrainer(cfg).train()


if __name__ == "__main__":
    main()
