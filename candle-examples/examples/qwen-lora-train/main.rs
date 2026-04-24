//! qwen-lora-train — Qwen2.5 LoRA fine-tuning on older CUDA GPUs + CPU.
//!
//! v1 uses an fp16 non-quantized base (candle's QMatMul has no backward pass
//! — see memory/reference_candle_qmatmul_no_backward.md). Base weights load
//! from HuggingFace safetensors (the `--base-dir` argument), LoRA adapters
//! are Vars in a trainable VarMap.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod adapter;
mod dataset;
mod lora;
mod qwen2_lora;

use anyhow::{Context, Result};
use candle::{DType, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use clap::Parser;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use adapter::PeftAdapterConfig;
use dataset::Dataset;
use lora::LoRAConfig;
use qwen2_lora::{parse_target_modules, Model};

#[derive(Parser, Debug)]
#[command(author, version, about = "Qwen2.5 LoRA fine-tune (older-GPU friendly)", long_about = None)]
struct Args {
    /// Directory containing safetensors base weights + config.json + tokenizer.json.
    /// Expected layout matches HF: {base_dir}/config.json, {base_dir}/tokenizer.json,
    /// {base_dir}/model.safetensors (or model-*.safetensors shards + .index.json).
    #[arg(long)]
    base_dir: PathBuf,

    /// Path to training JSONL in matt-voice corpus format.
    #[arg(long)]
    dataset: PathBuf,

    /// Where to write adapter_config.json + adapter_model.safetensors.
    #[arg(long)]
    output_dir: PathBuf,

    /// LoRA rank.
    #[arg(long, default_value_t = 16)]
    rank: usize,

    /// LoRA alpha (scaling numerator; effective scale is alpha / rank).
    #[arg(long, default_value_t = 32.0)]
    alpha: f32,

    /// LoRA dropout applied on adapter input during training.
    #[arg(long, default_value_t = 0.05)]
    dropout: f32,

    /// Comma-separated projection names to attach LoRA adapters to.
    #[arg(long, default_value = "q_proj,k_proj,v_proj,o_proj")]
    target_modules: String,

    /// Micro-batch size.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// Gradient accumulation steps.
    #[arg(long, default_value_t = 16)]
    grad_accum_steps: usize,

    /// Peak learning rate for AdamW.
    #[arg(long, default_value_t = 2e-4)]
    learning_rate: f64,

    /// Total optimizer steps.
    #[arg(long, default_value_t = 500)]
    max_steps: usize,

    /// Max sequence length (right-truncated).
    #[arg(long, default_value_t = 1024)]
    max_seq_len: usize,

    /// Pad token ID (Qwen2 default: 151643).
    #[arg(long, default_value_t = 151643)]
    pad_token_id: u32,

    /// Force CPU even if CUDA is available.
    #[arg(long)]
    cpu: bool,

    /// Random seed.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Log training metrics every N steps.
    #[arg(long, default_value_t = 10)]
    log_every: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };

    eprintln!("qwen-lora-train v1 (fp16 base + fp32 adapters)");
    eprintln!("  device         : {:?}", device);
    eprintln!("  dtype (base)   : {:?}", dtype);
    eprintln!("  base dir       : {}", args.base_dir.display());
    eprintln!("  dataset        : {}", args.dataset.display());
    eprintln!("  output dir     : {}", args.output_dir.display());
    eprintln!(
        "  LoRA r/α/drop  : {} / {} / {}",
        args.rank, args.alpha, args.dropout
    );
    eprintln!("  target modules : {}", args.target_modules);
    eprintln!(
        "  batch×accum=eff: {}×{}={}  lr={:.2e}  max_steps={}",
        args.batch_size,
        args.grad_accum_steps,
        args.batch_size * args.grad_accum_steps,
        args.learning_rate,
        args.max_steps
    );

    // --- tokenizer + config ---
    let tokenizer_path = args.base_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("load tokenizer from {tokenizer_path:?}"))?;

    let config_path = args.base_dir.join("config.json");
    let config_str =
        std::fs::read_to_string(&config_path).with_context(|| format!("read {config_path:?}"))?;
    let config: qwen2_lora::Config =
        serde_json::from_str(&config_str).context("parse config.json")?;
    eprintln!(
        "  model          : {} layers, hidden={} heads={} kv_heads={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads
    );

    // --- dataset ---
    let dataset = Dataset::load_jsonl(&args.dataset).context("load dataset")?;
    eprintln!("  dataset        : {} pairs", dataset.len());
    anyhow::ensure!(!dataset.is_empty(), "dataset is empty");

    // --- base VarBuilder (frozen, mmap'd safetensors) ---
    let safetensor_files = discover_safetensor_shards(&args.base_dir)?;
    eprintln!("  base shards    : {}", safetensor_files.len());
    let vb_base = unsafe {
        VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, &device)?
    };

    // --- LoRA VarBuilder (trainable, fresh VarMap) ---
    // Adapters stay fp32 even when base is fp16 — more stable optimizer math,
    // and the VRAM cost is tiny (thousands of params per layer, not millions).
    let lora_varmap = VarMap::new();
    let vb_lora = VarBuilder::from_varmap(&lora_varmap, DType::F32, &device);

    // --- build model ---
    let targets = parse_target_modules(&args.target_modules)?;
    let lora_cfg = LoRAConfig {
        rank: args.rank,
        alpha: args.alpha,
        dropout: args.dropout,
    };
    eprintln!(
        "  LoRA adapters  : {} projection types × {} layers = {} total",
        targets.len(),
        config.num_hidden_layers,
        targets.len() * config.num_hidden_layers
    );

    let mut model = Model::new(&config, &targets, &lora_cfg, vb_base, vb_lora)?;
    let num_trainable_params: usize = lora_varmap
        .all_vars()
        .iter()
        .map(|v| v.elem_count())
        .sum();
    eprintln!(
        "  trainable params: {} ({:.2} MB @ fp32)",
        num_trainable_params,
        (num_trainable_params * 4) as f64 / (1024.0 * 1024.0)
    );

    // --- optimizer ---
    let adamw_params = candle_nn::ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };
    let mut optim = candle_nn::AdamW::new(lora_varmap.all_vars(), adamw_params)?;

    // --- training loop ---
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut step = 0usize;
    let mut accum_count = 0usize;
    let mut accum_loss = 0f32;
    let total_micro_batches = args.max_steps * args.grad_accum_steps;

    eprintln!("\n=== training ===");
    'outer: for _epoch in 0..usize::MAX {
        // simple uniform sampling w/ replacement — keep it simple for v1
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        indices.shuffle(&mut rng);

        for &i in indices.iter() {
            let pair = dataset.get(i);
            let tok = dataset::tokenize_pair(&tokenizer, pair, args.max_seq_len)?;
            if tok.input_ids.len() < 2 {
                continue; // nothing to predict
            }

            let (input_ids, loss_mask) = dataset::batch_to_tensors(
                std::slice::from_ref(&tok),
                args.pad_token_id,
                &device,
            )?;

            // Shift: predict token at position t+1 from position t.
            let (_b, seq_len) = input_ids.dims2()?;
            let shifted_input = input_ids.narrow(1, 0, seq_len - 1)?;
            let shifted_target = input_ids.narrow(1, 1, seq_len - 1)?;
            let shifted_mask = loss_mask.narrow(1, 1, seq_len - 1)?;

            let logits = model.forward_train(&shifted_input)?;
            let loss = masked_cross_entropy(&logits, &shifted_target, &shifted_mask)?;

            // Gradient accumulation: scale loss so the step-level total is
            // roughly independent of grad_accum_steps.
            let scaled_loss = (&loss * (1.0 / args.grad_accum_steps as f64))?;
            optim.backward_step(&scaled_loss)?;

            accum_count += 1;
            accum_loss += loss.to_scalar::<f32>()?;

            if accum_count >= args.grad_accum_steps {
                step += 1;
                let avg_loss = accum_loss / accum_count as f32;
                if step % args.log_every == 0 || step == 1 {
                    eprintln!(
                        "step {:5}/{:<5}  loss {:.4}  (micro-batches processed: {}/{})",
                        step,
                        args.max_steps,
                        avg_loss,
                        step * args.grad_accum_steps,
                        total_micro_batches
                    );
                }
                accum_count = 0;
                accum_loss = 0.0;
                if step >= args.max_steps {
                    break 'outer;
                }
            }
        }
    }

    eprintln!("\n=== saving adapter ===");
    let targets_sorted: Vec<String> = {
        let mut v: Vec<String> = targets
            .iter()
            .map(|t| t.as_str().to_string())
            .collect();
        v.sort();
        v
    };
    let peft_cfg = PeftAdapterConfig::new(
        args.base_dir.to_string_lossy().into_owned(),
        args.rank,
        args.alpha,
        args.dropout,
        targets_sorted,
    );
    adapter::save_adapter(&args.output_dir, &peft_cfg, &lora_varmap)?;
    eprintln!("adapter saved to {}", args.output_dir.display());

    Ok(())
}

/// Masked token-level cross-entropy loss.
/// `logits` is `[B, L, V]`; `target` and `mask` are `[B, L]` with `u8` mask.
fn masked_cross_entropy(logits: &Tensor, target: &Tensor, mask: &Tensor) -> candle::Result<Tensor> {
    use candle_nn::ops;
    let (_b, seq_len, vocab) = logits.dims3()?;
    let logits_2d = logits.reshape(((), vocab))?;
    let target_1d = target.reshape(((),))?;
    let log_probs = ops::log_softmax(&logits_2d, 1)?;
    // gather log_probs at target index — shape [B*L, 1] -> [B*L]
    let target_idx = target_1d.unsqueeze(1)?;
    let picked = log_probs.gather(&target_idx, 1)?.squeeze(1)?;
    // mask + mean over loss positions only
    let mask_1d = mask.reshape(((),))?.to_dtype(picked.dtype())?;
    let num = (picked * &mask_1d)?.sum_all()?;
    let denom = mask_1d.sum_all()?;
    let denom_f = denom.to_scalar::<f32>()?.max(1.0) as f64;
    let nll = (num.affine(-1.0, 0.0)? / denom_f)?;
    let _ = seq_len;
    Ok(nll)
}

/// Discover all safetensors shards in `dir`. Prefers the sharded layout
/// (model-*.safetensors) if an index is present, else falls back to a single
/// model.safetensors file.
///
/// Zip-slip-equivalent hardening: shard filenames come from a JSON index
/// file, which we treat as potentially untrusted. We (a) reject any entry
/// that looks like a path-traversal attempt, (b) strip to the bare filename,
/// (c) canonicalize and assert the result descends from `dir`.
fn discover_safetensor_shards(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    let single = dir.join("model.safetensors");
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let idx_str = std::fs::read_to_string(&index)?;
        let idx: serde_json::Value = serde_json::from_str(&idx_str)?;
        let map = idx
            .get("weight_map")
            .and_then(|v| v.as_object())
            .context("no weight_map in index")?;
        let mut files: std::collections::BTreeSet<PathBuf> = Default::default();
        for fname in map.values() {
            let Some(raw) = fname.as_str() else { continue };
            // Reject anything suspicious up front. Valid shard names are of
            // the form `model-00001-of-00004.safetensors` — no separators,
            // no parent-refs, no null bytes.
            if raw.contains("..")
                || raw.contains('/')
                || raw.contains('\\')
                || raw.contains('\0')
                || !raw.ends_with(".safetensors")
            {
                anyhow::bail!("rejected suspicious shard name in index: {raw:?}");
            }
            let bare = std::path::Path::new(raw)
                .file_name()
                .context("shard entry has no file_name")?;
            let candidate = dir.join(bare);
            let validated = ensure_within_dir(dir, &candidate)
                .with_context(|| format!("shard {raw:?} failed containment check"))?;
            files.insert(validated);
        }
        Ok(files.into_iter().collect())
    } else if single.exists() {
        Ok(vec![single])
    } else {
        anyhow::bail!(
            "no safetensors found in {dir:?}: expected model.safetensors or model.safetensors.index.json"
        );
    }
}

/// Confirm `candidate` resolves to a descendant of `base` after canonicaliz-
/// ation, rejecting `..` traversal even if earlier sanitization missed it.
fn ensure_within_dir(base: &std::path::Path, candidate: &std::path::Path) -> Result<PathBuf> {
    let base_abs = base
        .canonicalize()
        .with_context(|| format!("canonicalize base {base:?}"))?;
    let parent = candidate
        .parent()
        .context("candidate has no parent")?;
    let file = candidate
        .file_name()
        .context("candidate has no file_name")?;
    let parent_abs = parent
        .canonicalize()
        .with_context(|| format!("canonicalize parent {parent:?}"))?;
    if !parent_abs.starts_with(&base_abs) {
        anyhow::bail!(
            "path traversal refused: {parent_abs:?} escapes base {base_abs:?}"
        );
    }
    Ok(parent_abs.join(file))
}
