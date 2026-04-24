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
mod bench;
mod checkpoint;
mod dataset;
mod lora;
mod prefetch;
mod qwen2_lora;
mod xentropy;

use anyhow::{Context, Result};
use candle::{DType, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use clap::Parser;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
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

    /// Save a training checkpoint every N steps (0 = disable intermediate
    /// checkpoints; the final adapter is always written at the end).
    /// Checkpoints land in `<output-dir>/checkpoints/step_<N>/`.
    #[arg(long, default_value_t = 0)]
    save_every: usize,

    /// Resume training state from a checkpoint directory (written by an
    /// earlier run's `--save-every`). Loads adapter weights + step counter.
    /// AdamW momentum/variance state is reset — we checkpoint the adapter
    /// weights only, not the optimizer's running statistics.
    #[arg(long)]
    resume_from: Option<PathBuf>,

    /// Enable gradient (activation) checkpointing at the DecoderLayer
    /// boundary. Saves ~10-20× peak activation memory in exchange for
    /// running each layer's forward twice — once during the original
    /// forward, once during the backward recompute. Required to fit any
    /// real config on a 4GB Maxwell card; useful on an 8GB Ampere to
    /// unlock longer sequences or more target modules.
    #[arg(long)]
    gradient_checkpoint: bool,

    /// Sequence-tile size for cross-entropy computation. The full logits
    /// tensor `[B, L, V]` can be 100-300MB at Qwen vocab; tiling along L
    /// means peak = `chunk × V` instead. 0 disables tiling.
    ///
    /// KNOWN ISSUE: values > 0 currently cause a training stall (GPU util
    /// drops to ~8% and no progress is made). The cause is under
    /// investigation — likely candle allocator thrash on the per-chunk
    /// logits alloc/free churn OR a subtle autograd interaction with the
    /// cross-chunk sum accumulator. Default is 0 until this is root-caused.
    #[arg(long, default_value_t = 0)]
    ce_chunk_size: usize,

    /// Bounded queue depth for the background tokenization prefetcher.
    /// Bigger = more overlap headroom, costs a few KB of CPU memory per
    /// slot. Zero disables prefetch and runs tokenization inline.
    ///
    /// KNOWN ISSUE: values > 0 currently cause a training stall when
    /// combined with certain configs. The prefetch thread + main thread
    /// CUDA context interaction is under investigation. Default is 0
    /// until this is root-caused. Inline tokenization is fast enough for
    /// Qwen-length Discord inputs anyway.
    #[arg(long, default_value_t = 0)]
    prefetch_queue: usize,

    /// Run in benchmark mode: 3 warmup + 20 measured optimizer steps with
    /// a fixed pre-tokenized pool of examples. Prints a single-line
    /// BENCH {...} json with tokens/sec, step latency, peak VRAM, and
    /// mean GPU util. Writes no adapter. Intended as the before/after
    /// measurement for every optimization attempt — changes that don't
    /// improve one of these metrics without regressing another aren't
    /// worth landing.
    #[arg(long)]
    benchmark: bool,

    /// Optional label stamped into the BENCH line for easy grep across
    /// runs. Default: empty.
    #[arg(long, default_value = "")]
    bench_label: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;
    // bf16 on CUDA: same memory footprint as fp16, same range as fp32 — no
    // overflow when loading bf16-stored safetensors (Qwen, most HF models).
    // fp32 on CPU for portability + no overhead from fake-bf16 kernels.
    let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };

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
    // Match adapter dtype to base dtype. Earlier attempt used fp32 adapters
    // for optimizer stability, but the to_dtype(base_dtype) cast on the
    // delta tensor appears to break candle's autograd — gradients stopped
    // at the cast and never reached lora_A / lora_B. Keeping everything in
    // base dtype eliminates all casts in the forward path.
    let mut lora_varmap = VarMap::new();
    let vb_lora = VarBuilder::from_varmap(&lora_varmap, dtype, &device);

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

    // --- resume from checkpoint if requested ---
    let mut resume_step = 0usize;
    if let Some(resume_dir) = &args.resume_from {
        let adapter_path = resume_dir.join("adapter_model.safetensors");
        if !adapter_path.exists() {
            anyhow::bail!("resume checkpoint missing adapter_model.safetensors: {adapter_path:?}");
        }
        lora_varmap
            .load(&adapter_path)
            .with_context(|| format!("load adapter from {adapter_path:?}"))?;
        let step_file = resume_dir.join("step.txt");
        if step_file.exists() {
            resume_step = std::fs::read_to_string(&step_file)?
                .trim()
                .parse()
                .context("parse step.txt")?;
        }
        eprintln!(
            "  resumed from   : {} at step {}",
            resume_dir.display(),
            resume_step
        );
    }

    // --- optimizer ---
    let adamw_params = candle_nn::ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };
    let mut optim = candle_nn::AdamW::new(lora_varmap.all_vars(), adamw_params)?;

    // --- training loop ---
    let mut step = resume_step;
    let mut accum_count = 0usize;
    let mut accum_loss = 0f32;
    let total_micro_batches = args.max_steps * args.grad_accum_steps;

    // Share dataset + tokenizer across prefetch worker and (potentially) fallback path.
    let dataset = Arc::new(dataset);
    let tokenizer = Arc::new(tokenizer);

    let prefetcher = if args.prefetch_queue > 0 {
        Some(prefetch::Prefetcher::new(
            dataset.clone(),
            tokenizer.clone(),
            args.max_seq_len,
            args.seed.wrapping_add(resume_step as u64),
            args.prefetch_queue,
        ))
    } else {
        None
    };
    eprintln!(
        "  prefetch       : {}",
        if prefetcher.is_some() {
            format!("on (queue={})", args.prefetch_queue)
        } else {
            "off (inline)".to_string()
        }
    );
    eprintln!("  ce chunk size  : {}", args.ce_chunk_size);

    // ======================================================================
    // BENCHMARK MODE: fixed-protocol measurement, no adapter written.
    //
    // When `--benchmark` is set we short-circuit the training loop, pre-
    // tokenize a small fixed pool of examples, and run 3 warmup + 20
    // measured optimizer steps through the same forward/backward path
    // used by real training. Output is a single BENCH {...} json line
    // which makes before/after comparisons trivial.
    // ======================================================================
    if args.benchmark {
        eprintln!("  mode           : BENCHMARK (3 warmup + 20 measured steps)");
        // Build a pool of N=32 tokenized examples to cycle — larger than
        // grad_accum * measure_steps is unnecessary and wastes tokenize time.
        let mut tok_pool: Vec<dataset::TokenizedExample> = Vec::with_capacity(32);
        let mut fallback_rng_bench = StdRng::seed_from_u64(args.seed);
        let mut fallback_idx: Vec<usize> = (0..dataset.len()).collect();
        fallback_idx.shuffle(&mut fallback_rng_bench);
        for &i in fallback_idx.iter().take(128) {
            let t = dataset::tokenize_pair(&tokenizer, dataset.get(i), args.max_seq_len)?;
            if t.input_ids.len() >= 2 {
                tok_pool.push(t);
                if tok_pool.len() >= 32 {
                    break;
                }
            }
        }
        anyhow::ensure!(!tok_pool.is_empty(), "couldn't build benchmark pool");

        let cfg = bench::BenchConfig {
            warmup_steps: 3,
            measure_steps: 20,
            sample_nvidia_smi: true,
            cfg_label: if args.bench_label.is_empty() {
                format!(
                    "rank={},targets={},seq={},ga={},gc={}",
                    args.rank,
                    args.target_modules,
                    args.max_seq_len,
                    args.grad_accum_steps,
                    args.gradient_checkpoint
                )
            } else {
                args.bench_label.clone()
            },
        };

        let mut pool_cursor = 0usize;
        let result = bench::run(&cfg, |_iter_idx| -> Result<usize> {
            let mut step_tokens = 0usize;
            for _ in 0..args.grad_accum_steps {
                let tok = &tok_pool[pool_cursor % tok_pool.len()];
                pool_cursor += 1;
                let (input_ids, loss_mask) = dataset::batch_to_tensors(
                    std::slice::from_ref(tok),
                    args.pad_token_id,
                    &device,
                )?;
                let (_b, seq_len) = input_ids.dims2()?;
                let shifted_input = input_ids.narrow(1, 0, seq_len - 1)?;
                let shifted_target = input_ids.narrow(1, 1, seq_len - 1)?;
                let shifted_mask = loss_mask.narrow(1, 1, seq_len - 1)?;

                if args.gradient_checkpoint {
                    let mut ctx = checkpoint::CheckpointContext::new();
                    let (hidden, attn_mask) =
                        model.forward_train_with_checkpoint(&shifted_input, &mut ctx)?;
                    let loss = xentropy::tiled_cross_entropy(
                        &hidden,
                        model.lm_head(),
                        &shifted_target,
                        &shifted_mask,
                        args.ce_chunk_size,
                    )?;
                    let scaled_loss = (&loss * (1.0 / args.grad_accum_steps as f64))?;
                    let mut grads = candle::backprop::GradStore::default();
                    scaled_loss.backward_into(&mut grads, None)?;
                    model.backward_through_checkpoints(&ctx, &mut grads, attn_mask.as_ref())?;
                    optim.step(&grads)?;
                } else {
                    let hidden = model.forward_train(&shifted_input)?;
                    let loss = xentropy::tiled_cross_entropy(
                        &hidden,
                        model.lm_head(),
                        &shifted_target,
                        &shifted_mask,
                        args.ce_chunk_size,
                    )?;
                    let scaled_loss = (&loss * (1.0 / args.grad_accum_steps as f64))?;
                    optim.backward_step(&scaled_loss)?;
                }
                step_tokens += tok.input_ids.len().saturating_sub(1);
            }
            Ok(step_tokens)
        })?;

        println!("{}", result.to_single_line());
        return Ok(());
    }


    let mut fallback_rng = StdRng::seed_from_u64(args.seed.wrapping_add(resume_step as u64));
    let mut fallback_indices: Vec<usize> = (0..dataset.len()).collect();
    fallback_indices.shuffle(&mut fallback_rng);
    let mut fallback_cursor = 0usize;

    eprintln!("\n=== training ===");
    'outer: loop {
        loop {
            // Pull next tokenized example: from the prefetch queue when
            // available, else tokenize inline.
            let tok = if let Some(pf) = &prefetcher {
                match pf.next() {
                    Some(Ok(t)) => t,
                    Some(Err(e)) => {
                        eprintln!("prefetch tokenize error: {e:#}");
                        continue;
                    }
                    None => break 'outer, // worker thread exited
                }
            } else {
                if fallback_cursor >= fallback_indices.len() {
                    fallback_indices.shuffle(&mut fallback_rng);
                    fallback_cursor = 0;
                }
                let idx = fallback_indices[fallback_cursor];
                fallback_cursor += 1;
                dataset::tokenize_pair(&tokenizer, dataset.get(idx), args.max_seq_len)?
            };

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

            let (loss, scaled_loss) = if args.gradient_checkpoint {
                let mut ctx = checkpoint::CheckpointContext::new();
                let (hidden, attn_mask) =
                    model.forward_train_with_checkpoint(&shifted_input, &mut ctx)?;
                let loss = xentropy::tiled_cross_entropy(
                    &hidden,
                    model.lm_head(),
                    &shifted_target,
                    &shifted_mask,
                    args.ce_chunk_size,
                )?;
                let scaled_loss = (&loss * (1.0 / args.grad_accum_steps as f64))?;

                // Drive the checkpointed backward: outer backward seeds grad
                // at the post-layer-stack detach point, then each layer's
                // forward is re-run in reverse with its Var grads
                // accumulating into `grads`. Finally we hand `grads` to the
                // optimizer directly instead of going through backward_step.
                let mut grads = candle::backprop::GradStore::default();
                scaled_loss.backward_into(&mut grads, None)?;
                model.backward_through_checkpoints(&ctx, &mut grads, attn_mask.as_ref())?;
                optim.step(&grads)?;
                (loss, scaled_loss)
            } else {
                let hidden = model.forward_train(&shifted_input)?;
                let loss = xentropy::tiled_cross_entropy(
                    &hidden,
                    model.lm_head(),
                    &shifted_target,
                    &shifted_mask,
                    args.ce_chunk_size,
                )?;
                let scaled_loss = (&loss * (1.0 / args.grad_accum_steps as f64))?;
                optim.backward_step(&scaled_loss)?;
                (loss, scaled_loss)
            };
            let _ = scaled_loss;

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

                // Intermediate checkpoint — weights + step counter. We
                // reuse the adapter save format, so a checkpoint dir is
                // just a mid-training adapter that you can point at with
                // --resume-from.
                if args.save_every > 0 && step % args.save_every == 0 {
                    let ckpt_dir = args
                        .output_dir
                        .join("checkpoints")
                        .join(format!("step_{step}"));
                    save_checkpoint(&ckpt_dir, &lora_varmap, &peft_cfg_stub(&args, &targets), step)
                        .with_context(|| format!("save checkpoint to {ckpt_dir:?}"))?;
                    eprintln!("  checkpoint: {}", ckpt_dir.display());
                }

                if step >= args.max_steps {
                    break 'outer;
                }
            }
        }
    }
    drop(prefetcher); // shut worker thread down cleanly before we print final stats

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

/// Build a stub PeftAdapterConfig for mid-training checkpoints, cloned from
/// the CLI args. The stub is adequate for resume (matches production config
/// exactly) and for basic inspection.
fn peft_cfg_stub(args: &Args, targets: &HashSet<qwen2_lora::TargetModule>) -> PeftAdapterConfig {
    let mut names: Vec<String> = targets.iter().map(|t| t.as_str().to_string()).collect();
    names.sort();
    PeftAdapterConfig::new(
        args.base_dir.to_string_lossy().into_owned(),
        args.rank,
        args.alpha,
        args.dropout,
        names,
    )
}

/// Write an intermediate training checkpoint: adapter weights + step.txt.
/// Reuses the PEFT-compatible save layout so you can inspect or resume from
/// any intermediate with the standard tooling.
fn save_checkpoint(
    dir: &std::path::Path,
    varmap: &VarMap,
    cfg: &PeftAdapterConfig,
    step: usize,
) -> Result<()> {
    adapter::save_adapter(dir, cfg, varmap)?;
    std::fs::write(dir.join("step.txt"), step.to_string())?;
    Ok(())
}

/// Masked token-level cross-entropy loss.
/// `logits` is `[B, L, V]`; `target` and `mask` are `[B, L]` with `u8` mask.
///
/// We cast logits to fp32 up front — log_softmax + gather + reductions in
/// fp16 are numerically unstable AND candle's op implementations often
/// require fp32 input. The loss itself is a scalar, cost is negligible.
fn masked_cross_entropy(logits: &Tensor, target: &Tensor, mask: &Tensor) -> candle::Result<Tensor> {
    use candle_nn::ops;
    let logits = if logits.dtype() == DType::F32 {
        logits.clone()
    } else {
        logits.to_dtype(DType::F32)?
    };
    let (_b, _seq_len, vocab) = logits.dims3()?;
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
