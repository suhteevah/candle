//! qwen-lora-train — Qwen2.5 LoRA fine-tuning on older CUDA GPUs + CPU.
//!
//! Scaffold stage: CLI, dataset loader, LoRA module, and adapter exporter are
//! in place. The base-model-with-LoRA-taps module (`qwen2_lora.rs`) is the
//! critical missing piece — until it lands the trainer here aborts early with
//! a "not yet implemented" banner. See README.md for the plan.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod adapter;
mod dataset;
mod lora;
// mod qwen2_lora;  // TODO: fork quantized_qwen2 with LoRA taps

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use adapter::PeftAdapterConfig;
use dataset::Dataset;

#[derive(Parser, Debug)]
#[command(author, version, about = "Qwen2.5 LoRA fine-tune (older-GPU friendly)", long_about = None)]
struct Args {
    /// Path to Qwen2.5 GGUF base model (Q4_K_M or Q5_0 recommended).
    #[arg(long)]
    base_model: PathBuf,

    /// Path to tokenizer.json (usually bundled with the HF repo for the base).
    #[arg(long)]
    tokenizer: PathBuf,

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

    /// Micro-batch size (per forward pass).
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// Number of micro-batches accumulated before an optimizer step.
    #[arg(long, default_value_t = 16)]
    grad_accum_steps: usize,

    /// Peak learning rate for AdamW.
    #[arg(long, default_value_t = 2e-4)]
    learning_rate: f64,

    /// Total optimizer steps (not micro-batches).
    #[arg(long, default_value_t = 500)]
    max_steps: usize,

    /// Sequence length for training examples (right-truncated).
    #[arg(long, default_value_t = 1024)]
    max_seq_len: usize,

    /// Pad token ID for batching (Qwen2 default: 151643).
    #[arg(long, default_value_t = 151643)]
    pad_token_id: u32,

    /// Force CPU even if CUDA is available.
    #[arg(long)]
    cpu: bool,

    /// Random seed (for both dataset shuffle and optimizer init).
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("qwen-lora-train — scaffold build");
    eprintln!("  base_model     : {}", args.base_model.display());
    eprintln!("  tokenizer      : {}", args.tokenizer.display());
    eprintln!("  dataset        : {}", args.dataset.display());
    eprintln!("  output_dir     : {}", args.output_dir.display());
    eprintln!(
        "  rank/alpha/drop: {} / {} / {}",
        args.rank, args.alpha, args.dropout
    );
    eprintln!("  target modules : {}", args.target_modules);
    eprintln!(
        "  batch/accum/lr : {} * {} = {} effective; lr={:.2e}",
        args.batch_size,
        args.grad_accum_steps,
        args.batch_size * args.grad_accum_steps,
        args.learning_rate
    );

    // Load tokenizer — validated early so we fail fast on bad paths.
    let tokenizer =
        Tokenizer::from_file(&args.tokenizer).map_err(anyhow::Error::msg)
            .context("load tokenizer")?;
    eprintln!("  tokenizer vocab: {}", tokenizer.get_vocab_size(true));

    // Load dataset and tokenize one example end-to-end to validate the chain.
    let dataset = Dataset::load_jsonl(&args.dataset).context("load dataset")?;
    eprintln!("  dataset pairs  : {}", dataset.len());
    if dataset.is_empty() {
        anyhow::bail!("dataset is empty");
    }

    let first = dataset.get(0);
    let tok = dataset::tokenize_pair(&tokenizer, first, args.max_seq_len)
        .context("tokenize first pair")?;
    eprintln!(
        "  first example  : {} tokens, {} loss-mask positions",
        tok.input_ids.len(),
        tok.loss_mask.iter().filter(|&&b| b == 1).count()
    );

    // Parse target modules just to validate the format.
    let targets: Vec<String> = args
        .target_modules
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // Scaffold-stage exit: we've validated CLI + dataset + tokenizer path +
    // adapter-config construction. The actual training loop requires
    // `qwen2_lora.rs`, which is still TODO.
    let cfg = PeftAdapterConfig::new(
        args.base_model.to_string_lossy().into_owned(),
        args.rank,
        args.alpha,
        args.dropout,
        targets,
    );
    eprintln!(
        "  adapter config : {}",
        serde_json::to_string(&cfg).unwrap_or_default()
    );

    eprintln!();
    eprintln!(
        "scaffold OK — validated all plumbing except the base-model-with-LoRA \
         forward. Next step: implement qwen2_lora.rs. See README.md."
    );
    eprintln!();

    Ok(())
}
