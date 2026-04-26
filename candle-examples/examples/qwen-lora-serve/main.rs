//! qwen-lora-serve — inference for Qwen2 GGUF + LoRA adapter trained by qwen-lora-train.
//!
//! Two modes:
//!
//!   * `--prompt "..."`  — one-shot CLI generation, prints output to stdout, exits.
//!   * (default)         — HTTP server. POST `/generate` with `{"prompt": "...",
//!                         "max_tokens": N, "temperature": T}` returns `{"text": "..."}`.
//!
//! The model graph is `qwen2_lora_quantized::Model` from the training crate
//! (shared via `#[path]`). At startup we build the model from the GGUF, then
//! call `VarMap::load(adapter_model.safetensors)` to pull the trained adapter
//! weights into the freshly-initialized A/B Vars. Inference uses the existing
//! `Model::forward(x, index_pos)` which is already kv-cache-aware.
//!
//! Build:
//!   cargo build --release --example qwen-lora-serve --features cuda
//!
//! Run (server):
//!   qwen-lora-serve --gguf model.gguf --tokenizer tokenizer.json \
//!       --adapter J:/path/to/adapter_dir --port 8080
//!
//! Run (CLI):
//!   qwen-lora-serve --gguf ... --tokenizer ... --adapter ... \
//!       --prompt "hey claude, what's up" --max-tokens 128

#[path = "../qwen-lora-train/lora.rs"]
mod lora;
#[path = "../qwen-lora-train/fused_ops.rs"]
mod fused_ops;
#[path = "../qwen-lora-train/checkpoint.rs"]
mod checkpoint;
#[path = "../qwen-lora-train/qwen2_lora.rs"]
mod qwen2_lora;
#[path = "../qwen-lora-train/qwen2_lora_quantized.rs"]
mod qwen2_lora_quantized;

use anyhow::{Context, Result};
use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::post,
    Json, Router,
};
use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use clap::Parser;
use futures::stream::Stream;
use minijinja::{context, Environment};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;

use lora::LoRAConfig;
use qwen2_lora::TargetModule;
use qwen2_lora_quantized::Model;

/// Result of `Engine::bench_decode`. Single-line BENCH_INFER JSON output
/// is built from this — kept stable so bench/run-bench-infer.ps1 can
/// regex it out reliably.
#[derive(Debug, Serialize)]
struct BenchInferResult {
    /// Tokens per second AFTER the first sampled token (excludes prefill).
    decode_tok_per_sec: f64,
    /// Wall ms from start to first sampled token (includes prefill).
    first_token_ms: f64,
    /// Wall ms for the prefill forward (full prompt at index_pos=0).
    prefill_ms: f64,
    /// Wall ms for the decode loop (n_tokens-1 forwards).
    decode_ms: f64,
    /// Total wall ms.
    total_ms: f64,
    /// Number of generated tokens (matches `--benchmark-decode N`).
    n_tokens: usize,
    /// Prompt length (input tokens before generation).
    prompt_len: usize,
    /// Peak VRAM during the run, in MB. None if nvidia-smi unavailable.
    peak_vram_mb: Option<u64>,
}

/// Best-effort peak VRAM from nvidia-smi. Single shot — caller is
/// responsible for taking the max across multiple samples if they want
/// a true peak. We use a single sample at end-of-run; the steady-state
/// VRAM for an inference workload is largely flat after warm-up.
fn peak_vram_mb_via_nvidia_smi() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8(output.stdout).ok()?;
    s.lines().next()?.trim().parse::<u64>().ok()
}

/// HF tokenizer_config.json shape — only the chat_template field is read.
/// Some configs ship `chat_template` as a string; others as an array of
/// `{name, template}` dicts. We pick the default (no `name`) entry when
/// it's an array, falling back to the first entry.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ChatTemplateField {
    Single(String),
    Multi(Vec<ChatTemplateEntry>),
}

#[derive(Debug, Deserialize)]
struct ChatTemplateEntry {
    #[serde(default)]
    name: Option<String>,
    template: String,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    chat_template: Option<ChatTemplateField>,
}

impl TokenizerConfig {
    fn extract_template(self) -> Option<String> {
        match self.chat_template? {
            ChatTemplateField::Single(s) => Some(s),
            ChatTemplateField::Multi(entries) => {
                // Prefer the entry named "default" if present, else first.
                entries
                    .iter()
                    .find(|e| e.name.as_deref() == Some("default"))
                    .map(|e| e.template.clone())
                    .or_else(|| entries.into_iter().next().map(|e| e.template))
            }
        }
    }
}

/// Compiled chat template — a minijinja Environment holding the rendered
/// template. Cheap to clone (Arc inside).
struct ChatTemplate {
    env: Environment<'static>,
    template_name: &'static str,
}

impl ChatTemplate {
    fn from_string(template: String) -> Result<Self> {
        let mut env = Environment::new();
        // Add the `tojson` filter HF templates expect for tool definitions.
        // minijinja ships it under a different name; alias for compatibility.
        env.add_template_owned("chat", template)
            .context("compile chat_template")?;
        Ok(Self {
            env,
            template_name: "chat",
        })
    }

    fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        let tmpl = self
            .env
            .get_template(self.template_name)
            .context("template not found")?;
        let rendered = tmpl
            .render(context! {
                messages => messages,
                add_generation_prompt => add_generation_prompt,
            })
            .context("render chat template")?;
        Ok(rendered)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// PEFT adapter_config.json shape — strict subset of what we read.
#[derive(Debug, Deserialize)]
struct PeftAdapterConfig {
    r: usize,
    lora_alpha: f32,
    #[serde(default)]
    lora_dropout: f32,
    target_modules: Vec<String>,
}

#[derive(Parser, Debug)]
#[command(about = "Qwen2 + LoRA inference server / CLI")]
struct Args {
    /// Path to the base model in GGUF format (single-file).
    #[arg(long)]
    gguf: PathBuf,

    /// Path to a HuggingFace tokenizer.json. Family-shared across Qwen2.5 sizes.
    #[arg(long)]
    tokenizer: PathBuf,

    /// Path to an adapter directory (containing `adapter_config.json` +
    /// `adapter_model.safetensors`) OR directly to the safetensors file.
    /// Optional — if omitted, base GGUF runs with no adapter.
    ///
    /// When pointing at a `.safetensors` file directly, target modules and
    /// rank/alpha must be supplied via the override flags below.
    #[arg(long)]
    adapter: Option<PathBuf>,

    /// Override target modules (comma-separated, e.g. `q_proj,k_proj,v_proj,o_proj`).
    /// When omitted and `--adapter` points at a directory, read from
    /// adapter_config.json.
    #[arg(long)]
    target_modules: Option<String>,

    /// Override LoRA rank.
    #[arg(long)]
    rank: Option<usize>,

    /// Override LoRA alpha.
    #[arg(long)]
    alpha: Option<f32>,

    /// Pre-dequantize base GGUF weights to f16 at load time (instead of
    /// dequantizing on the fly during forward). Trades persistent VRAM
    /// for per-token speed. Recommended for >=16GB cards.
    /// 7B Q4_K_M ~4.5GB → ~14GB f16.
    #[arg(long)]
    prequantize_base: bool,

    /// Fold the LoRA adapter into the base weight at load time, then drop
    /// the adapter. After merge, forward does a single matmul per
    /// projection instead of `base + scaling * B(A(x))`. 10-20% faster
    /// decode at the cost of replacing quantized base with f16 (so VRAM
    /// goes up like `--prequantize-base`). Mirrors PEFT's
    /// `model.merge_and_unload()`.
    ///
    /// Implies `--prequantize-base` semantics for the QLoRA path: the
    /// merged weight cannot be represented as Q4_K_M without quality
    /// loss, so we keep it as f16. If memory is tight, leave this OFF
    /// and accept the slight per-token overhead.
    #[arg(long)]
    merge_adapters: bool,

    /// Stack q/k/v base weights into a single matmul per attention layer.
    /// ~5-10% decode speedup. Requires `--prequantize-base`. Compatible
    /// with `--merge-adapters` (which also implies prequantize_base).
    #[arg(long)]
    fuse_qkv: bool,

    /// Enable TF32 GEMMs on Ampere+. ~5-15% inference speedup for f32
    /// paths (LoRA adapters, fp32 lm_head). No effect on pre-Ampere.
    #[arg(long)]
    tf32: bool,

    /// Enable fp16-accumulating f16 GEMMs. Higher quality risk than
    /// TF32; useful for inference where prompt-length reductions are
    /// short. Default OFF.
    #[arg(long)]
    reduced_precision_f16: bool,

    /// Benchmark-decode mode. When set, generate exactly N tokens (no
    /// EOS-early-stop) using `--prompt`, then print a single-line JSON:
    ///   `BENCH_INFER {"decode_tok_per_sec":...,"first_token_ms":...,"total_ms":...,"peak_vram_mb":...}`
    /// and exit 0. Used by bench/run-bench-infer.ps1 for deterministic
    /// inference-side A/B testing of adapter-merge, fuse-qkv, etc.
    #[arg(long)]
    benchmark_decode: Option<usize>,

    /// Bench-only label embedded in the BENCH_INFER line for grep-ability.
    #[arg(long, default_value = "")]
    bench_label: String,

    /// CLI mode: generate one response for this prompt, print, exit.
    #[arg(long)]
    prompt: Option<String>,

    /// Max generation length.
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// Sampling temperature. <=0 disables sampling (argmax).
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Top-p nucleus sampling. None disables.
    #[arg(long)]
    top_p: Option<f64>,

    /// Random seed for sampling.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// HTTP server bind address (server mode). Ignored if --prompt set.
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,

    /// Force CPU. By default uses CUDA device 0.
    #[arg(long)]
    cpu: bool,

    /// Path to the tokenizer_config.json (HuggingFace-format) that contains
    /// the model's `chat_template` Jinja string. When set, the HTTP API
    /// accepts `messages: [{role, content}, ...]` and renders them through
    /// the chat template before generation. Without this flag, only raw
    /// `prompt: "..."` requests are supported.
    ///
    /// Auto-discovered as `<tokenizer_path>/../tokenizer_config.json` when
    /// --tokenizer points at a directory; specify explicitly otherwise.
    #[arg(long)]
    tokenizer_config: Option<PathBuf>,
}

/// Canonicalize and validate an operator-supplied path. The adapter path
/// comes from the CLI (`--adapter`) at process startup, never from network
/// input — the HTTP `/generate` handler only accepts a `prompt` string. We
/// still canonicalize here to fail fast on missing/non-existent paths and
/// to make the intent explicit.
fn canonical_existing(p: &std::path::Path) -> Result<PathBuf> {
    let abs = std::fs::canonicalize(p) // nosemgrep
        .with_context(|| format!("canonicalize {p:?} (does it exist?)"))?;
    Ok(abs)
}

/// Load PEFT adapter weights from a canonicalized safetensors path into the
/// VarMap. The path is the operator's CLI argument, canonicalized + verified
/// to exist before this call.
fn load_adapter_weights(varmap: &mut VarMap, weights_path: &std::path::Path) -> Result<()> {
    eprintln!("loading adapter weights: {}", weights_path.display());
    varmap.load(weights_path) // nosemgrep
        .with_context(|| format!("load {weights_path:?}"))?;
    eprintln!("adapter weights loaded");
    Ok(())
}

/// Resolves the adapter dir/file argument into (config, weights_path).
/// If a directory is given, reads adapter_config.json and uses adapter_model.safetensors.
/// If a file is given, returns just the path; caller must override config via flags.
fn resolve_adapter(
    adapter: &std::path::Path,
    target_modules_override: Option<&str>,
    rank_override: Option<usize>,
    alpha_override: Option<f32>,
) -> Result<(PeftAdapterConfig, PathBuf)> {
    let adapter = canonical_existing(adapter)?;
    if adapter.is_dir() {
        let cfg_path = adapter.join("adapter_config.json");
        let weights_path = adapter.join("adapter_model.safetensors");
        let cfg_path = canonical_existing(&cfg_path)?;
        let weights_path = canonical_existing(&weights_path)?;
        let cfg_str = std::fs::read_to_string(&cfg_path) // nosemgrep
            .with_context(|| format!("read {cfg_path:?}"))?;
        let mut cfg: PeftAdapterConfig =
            serde_json::from_str(&cfg_str).with_context(|| format!("parse {cfg_path:?}"))?;
        if let Some(tm) = target_modules_override {
            cfg.target_modules = tm.split(',').map(|s| s.trim().to_string()).collect();
        }
        if let Some(r) = rank_override {
            cfg.r = r;
        }
        if let Some(a) = alpha_override {
            cfg.lora_alpha = a;
        }
        Ok((cfg, weights_path))
    } else {
        let target_modules = target_modules_override
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "when --adapter points at a file, --target-modules is required"
                )
            })?
            .split(',')
            .map(|s| s.trim().to_string())
            .collect::<Vec<_>>();
        let r = rank_override
            .ok_or_else(|| anyhow::anyhow!("when --adapter points at a file, --rank is required"))?;
        let alpha = alpha_override.ok_or_else(|| {
            anyhow::anyhow!("when --adapter points at a file, --alpha is required")
        })?;
        Ok((
            PeftAdapterConfig {
                r,
                lora_alpha: alpha,
                lora_dropout: 0.0,
                target_modules,
            },
            adapter,
        ))
    }
}

fn parse_targets(names: &[String]) -> Result<HashSet<TargetModule>> {
    let mut out = HashSet::new();
    for n in names {
        let m = TargetModule::from_str(n)
            .ok_or_else(|| anyhow::anyhow!("unknown target module: {n}"))?;
        out.insert(m);
    }
    Ok(out)
}

/// Wraps the loaded model + tokenizer for serving.
struct Engine {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
    chat_template: Option<ChatTemplate>,
}

impl Engine {
    fn load(args: &Args) -> Result<Self> {
        let device = if args.cpu {
            Device::Cpu
        } else {
            Device::new_cuda(0)?
        };
        eprintln!("device: {:?}", device);

        // Tokenizer.
        let tokenizer = Tokenizer::from_file(&args.tokenizer)
            .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
        let eos_token_id = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("</s>"))
            .unwrap_or(151643);

        // Chat template — auto-discover from `<tokenizer_dir>/tokenizer_config.json`
        // if not specified, matching how HF transformers loads it.
        let tc_path: Option<PathBuf> = args.tokenizer_config.clone().or_else(|| {
            args.tokenizer
                .parent()
                .map(|p| p.join("tokenizer_config.json"))
                .filter(|p| p.exists())
        });
        let chat_template = if let Some(p) = tc_path {
            let p = canonical_existing(&p)?;
            eprintln!("loading tokenizer_config: {}", p.display());
            let cfg_str = std::fs::read_to_string(&p) // nosemgrep
                .with_context(|| format!("read {p:?}"))?;
            let cfg: TokenizerConfig =
                serde_json::from_str(&cfg_str).with_context(|| format!("parse {p:?}"))?;
            match cfg.extract_template() {
                Some(t) => {
                    eprintln!("chat_template: loaded ({} chars)", t.len());
                    Some(ChatTemplate::from_string(t)?)
                }
                None => {
                    eprintln!("chat_template: NONE (config exists but no template field)");
                    None
                }
            }
        } else {
            eprintln!("chat_template: NONE (no tokenizer_config.json found)");
            None
        };

        // Resolve the adapter ONCE upfront so we have one canonicalized
        // weights_path to thread through. Both the LoRA shell construction
        // and the eventual VarMap::load operate on locally-bound paths.
        let resolved = match args.adapter.as_ref() {
            Some(adapter) => Some(resolve_adapter(
                adapter,
                args.target_modules.as_deref(),
                args.rank,
                args.alpha,
            )?),
            None => None,
        };

        let (targets, lora_cfg) = match resolved.as_ref() {
            Some((cfg, _)) => {
                let targets = parse_targets(&cfg.target_modules)?;
                let lora_cfg = LoRAConfig {
                    rank: cfg.r,
                    alpha: cfg.lora_alpha,
                    dropout: 0.0, // no dropout at inference
                };
                eprintln!(
                    "adapter: r={}, alpha={}, targets={:?}",
                    lora_cfg.rank, lora_cfg.alpha, cfg.target_modules
                );
                (targets, lora_cfg)
            }
            None => {
                eprintln!("adapter: NONE (base GGUF only)");
                (HashSet::new(), LoRAConfig::default())
            }
        };

        // Build VarMap (will hold adapter weights once loaded).
        let mut varmap = VarMap::new();
        let vb_lora = VarBuilder::from_varmap(&varmap, candle::DType::F32, &device);

        // Load the base GGUF + attach LoRA shells. The GGUF path is an
        // operator CLI argument — process startup, not network input.
        // The HTTP `/generate` handler accepts only a `prompt` string.
        let gguf_path = canonical_existing(&args.gguf)?;
        eprintln!("loading GGUF: {}", gguf_path.display());
        let mut f = std::fs::File::open(&gguf_path) // nosemgrep
            .with_context(|| format!("open {gguf_path:?}"))?;
        let ct = gguf_file::Content::read(&mut f)
            .with_context(|| format!("parse {gguf_path:?}"))?;
        // Merging implies pre-dequant: the merged weight is f16, can't go
        // back to Q4_K_M cleanly, so we might as well start with f16.
        // Fused QKV requires prequantize_base too (stacking quantized
        // weights doesn't make sense).
        let prequantize_base = args.prequantize_base || args.merge_adapters || args.fuse_qkv;
        let mut model = Model::from_gguf(
            ct, &mut f, &targets, &lora_cfg, vb_lora, &device,
            prequantize_base, args.fuse_qkv,
        )?;
        eprintln!(
            "model loaded (prequantize_base={prequantize_base}, fuse_qkv={})",
            args.fuse_qkv
        );

        // Pull trained adapter weights into the VarMap. The path was
        // canonicalized + existence-checked in resolve_adapter above.
        if let Some((_, weights_path)) = resolved {
            load_adapter_weights(&mut varmap, &weights_path)?;
        }

        // After the adapter weights are in the VarMap, optionally fold
        // them into the base. Order matters: weights must be loaded
        // BEFORE merge so the merged weight reflects the trained values.
        if args.merge_adapters {
            if args.adapter.is_none() {
                eprintln!(
                    "warning: --merge-adapters requested but no --adapter \
                     supplied; nothing to merge, base will be used as-is"
                );
            } else {
                eprintln!("merging adapters into base weights...");
                model.merge_adapters_into_base()?;
                eprintln!("merge complete; per-token adapter compute eliminated");
            }
        }

        Ok(Self {
            model,
            tokenizer,
            device,
            eos_token_id,
            chat_template,
        })
    }

    /// Generate up to `max_tokens` tokens after the prompt. Stops on EOS.
    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        seed: u64,
    ) -> Result<String> {
        // Reset kv cache between calls (the kv cache is per-Engine; without
        // reset, a 2nd call would prepend the previous conversation).
        self.model.clear_kv_cache();

        // Tokenize.
        let enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let prompt_ids: Vec<u32> = enc.get_ids().to_vec();

        // Sampling strategy.
        let sampling = if temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            match top_p {
                Some(p) => Sampling::TopP { p, temperature },
                None => Sampling::All { temperature },
            }
        };
        let mut logits_proc = LogitsProcessor::from_sampling(seed, sampling);

        // Prefill (full prompt at index_pos=0, batch=1).
        let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &self.device)?
            .unsqueeze(0)?;
        let mut logits = self.model.forward(&prompt_tensor, 0)?;
        let mut next_token = logits_proc.sample(&logits.squeeze(0)?)?;
        let mut produced: Vec<u32> = vec![next_token];

        // Decode loop.
        let mut index_pos = prompt_ids.len();
        for _ in 1..max_tokens {
            if next_token == self.eos_token_id {
                break;
            }
            let inp = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&inp, index_pos)?;
            next_token = logits_proc.sample(&logits.squeeze(0)?)?;
            produced.push(next_token);
            index_pos += 1;
        }

        // Strip trailing EOS for cleaner output.
        if produced.last() == Some(&self.eos_token_id) {
            produced.pop();
        }

        let text = self
            .tokenizer
            .decode(&produced, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        Ok(text)
    }

    /// Streaming variant of `generate`: pushes each newly-decoded token's
    /// text fragment into `tx` as it's produced. Drops out cleanly if the
    /// receiver is dropped (e.g. client disconnects). Returns the full
    /// concatenated output string for logging.
    ///
    /// Per-token decode: we maintain a running `decoded_so_far` string and
    /// re-decode the full token list each iteration; the delta is the new
    /// text. This is the standard pattern — naively decoding only the
    /// last token can produce broken output for multi-byte tokens or
    /// merged BPE sequences.
    fn generate_streaming(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        seed: u64,
        tx: &std::sync::mpsc::Sender<String>,
    ) -> Result<String> {
        self.model.clear_kv_cache();

        let enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let prompt_ids: Vec<u32> = enc.get_ids().to_vec();

        let sampling = if temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            match top_p {
                Some(p) => Sampling::TopP { p, temperature },
                None => Sampling::All { temperature },
            }
        };
        let mut logits_proc = LogitsProcessor::from_sampling(seed, sampling);

        let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &self.device)?
            .unsqueeze(0)?;
        let mut logits = self.model.forward(&prompt_tensor, 0)?;
        let mut next_token = logits_proc.sample(&logits.squeeze(0)?)?;
        let mut produced: Vec<u32> = vec![next_token];

        // Decode the first token and emit. After this, we incrementally
        // decode and emit the delta.
        let mut decoded_so_far = self
            .tokenizer
            .decode(&produced, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        if !decoded_so_far.is_empty() {
            // Best-effort send; ignore errors (client may have disconnected).
            let _ = tx.send(decoded_so_far.clone());
        }

        let mut index_pos = prompt_ids.len();
        for _ in 1..max_tokens {
            if next_token == self.eos_token_id {
                break;
            }
            let inp = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&inp, index_pos)?;
            next_token = logits_proc.sample(&logits.squeeze(0)?)?;
            produced.push(next_token);
            index_pos += 1;

            let new_full = self
                .tokenizer
                .decode(&produced, true)
                .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
            // Emit the delta. The tokenizer occasionally rewrites earlier
            // text (e.g. merging BPE pieces) so we can't strictly assume
            // monotonic prefix-extension; treat the delta as everything
            // beyond the currently-emitted prefix length.
            if new_full.len() > decoded_so_far.len() {
                let delta = new_full[decoded_so_far.len()..].to_string();
                if tx.send(delta).is_err() {
                    // Receiver dropped (client disconnected). Stop early.
                    decoded_so_far = new_full;
                    break;
                }
            }
            decoded_so_far = new_full;
        }

        // Strip trailing EOS for the returned full string (clients that
        // care about the EOS token can detect the [DONE] sentinel).
        if produced.last() == Some(&self.eos_token_id) {
            produced.pop();
            // Re-decode the final clean string for the return value.
            decoded_so_far = self
                .tokenizer
                .decode(&produced, true)
                .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        }
        Ok(decoded_so_far)
    }

    /// Render a chat-message array through the loaded chat_template, then
    /// hand off to `generate()`. Errors clearly if no chat_template was
    /// loaded at startup.
    fn chat(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        seed: u64,
    ) -> Result<String> {
        let tmpl = self
            .chat_template
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!(
                "chat endpoint requires --tokenizer-config (or auto-discovery via \
                 a tokenizer_config.json next to --tokenizer); start the server \
                 with that flag, or use the /generate endpoint with a raw prompt"
            ))?;
        let prompt = tmpl.render(messages, true)?;
        self.generate(&prompt, max_tokens, temperature, top_p, seed)
    }

    /// Deterministic decode benchmark. Generates exactly `n_tokens`
    /// without EOS-early-stop, sampling argmax for reproducibility.
    /// Returns timing breakdown for the BENCH_INFER JSON line.
    ///
    /// Phases:
    ///   prefill_ms     — full-prompt forward (most of which overlaps
    ///                    with first-token decode in real serving).
    ///   first_token_ms — wall clock from start to first sampled token,
    ///                    INCLUDING prefill.
    ///   decode_ms      — wall clock from first token to last token.
    ///   decode_tok_per_sec — (n_tokens - 1) / decode_ms.
    ///   total_ms       — full wall clock.
    fn bench_decode(
        &mut self,
        prompt: &str,
        n_tokens: usize,
    ) -> Result<BenchInferResult> {
        use std::time::Instant;
        if n_tokens < 2 {
            anyhow::bail!("benchmark_decode requires n_tokens >= 2");
        }
        self.model.clear_kv_cache();

        let total_start = Instant::now();

        let enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
        let prompt_len = prompt_ids.len();

        // ArgMax sampling for determinism — bench should be repeatable.
        let mut logits_proc =
            LogitsProcessor::from_sampling(0, Sampling::ArgMax);

        // Prefill.
        let prefill_start = Instant::now();
        let prompt_tensor =
            Tensor::new(prompt_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(&prompt_tensor, 0)?;
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        let mut next_token = logits_proc.sample(&logits.squeeze(0)?)?;
        let first_token_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        // Decode loop — exactly n_tokens-1 more tokens (we already have 1).
        let decode_start = Instant::now();
        let mut index_pos = prompt_len;
        for _ in 1..n_tokens {
            let inp = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&inp, index_pos)?;
            next_token = logits_proc.sample(&logits.squeeze(0)?)?;
            index_pos += 1;
        }
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        let decoded_tokens = (n_tokens - 1) as f64;
        let decode_tok_per_sec = if decode_ms > 0.0 {
            decoded_tokens / (decode_ms / 1000.0)
        } else {
            0.0
        };

        // Best-effort peak VRAM from nvidia-smi.
        let peak_vram_mb = peak_vram_mb_via_nvidia_smi();

        Ok(BenchInferResult {
            decode_tok_per_sec,
            first_token_ms,
            prefill_ms,
            decode_ms,
            total_ms,
            n_tokens,
            prompt_len,
            peak_vram_mb,
        })
    }

    /// Streaming variant of `chat`. Renders the chat template then calls
    /// `generate_streaming`.
    fn chat_streaming(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        seed: u64,
        tx: &std::sync::mpsc::Sender<String>,
    ) -> Result<String> {
        let tmpl = self
            .chat_template
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!(
                "chat/stream endpoint requires --tokenizer-config"
            ))?;
        let prompt = tmpl.render(messages, true)?;
        self.generate_streaming(&prompt, max_tokens, temperature, top_p, seed, tx)
    }
}

#[derive(Debug, Deserialize)]
struct GenerateRequest {
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Debug, Serialize)]
struct GenerateResponse {
    text: String,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<Engine>>,
    defaults: GenerationDefaults,
}

#[derive(Clone)]
struct GenerationDefaults {
    max_tokens: usize,
    temperature: f64,
    top_p: Option<f64>,
    seed: u64,
}

async fn generate_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let max_tokens = req.max_tokens.unwrap_or(state.defaults.max_tokens);
    let temperature = req.temperature.unwrap_or(state.defaults.temperature);
    let top_p = req.top_p.or(state.defaults.top_p);
    let seed = req.seed.unwrap_or(state.defaults.seed);

    // Inference is single-threaded against the model — serialize with a Mutex.
    let mut engine = state.engine.lock().await;
    match engine.generate(&req.prompt, max_tokens, temperature, top_p, seed) {
        Ok(text) => Ok(Json(GenerateResponse { text })),
        Err(e) => Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("{e:#}"),
            }),
        )),
    }
}

/// `/chat` request shape: messages array (role + content), plus the same
/// sampling overrides as `/generate`. The chat_template is applied by
/// the server before generation.
#[derive(Debug, Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    seed: Option<u64>,
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<GenerateResponse>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let max_tokens = req.max_tokens.unwrap_or(state.defaults.max_tokens);
    let temperature = req.temperature.unwrap_or(state.defaults.temperature);
    let top_p = req.top_p.or(state.defaults.top_p);
    let seed = req.seed.unwrap_or(state.defaults.seed);
    let mut engine = state.engine.lock().await;
    match engine.chat(&req.messages, max_tokens, temperature, top_p, seed) {
        Ok(text) => Ok(Json(GenerateResponse { text })),
        Err(e) => Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("{e:#}"),
            }),
        )),
    }
}

/// Build an SSE stream that runs the supplied generation closure on a
/// blocking thread and forwards each token-text-delta as a `data: ...`
/// event to the client. Sends a final `data: [DONE]` sentinel on success
/// (matching the OpenAI streaming convention) or `data: [ERROR] {msg}`
/// on failure. Drops cleanly if the client disconnects (the closure's
/// `tx.send` will start failing and it'll exit early).
fn sse_from_blocking_gen<F>(
    state: AppState,
    work: F,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
where
    F: FnOnce(&mut Engine, &std::sync::mpsc::Sender<String>) -> Result<String> + Send + 'static,
{
    // Two channels: the std::sync one the synchronous Engine generation
    // sends through, and the tokio one we expose to the SSE stream.
    let (sync_tx, sync_rx) = std::sync::mpsc::channel::<String>();
    let (async_tx, async_rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    // Bridge thread: pulls from sync_rx, pushes to async_tx. Cheap; one
    // tokio task per request and one bridge thread per request.
    let bridge_async_tx = async_tx.clone();
    std::thread::spawn(move || {
        for chunk in sync_rx.iter() {
            let evt = Event::default().data(chunk);
            if bridge_async_tx.blocking_send(Ok(evt)).is_err() {
                // SSE stream dropped — stop bridging.
                break;
            }
        }
    });

    // Worker thread: locks the engine and runs the generation, pushing
    // through sync_tx. When done, sends [DONE] (or [ERROR]) via async_tx.
    let async_tx_done = async_tx.clone();
    tokio::task::spawn_blocking(move || {
        // tokio::sync::Mutex requires a runtime; we use blocking_lock
        // since we're on a blocking thread.
        let mut engine = state.engine.blocking_lock();
        let result = work(&mut engine, &sync_tx);
        drop(sync_tx); // close sync side so bridge thread exits

        let final_event = match result {
            Ok(_) => Event::default().data("[DONE]"),
            Err(e) => Event::default().data(format!("[ERROR] {e:#}")),
        };
        // Best-effort; ignore if client is gone.
        let _ = async_tx_done.blocking_send(Ok(final_event));
    });

    Sse::new(ReceiverStream::new(async_rx)).keep_alive(KeepAlive::default())
}

async fn generate_stream_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let max_tokens = req.max_tokens.unwrap_or(state.defaults.max_tokens);
    let temperature = req.temperature.unwrap_or(state.defaults.temperature);
    let top_p = req.top_p.or(state.defaults.top_p);
    let seed = req.seed.unwrap_or(state.defaults.seed);
    let prompt = req.prompt.clone();
    sse_from_blocking_gen(state, move |engine, tx| {
        engine.generate_streaming(&prompt, max_tokens, temperature, top_p, seed, tx)
    })
}

async fn chat_stream_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let max_tokens = req.max_tokens.unwrap_or(state.defaults.max_tokens);
    let temperature = req.temperature.unwrap_or(state.defaults.temperature);
    let top_p = req.top_p.or(state.defaults.top_p);
    let seed = req.seed.unwrap_or(state.defaults.seed);
    let messages = req.messages.clone();
    sse_from_blocking_gen(state, move |engine, tx| {
        engine.chat_streaming(&messages, max_tokens, temperature, top_p, seed, tx)
    })
}

async fn health() -> &'static str {
    "ok"
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Apply CUDA precision-mode flags BEFORE loading the model so the
    // first cuBLAS dispatch picks them up. No-op on Pascal/CPU.
    #[cfg(feature = "cuda")]
    {
        if args.tf32 {
            candle::cuda_backend::set_gemm_reduced_precision_f32(true);
            eprintln!("tf32: ENABLED");
        }
        if args.reduced_precision_f16 {
            candle::cuda_backend::set_gemm_reduced_precision_f16(true);
            eprintln!("f16-reduce: ENABLED");
        }
    }

    let mut engine = Engine::load(&args)?;

    // Benchmark-decode mode: deterministic n-token decode, emit
    // single-line `BENCH_INFER {...}` JSON, exit 0.
    if let Some(n_tokens) = args.benchmark_decode {
        let prompt = args.prompt.as_deref().unwrap_or(
            "The quick brown fox jumps over the lazy dog. \
             Write a short story about that fox's next adventure.",
        );
        let result = engine.bench_decode(prompt, n_tokens)?;
        // Emit a single-line JSON keyed BENCH_INFER for the runner to grep.
        let mut json = serde_json::to_value(&result)?;
        if let Some(obj) = json.as_object_mut() {
            obj.insert(
                "label".into(),
                serde_json::Value::String(args.bench_label.clone()),
            );
        }
        println!("BENCH_INFER {}", serde_json::to_string(&json)?);
        return Ok(());
    }

    // CLI mode — generate once and exit.
    if let Some(prompt) = args.prompt.as_ref() {
        let text = engine.generate(
            prompt,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.seed,
        )?;
        println!("{}", text);
        return Ok(());
    }

    // Server mode.
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        defaults: GenerationDefaults {
            max_tokens: args.max_tokens,
            temperature: args.temperature,
            top_p: args.top_p,
            seed: args.seed,
        },
    };

    let app = Router::new()
        .route("/generate", post(generate_handler))
        .route("/generate/stream", post(generate_stream_handler))
        .route("/chat", post(chat_handler))
        .route("/chat/stream", post(chat_stream_handler))
        .route("/health", axum::routing::get(health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&args.bind)
        .await
        .with_context(|| format!("bind {}", args.bind))?;
    eprintln!("listening on http://{}", args.bind);
    axum::serve(listener, app).await?;
    Ok(())
}
