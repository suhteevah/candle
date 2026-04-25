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
use axum::{extract::State, routing::post, Json, Router};
use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use lora::LoRAConfig;
use qwen2_lora::TargetModule;
use qwen2_lora_quantized::Model;

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
        let model = Model::from_gguf(ct, &mut f, &targets, &lora_cfg, vb_lora, &device)?;
        eprintln!("model loaded");

        // Pull trained adapter weights into the VarMap. The path was
        // canonicalized + existence-checked in resolve_adapter above.
        if let Some((_, weights_path)) = resolved {
            load_adapter_weights(&mut varmap, &weights_path)?;
        }

        Ok(Self {
            model,
            tokenizer,
            device,
            eos_token_id,
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

async fn health() -> &'static str {
    "ok"
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    let args = Args::parse();

    let mut engine = Engine::load(&args)?;

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
        .route("/health", axum::routing::get(health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&args.bind)
        .await
        .with_context(|| format!("bind {}", args.bind))?;
    eprintln!("listening on http://{}", args.bind);
    axum::serve(listener, app).await?;
    Ok(())
}
