//! PEFT-compatible adapter export.
//!
//! We write two files that together load cleanly in any HuggingFace PEFT-
//! aware stack (transformers + peft, axolotl, unsloth inference, vLLM PEFT
//! support, llama.cpp LoRA merge, etc.):
//!
//!   adapter_config.json
//!   adapter_model.safetensors
//!
//! The config is a strict subset of what PEFT writes. The safetensors file
//! uses HF naming: `base_model.model.<layer-path>.lora_A.weight` etc. Our
//! VarBuilder prefixes during training must match this convention so we can
//! snapshot the VarMap directly without a post-hoc rename pass.

use anyhow::{Context, Result};
use candle_nn::VarMap;
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone, Serialize)]
pub struct PeftAdapterConfig {
    pub peft_type: String,
    pub task_type: String,
    pub base_model_name_or_path: String,
    pub r: usize,
    pub lora_alpha: f32,
    pub lora_dropout: f32,
    pub bias: String,
    pub target_modules: Vec<String>,
    pub fan_in_fan_out: bool,
    pub inference_mode: bool,
}

impl PeftAdapterConfig {
    /// Build a minimal but PEFT-compatible config.
    pub fn new(
        base_model: impl Into<String>,
        r: usize,
        lora_alpha: f32,
        lora_dropout: f32,
        target_modules: Vec<String>,
    ) -> Self {
        Self {
            peft_type: "LORA".to_string(),
            task_type: "CAUSAL_LM".to_string(),
            base_model_name_or_path: base_model.into(),
            r,
            lora_alpha,
            lora_dropout,
            bias: "none".to_string(),
            target_modules,
            fan_in_fan_out: false,
            inference_mode: false,
        }
    }
}

/// Write `adapter_config.json` + `adapter_model.safetensors` into `out_dir`.
pub fn save_adapter(
    out_dir: &Path,
    config: &PeftAdapterConfig,
    varmap: &VarMap,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("create {out_dir:?}"))?;

    // 1. config
    let cfg_path = out_dir.join("adapter_config.json");
    let cfg_json = serde_json::to_string_pretty(config)?;
    std::fs::write(&cfg_path, cfg_json)
        .with_context(|| format!("write {cfg_path:?}"))?;

    // 2. weights (VarMap::save writes safetensors; naming comes from how we
    //    constructed the VarBuilder during training, so upstream we must use
    //    the PEFT-convention path prefixes).
    let weights_path = out_dir.join("adapter_model.safetensors");
    varmap
        .save(&weights_path)
        .with_context(|| format!("save {weights_path:?}"))?;

    Ok(())
}
