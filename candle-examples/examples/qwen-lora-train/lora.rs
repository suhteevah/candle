//! Reusable LoRA adapter — `y = Wx + (alpha/r) * B(A(x))`.
//!
//! - `W` is the frozen base-model projection (not stored here; caller holds it).
//! - `A` is rank-r down-projection, xavier-init.
//! - `B` is rank-r up-projection, zero-init (so the adapter is a no-op at step 0).
//!
//! At save time, we serialize only A + B; the base stays GGUF-quantized untouched.

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};

/// Hyper-parameters for a LoRA adapter.
#[derive(Clone, Debug)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
        }
    }
}

/// Rank-r low-rank adapter `y = (alpha/r) * B(A(x))`.
///
/// Use alongside a frozen base linear; `forward_delta` returns the additive
/// correction. Caller adds it to the base projection output.
#[derive(Debug)]
pub struct LoRALinear {
    a: Linear,
    b: Linear,
    scaling: f64,
    dropout: Option<candle_nn::Dropout>,
}

impl LoRALinear {
    /// Build a new LoRA adapter registered in the provided VarBuilder.
    ///
    /// `vs` should be scoped to the full parameter path, e.g.
    /// `vb.pp("model.layers.0.self_attn.q_proj.lora")` — the A and B weights
    /// will register as `{prefix}.lora_A.weight` and `{prefix}.lora_B.weight`
    /// to match the HuggingFace PEFT naming convention on export.
    pub fn new(
        in_features: usize,
        out_features: usize,
        cfg: &LoRAConfig,
        vs: VarBuilder,
    ) -> Result<Self> {
        // Kaiming / Xavier uniform for A matches HF PEFT's default init.
        let bound = (1.0f64 / in_features as f64).sqrt();
        let a_init = candle_nn::init::Init::Uniform {
            lo: -bound,
            up: bound,
        };
        let a_w = vs.get_with_hints((cfg.rank, in_features), "lora_A.weight", a_init)?;
        let a = Linear::new(a_w, None);

        // B is zero-initialized so the whole adapter is a no-op at step 0 —
        // base-model behavior preserved until training starts.
        let b_w = vs.get_with_hints(
            (out_features, cfg.rank),
            "lora_B.weight",
            candle_nn::init::ZERO,
        )?;
        let b = Linear::new(b_w, None);

        let scaling = (cfg.alpha as f64) / (cfg.rank as f64);
        let dropout = if cfg.dropout > 0.0 {
            Some(candle_nn::Dropout::new(cfg.dropout))
        } else {
            None
        };

        Ok(Self {
            a,
            b,
            scaling,
            dropout,
        })
    }

    /// Returns the merged-into-base weight delta: `scaling * B @ A`. Adding
    /// this to the base projection's `(out, in)` weight tensor produces a
    /// merged base that, when used by itself, gives the same output as
    /// `base @ x + scaling * B(A(x))`. Used at inference deployment time
    /// to flatten LoRA adapters into the base — eliminates per-token
    /// adapter compute and frees adapter VRAM.
    ///
    /// The returned tensor is in the dtype of `B` (typically fp32 during
    /// training; callers should `to_dtype` to match the base they're
    /// merging into, usually f16).
    pub fn merged_weight_delta(&self) -> Result<Tensor> {
        let a_w = self.a.weight();          // (rank, in)
        let b_w = self.b.weight();          // (out, rank)
        let ba = b_w.matmul(a_w)?;          // (out, in)
        ba.affine(self.scaling, 0.0)
    }

    /// Returns the additive LoRA delta to add to the base projection output.
    /// `training=true` applies dropout; `false` skips it (for eval).
    pub fn forward_delta(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let x = if let (Some(d), true) = (&self.dropout, training) {
            d.forward(x, true)?
        } else {
            x.clone()
        };
        let ax = self.a.forward(&x)?;
        let bax = self.b.forward(&ax)?;
        bax.affine(self.scaling, 0.0)
    }
}

/// Convenience wrapper: a LoRA-decorated linear that internally holds a
/// reference to the frozen base Linear-like callable. Caller supplies a
/// closure that invokes the base (e.g. a dequant-on-read path into the GGUF
/// model), and we add the LoRA delta.
///
/// This lets us keep `qwen2_lora.rs` thin — the quantized base stays in the
/// ModelWeights struct, and we just drop a `LoRALinear` next to each targeted
/// projection and compose at forward time.
pub fn add_lora_delta(base_out: &Tensor, adapter: &LoRALinear, adapter_input: &Tensor, training: bool) -> Result<Tensor> {
    let delta = adapter.forward_delta(adapter_input, training)?;
    // cast delta to match base dtype (base may be fp16, delta fp32 for stability)
    let delta = if delta.dtype() != base_out.dtype() {
        delta.to_dtype(base_out.dtype())?
    } else {
        delta
    };
    base_out.broadcast_add(&delta)
}

/// Tiny helper for debugging: print the L2 norm of both adapter matrices.
/// Useful during training to verify both are actually updating.
pub fn param_norms(adapter: &LoRALinear) -> Result<(f64, f64)> {
    let a_sq = adapter.a.weight().sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    let b_sq = adapter.b.weight().sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    Ok((a_sq.sqrt(), b_sq.sqrt()))
}

// Silences unused-import warnings for Device/DType/D/Module during scaffold
// phase; they'll be used as we fill in more methods.
#[allow(dead_code)]
fn _type_fence(_d: &Device, _t: &DType, _m: &dyn Module, _dim: D) {}
