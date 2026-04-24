//! Qwen2 model with LoRA adapters on attention + MLP projections.
//!
//! This is a minimum-diff fork of `candle-transformers::models::qwen2` that:
//! 1. Accepts a second `VarBuilder` scoped to a trainable `VarMap` for adapters.
//! 2. Adds an optional `LoRALinear` next to each of q_proj, k_proj, v_proj,
//!    o_proj, gate_proj, up_proj, down_proj — selected by a `TargetModules`
//!    set at build time.
//! 3. Composes `y = base(x) + lora_delta(x)` at forward time (only when the
//!    adapter is present).
//!
//! Base weights are loaded from a frozen `VarBuilder` (e.g. safetensors);
//! LoRA weights come from a trainable `VarBuilder` backed by a VarMap.
//! Gradients flow only through the LoRA branches.
//!
//! Why we forked qwen2.rs instead of quantized_qwen2.rs: candle's QMatMul has
//! no backward pass, so gradient can't propagate through a quantized base.
//! See memory/reference_candle_qmatmul_no_backward.md.

use crate::lora::{LoRAConfig, LoRALinear};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use std::collections::HashSet;
use std::sync::Arc;

/// Differentiable RmsNorm. Candle's `candle_nn::RmsNorm` calls
/// `ops::rms_norm` (no_bwd) — fine for inference, fatal for training. We
/// hold only the alpha weight and call `ops::rms_norm_slow` at forward time.
#[derive(Debug, Clone)]
struct DiffRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl DiffRmsNorm {
    fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(hidden_size, "weight", candle_nn::init::ONE)?;
        Ok(Self { weight, eps })
    }
}

impl Module for DiffRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm_slow(xs, &self.weight, self.eps as f32)
    }
}

/// Projections that can be LoRA-adapted. Mirrors HuggingFace PEFT naming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetModule {
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
}

impl TargetModule {
    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s.trim() {
            "q_proj" => Self::QProj,
            "k_proj" => Self::KProj,
            "v_proj" => Self::VProj,
            "o_proj" => Self::OProj,
            "gate_proj" => Self::GateProj,
            "up_proj" => Self::UpProj,
            "down_proj" => Self::DownProj,
            _ => return None,
        })
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::QProj => "q_proj",
            Self::KProj => "k_proj",
            Self::VProj => "v_proj",
            Self::OProj => "o_proj",
            Self::GateProj => "gate_proj",
            Self::UpProj => "up_proj",
            Self::DownProj => "down_proj",
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        // rope_slow uses standard differentiable ops (narrow + cat + mul),
        // preserving the autograd graph through q_proj/k_proj LoRA adapters.
        // The fast `rope` function calls apply_op3_no_bwd which silently cuts
        // the gradient graph — catastrophic for LoRA training. The speed
        // difference vs rope_slow is negligible compared to the cost of a
        // full forward+backward pass through 28 transformer layers.
        let q_embed = candle_nn::rotary_emb::rope_slow(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_slow(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

/// A linear projection optionally augmented by a LoRA adapter. The base
/// Linear comes from the frozen `VarBuilder`; the adapter (if any) comes from
/// the trainable `VarBuilder` at the same logical path (+ a `lora_A`/`lora_B`
/// leaf). Composition: `y = base(x) + delta(x)`.
#[derive(Debug)]
struct LoRAProjection {
    base: candle_nn::Linear,
    adapter: Option<LoRALinear>,
}

impl LoRAProjection {
    fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        target: TargetModule,
        targets: &HashSet<TargetModule>,
        lora_cfg: &LoRAConfig,
        vb_base: VarBuilder,
        vb_lora: VarBuilder,
    ) -> Result<Self> {
        let base = if bias {
            candle_nn::linear(in_features, out_features, vb_base)?
        } else {
            candle_nn::linear_no_bias(in_features, out_features, vb_base)?
        };
        let adapter = if targets.contains(&target) {
            Some(LoRALinear::new(in_features, out_features, lora_cfg, vb_lora)?)
        } else {
            None
        };
        Ok(Self { base, adapter })
    }

    fn forward(&self, xs: &Tensor, training: bool) -> Result<Tensor> {
        let base_out = self.base.forward(xs)?;
        match &self.adapter {
            None => Ok(base_out),
            Some(a) => {
                // LoRA adapters now match base dtype — no cast, no autograd
                // graph breakage. See main.rs for the rationale.
                let delta = a.forward_delta(xs, training)?;
                base_out.broadcast_add(&delta)
            }
        }
    }
}

#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: LoRAProjection,
    up_proj: LoRAProjection,
    down_proj: LoRAProjection,
    act_fn: Activation,
}

impl MLP {
    fn new(
        cfg: &Config,
        targets: &HashSet<TargetModule>,
        lora_cfg: &LoRAConfig,
        vb_base: VarBuilder,
        vb_lora: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = LoRAProjection::new(
            hidden_sz,
            intermediate_sz,
            false,
            TargetModule::GateProj,
            targets,
            lora_cfg,
            vb_base.pp("gate_proj"),
            vb_lora.pp("gate_proj"),
        )?;
        let up_proj = LoRAProjection::new(
            hidden_sz,
            intermediate_sz,
            false,
            TargetModule::UpProj,
            targets,
            lora_cfg,
            vb_base.pp("up_proj"),
            vb_lora.pp("up_proj"),
        )?;
        let down_proj = LoRAProjection::new(
            intermediate_sz,
            hidden_sz,
            false,
            TargetModule::DownProj,
            targets,
            lora_cfg,
            vb_base.pp("down_proj"),
            vb_lora.pp("down_proj"),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor, training: bool) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(xs, training)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(xs, training)?;
        self.down_proj.forward(&(lhs * rhs)?, training)
    }
}

#[derive(Debug)]
struct Attention {
    q_proj: LoRAProjection,
    k_proj: LoRAProjection,
    v_proj: LoRAProjection,
    o_proj: LoRAProjection,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        targets: &HashSet<TargetModule>,
        lora_cfg: &LoRAConfig,
        vb_base: VarBuilder,
        vb_lora: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = LoRAProjection::new(
            hidden_sz,
            num_heads * head_dim,
            true,
            TargetModule::QProj,
            targets,
            lora_cfg,
            vb_base.pp("q_proj"),
            vb_lora.pp("q_proj"),
        )?;
        let k_proj = LoRAProjection::new(
            hidden_sz,
            num_kv_heads * head_dim,
            true,
            TargetModule::KProj,
            targets,
            lora_cfg,
            vb_base.pp("k_proj"),
            vb_lora.pp("k_proj"),
        )?;
        let v_proj = LoRAProjection::new(
            hidden_sz,
            num_kv_heads * head_dim,
            true,
            TargetModule::VProj,
            targets,
            lora_cfg,
            vb_base.pp("v_proj"),
            vb_lora.pp("v_proj"),
        )?;
        let o_proj = LoRAProjection::new(
            num_heads * head_dim,
            hidden_sz,
            false,
            TargetModule::OProj,
            targets,
            lora_cfg,
            vb_base.pp("o_proj"),
            vb_lora.pp("o_proj"),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        training: bool,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs, training)?;
        let key_states = self.k_proj.forward(xs, training)?;
        let value_states = self.v_proj.forward(xs, training)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        // NOTE: we deliberately skip KV cache during training — teacher-forced
        // cross-entropy training processes the full sequence in one shot, so
        // a cache just wastes memory and complicates grad flow.
        let (key_states, value_states) = if training {
            (key_states, value_states)
        } else {
            match &self.kv_cache {
                None => (key_states, value_states),
                Some((prev_k, prev_v)) => {
                    let k = Tensor::cat(&[prev_k, &key_states], 2)?;
                    let v = Tensor::cat(&[prev_v, &value_states], 2)?;
                    (k, v)
                }
            }
        };
        if !training {
            self.kv_cache = Some((key_states.clone(), value_states.clone()));
        }

        let key_states = candle_transformers::utils::repeat_kv(key_states, self.num_kv_groups)?
            .contiguous()?;
        let value_states = candle_transformers::utils::repeat_kv(value_states, self.num_kv_groups)?
            .contiguous()?;

        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;
            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            // fp16 softmax is a known overflow/NaN hazard — pre-softmax
            // logits commonly exceed fp16's 65504 range after scaling by
            // sqrt(head_dim). Cast to fp32 for softmax, cast back for the
            // subsequent matmul against values.
            let attn_dtype_orig = attn_weights.dtype();
            let attn_weights = if attn_dtype_orig == DType::F32 {
                attn_weights
            } else {
                attn_weights.to_dtype(DType::F32)?
            };
            // Use the differentiable softmax (built from max+sub+exp+sum+div
            // with full autograd support) instead of softmax_last_dim (no_bwd).
            let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
            let attn_weights = if attn_weights.dtype() == value_states.dtype() {
                attn_weights
            } else {
                attn_weights.to_dtype(value_states.dtype())?
            };
            attn_weights.matmul(&value_states)?
        };
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?;
        self.o_proj.forward(&attn_output, training)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: DiffRmsNorm,
    post_attention_layernorm: DiffRmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        targets: &HashSet<TargetModule>,
        lora_cfg: &LoRAConfig,
        vb_base: VarBuilder,
        vb_lora: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            targets,
            lora_cfg,
            vb_base.pp("self_attn"),
            vb_lora.pp("self_attn"),
        )?;
        let mlp = MLP::new(
            cfg,
            targets,
            lora_cfg,
            vb_base.pp("mlp"),
            vb_lora.pp("mlp"),
        )?;
        let input_layernorm = DiffRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_base.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = DiffRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_base.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        training: bool,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset, training)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs_mlp_in = self.post_attention_layernorm.forward(&xs)?;
        let xs_mlp_out = self.mlp.forward(&xs_mlp_in, training)?;
        residual + xs_mlp_out
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

#[derive(Debug)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: DiffRmsNorm,
    lm_head: candle_nn::Linear,
    sliding_window: usize,
    device: Device,
    dtype: DType,
}

impl Model {
    /// Build a Qwen2 model with LoRA adapters attached to the selected target
    /// projections. Base weights come from `vb_base` (frozen); adapter
    /// weights come from `vb_lora` (trainable).
    ///
    /// The two VarBuilders should target the same logical tree — the adapter
    /// at path `model.layers.{i}.self_attn.q_proj.lora_A.weight` lives under
    /// `vb_lora`, while the frozen base weight at
    /// `model.layers.{i}.self_attn.q_proj.weight` lives under `vb_base`.
    pub fn new(
        cfg: &Config,
        targets: &HashSet<TargetModule>,
        lora_cfg: &LoRAConfig,
        vb_base: VarBuilder,
        vb_lora: VarBuilder,
    ) -> Result<Self> {
        let vb_m = vb_base.pp("model");
        let vb_lora_m = vb_lora.pp("model");
        let embed_tokens = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_m.pp("embed_tokens"),
        )?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb_base.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let vb_lora_l = vb_lora_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                targets,
                lora_cfg,
                vb_l.pp(layer_idx),
                vb_lora_l.pp(layer_idx),
            )?;
            layers.push(layer)
        }
        let norm = DiffRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if vb_base.contains_tensor("lm_head.weight") {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_base.pp("lm_head"))?
        } else {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.sliding_window,
            device: vb_base.device().clone(),
            dtype: vb_base.dtype(),
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + self.sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    /// Training forward pass — returns full-sequence logits `[B, L, V]` so
    /// the training loop can compute token-level cross-entropy with a loss
    /// mask.
    pub fn forward_train(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, 0)?)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), 0, true)?;
        }
        let xs = xs.apply(&self.norm)?;
        xs.apply(&self.lm_head)
    }

    /// Training forward pass with layer-boundary activation recomputation.
    /// Returns `(logits, ctx, attention_mask)` — `ctx` records the per-layer
    /// saved inputs and detached-output TensorIds needed for the backward
    /// recompute; `attention_mask` is passed back so the backward driver
    /// can re-use the exact same mask during each layer's recompute.
    pub fn forward_train_with_checkpoint(
        &mut self,
        input_ids: &Tensor,
        ctx: &mut crate::checkpoint::CheckpointContext,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, 0)?)
        };
        let xs = self.embed_tokens.forward(input_ids)?;
        let mut current = xs;
        let num_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let input_for_layer = current;
            let out = layer.forward(&input_for_layer, attention_mask.as_ref(), 0, true)?;
            if i + 1 < num_layers {
                // Normal checkpoint boundary: detach output so the next
                // layer sees a leaf and the activations for this layer can
                // be dropped.
                current = ctx.record_boundary(input_for_layer, out)?;
            } else {
                // Last layer: do NOT detach. Its output feeds directly into
                // norm + lm_head + loss, so the backward graph must remain
                // intact from loss through this layer's Vars. We still save
                // the input so the outer backward's grad at this input can
                // be used as upstream for layer N-2's recompute.
                ctx.record_last_layer_input(input_for_layer.detach())?;
                current = out;
            }
        }
        let xs = current.apply(&self.norm)?;
        let logits = xs.apply(&self.lm_head)?;
        Ok((logits, attention_mask))
    }

    /// Drive the backward recompute for a run that used
    /// `forward_train_with_checkpoint`. Must be called AFTER the outer
    /// `loss.backward_into(&mut grads, None)` has populated the
    /// post-layer-stack gradient.
    pub fn backward_through_checkpoints(
        &mut self,
        ctx: &crate::checkpoint::CheckpointContext,
        grads: &mut candle::backprop::GradStore,
        attention_mask: Option<&Tensor>,
    ) -> Result<()> {
        let layers = &mut self.layers;
        ctx.backward_through_segments(grads, |i, saved_input, upstream, grads| {
            // Re-run the layer forward on the detached input; the fresh
            // graph contains the layer's Var weights, so backward_into
            // will accumulate their gradients in `grads`.
            let fresh_out = layers[i].forward(saved_input, attention_mask, 0, true)?;
            fresh_out.backward_into(grads, Some(upstream))?;
            Ok(())
        })
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

/// Parse comma-separated target-module names into a `HashSet`.
pub fn parse_target_modules(s: &str) -> Result<HashSet<TargetModule>> {
    let mut out = HashSet::new();
    for part in s.split(',').map(|p| p.trim()).filter(|p| !p.is_empty()) {
        match TargetModule::from_str(part) {
            Some(t) => {
                out.insert(t);
            }
            None => candle::bail!("unknown target module: {part}"),
        }
    }
    Ok(out)
}

/// Convenience helper: index a full-seq logits tensor `[B, L, V]` at the last
/// token position to mirror the inference-path `ModelForCausalLM::forward`
/// output. Useful for sanity-checking generation during training.
pub fn last_token_logits(logits: &Tensor) -> Result<Tensor> {
    let (_b, seq_len, _v) = logits.dims3()?;
    logits.i((.., seq_len - 1, ..))
}
