//! Qwen2 model with **quantized** base weights + LoRA adapters.
//!
//! This is the QLoRA path: base model stays in quantized GGUF format in
//! VRAM (Qwen2.5-7B Q4_K_M = ~4.5GB instead of ~15GB bf16), only the
//! LoRA adapters hold trainable state. Gradient flows through QMatMul's
//! new backward (added to candle-core in this branch) into the adapters
//! via the frozen-but-differentiable base.
//!
//! # Structure
//!
//! Ported from `candle-transformers::models::quantized_qwen2` with two
//! modifications: (a) `DiffRmsNorm` + `rope_slow` + `ops::softmax`
//! substitutions so the forward-through-backward training path doesn't
//! hit no_bwd traps; (b) LoRAProjection wrapping each of q/k/v/o and
//! optionally MLP projections, with adapter output added to the base
//! projection's output.
//!
//! # Load path
//!
//! Expects a GGUF file (single or sharded) as produced by llama.cpp's
//! quantize tool from a Qwen2-architecture HF repo.

use crate::lora::{LoRAConfig, LoRALinear};
use candle::quantized::{gguf_file, QMatMul};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{ops::rms_norm_slow, VarBuilder};
use std::collections::{HashMap, HashSet};

use crate::qwen2_lora::TargetModule;

/// LoRA projection over a quantized base weight. Same shape as
/// qwen2_lora::LoRAProjection but base is `QMatMul` (frozen) and adapter
/// is `Option<LoRALinear>` (trainable).
struct QLoRAProjection {
    base: QMatMul,
    /// Optional additive bias tensor (dequantized up front, stays on device).
    bias: Option<Tensor>,
    adapter: Option<LoRALinear>,
}

impl QLoRAProjection {
    fn new(
        base: QMatMul,
        bias: Option<Tensor>,
        in_features: usize,
        out_features: usize,
        target: TargetModule,
        targets: &HashSet<TargetModule>,
        lora_cfg: &LoRAConfig,
        vb_lora: VarBuilder,
    ) -> Result<Self> {
        let adapter = if targets.contains(&target) {
            Some(LoRALinear::new(in_features, out_features, lora_cfg, vb_lora)?)
        } else {
            None
        };
        Ok(Self {
            base,
            bias,
            adapter,
        })
    }

    fn forward(&self, xs: &Tensor, training: bool) -> Result<Tensor> {
        let mut y = self.base.forward(xs)?;
        if let Some(b) = &self.bias {
            y = y.broadcast_add(b)?;
        }
        if let Some(a) = &self.adapter {
            let delta = a.forward_delta(xs, training)?;
            y = y.broadcast_add(&delta)?;
        }
        Ok(y)
    }

    /// Returns the base weight as an f16 Tensor when the base is stored
    /// as `QMatMul::TensorF16` (prequantize_base path). Returns None if
    /// the base is still a quantized QTensor — caller must dequantize
    /// first or take a different path.
    fn base_f16_weight(&self) -> Option<Tensor> {
        match &self.base {
            QMatMul::TensorF16(t) => Some(t.clone()),
            _ => None,
        }
    }

    /// Variant of `forward` for the fused-QKV path: skips the base matmul
    /// (caller already computed it from a stacked weight), adds bias and
    /// adapter delta on top of the supplied `base_out`. The adapter still
    /// needs the original `xs` for its forward, so we pass it explicitly.
    fn forward_post_base(&self, base_out: Tensor, xs: &Tensor, training: bool) -> Result<Tensor> {
        let mut y = base_out;
        if let Some(b) = &self.bias {
            y = y.broadcast_add(b)?;
        }
        if let Some(a) = &self.adapter {
            let delta = a.forward_delta(xs, training)?;
            y = y.broadcast_add(&delta)?;
        }
        Ok(y)
    }

    /// Fold the LoRA adapter (if any) into the base weight. After merge,
    /// `forward` skips the adapter compute entirely — the base alone
    /// produces the same output. Used at inference deployment time.
    ///
    /// Mechanics:
    ///   `W_eff = W_base + scaling * B @ A`
    ///   `forward(x) = W_eff @ x + bias`
    ///
    /// The base is converted to (or kept as) `QMatMul::TensorF16` since the
    /// merged weight is no longer quantized — once we add an fp adapter
    /// delta, the result is no longer a Q4_K_M-clean weight tensor.
    fn merge_adapter(&mut self, device: &Device) -> Result<()> {
        let Some(adapter) = self.adapter.take() else {
            return Ok(()); // no adapter, nothing to merge
        };
        // Get base weight as a regular f16 tensor (same shape as quantized:
        // [out_features, in_features]).
        let w_base_f16 = match &self.base {
            QMatMul::TensorF16(t) => t.clone(),
            QMatMul::Tensor(t) => t.to_dtype(DType::F16)?,
            QMatMul::QTensor(qt) => qt.dequantize_f16(device)?,
        };
        // Compute scaling * B @ A in adapter dtype (typically fp32), then
        // cast to f16 and add. The double cast keeps the matmul precise.
        let delta = adapter.merged_weight_delta()?;
        let delta_f16 = if delta.dtype() == DType::F16 {
            delta
        } else {
            delta.to_dtype(DType::F16)?
        };
        let merged = w_base_f16.add(&delta_f16)?;
        self.base = QMatMul::TensorF16(merged);
        Ok(())
    }
}

struct QLoRAMlp {
    gate: QLoRAProjection,
    up: QLoRAProjection,
    down: QLoRAProjection,
}

impl QLoRAMlp {
    fn forward(&self, xs: &Tensor, training: bool) -> Result<Tensor> {
        let lhs = self.gate.forward(xs, training)?;
        let lhs = candle_nn::ops::silu(&lhs)?;
        let rhs = self.up.forward(xs, training)?;
        self.down.forward(&(lhs * rhs)?, training)
    }
}

struct DiffRmsNorm {
    weight: Tensor,
    eps: f32,
}

impl DiffRmsNorm {
    fn from_qtensor(qt: std::sync::Arc<candle::quantized::QTensor>, eps: f32) -> Result<Self> {
        let weight = qt.dequantize(&qt.device())?;
        Ok(Self { weight, eps })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        crate::fused_ops::fused_rms_norm(xs, &self.weight, self.eps)
    }
}

struct QLoRAAttention {
    q: QLoRAProjection,
    k: QLoRAProjection,
    v: QLoRAProjection,
    o: QLoRAProjection,
    /// Optional fused-QKV base weight, shape `(n_head*head_dim + 2*n_kv_head*head_dim, hidden)`.
    /// When present, the attention forward computes Q/K/V via a single matmul
    /// against this stacked weight instead of three independent matmuls
    /// against `q.base`, `k.base`, `v.base`. Bias and adapter delta are still
    /// applied per-projection via `forward_post_base`. Built only when both
    /// `--prequantize-base` and `--fuse-qkv` are set, since stacking quantized
    /// weights doesn't make sense (each is independently block-quantized).
    qkv_fused: Option<QMatMul>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    hidden: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl QLoRAAttention {
    fn forward(
        &mut self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        training: bool,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _n_embd) = xs.dims3()?;
        // Fused QKV: when a stacked weight is present, compute the Q+K+V
        // base outputs via a single matmul, then split. Bias + LoRA adapter
        // delta are still applied per-projection via `forward_post_base`.
        // Otherwise fall through to the three independent forwards.
        let (q, k, v) = if let Some(qkv_w) = &self.qkv_fused {
            let q_dim = self.n_head * self.head_dim;
            let kv_dim = self.n_kv_head * self.head_dim;
            let qkv = qkv_w.forward(xs)?; // [B, L, q_dim + 2*kv_dim]
            let q_out = qkv.narrow(D::Minus1, 0, q_dim)?;
            let k_out = qkv.narrow(D::Minus1, q_dim, kv_dim)?;
            let v_out = qkv.narrow(D::Minus1, q_dim + kv_dim, kv_dim)?;
            (
                self.q.forward_post_base(q_out, xs, training)?,
                self.k.forward_post_base(k_out, xs, training)?,
                self.v.forward_post_base(v_out, xs, training)?,
            )
        } else {
            (
                self.q.forward(xs, training)?,
                self.k.forward(xs, training)?,
                self.v.forward(xs, training)?,
            )
        };

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q = candle_nn::rotary_emb::rope_slow(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope_slow(&k.contiguous()?, &cos, &sin)?;

        let (k, v) = if training {
            (k, v)
        } else {
            match &self.kv_cache {
                None => (k, v),
                Some((kc, vc)) => {
                    if index_pos == 0 {
                        (k, v)
                    } else {
                        (Tensor::cat(&[kc, &k], 2)?, Tensor::cat(&[vc, &v], 2)?)
                    }
                }
            }
        };
        if !training {
            self.kv_cache = Some((k.clone(), v.clone()));
        }

        let k = candle_transformers::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = candle_transformers::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(m) => {
                let m = m.broadcast_as(att.shape())?;
                let neg = self.neg_inf.broadcast_as(att.shape().dims())?;
                m.where_cond(&neg, &att)?
            }
        };
        let att = {
            let orig = att.dtype();
            let att32 = if orig == DType::F32 {
                att
            } else {
                att.to_dtype(DType::F32)?
            };
            let sm = crate::fused_ops::fused_softmax_last_dim(&att32)?;
            if orig == DType::F32 {
                sm
            } else {
                sm.to_dtype(orig)?
            }
        };
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, self.hidden])?;
        self.o.forward(&y, training)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

struct DecoderLayer {
    attn: QLoRAAttention,
    mlp: QLoRAMlp,
    attn_norm: DiffRmsNorm,
    ffn_norm: DiffRmsNorm,
}

impl DecoderLayer {
    fn forward(
        &mut self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        training: bool,
    ) -> Result<Tensor> {
        let residual = xs;
        let x = self.attn_norm.forward(xs)?;
        let x = self.attn.forward(&x, mask, index_pos, training)?;
        let x = (x + residual)?;
        let residual = &x;
        let y = self.ffn_norm.forward(&x)?;
        let y = self.mlp.forward(&y, training)?;
        residual + y
    }

    fn clear_kv_cache(&mut self) {
        self.attn.clear_kv_cache();
    }
}

pub struct Model {
    tok_embeddings: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: DiffRmsNorm,
    output: QMatMul,
    masks: HashMap<(usize, usize), Tensor>,
    device: Device,
    dtype: DType,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    ctx: usize,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), dev)?;
    let idx = Tensor::arange(0, ctx as u32, dev)?
        .to_dtype(DType::F32)?
        .reshape((ctx, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx.cos()?, idx.sin()?))
}

impl Model {
    /// Load a Qwen2 model from GGUF + attach LoRA adapters using a
    /// trainable VarBuilder (typically backed by a fresh VarMap).
    ///
    /// `prequantize_base`: when true, every base projection weight is
    /// dequantized to f16 at load time and stored as a regular `Tensor`
    /// instead of a `QTensor`. This trades persistent VRAM for backward
    /// speed: the QMatMul backward dequantize-on-the-fly path is skipped
    /// entirely, replaced by a standard fp16 matmul that autograd
    /// handles natively. Empirically 2-3× faster QLoRA backward at the
    /// cost of ~2× more persistent base VRAM (e.g. 4.5GB Q4_K_M -> ~14GB
    /// f16 for Qwen2.5-7B). Recommended for >=16GB cards (P100 16GB,
    /// V100, A4000+, RTX 4090, etc.). Leave off for 8GB cards where the
    /// quantized base is required to fit at all.
    ///
    /// `fuse_qkv`: when true, stack the q/k/v base projection weights into
    /// a single fused matmul per attention layer. 3 matmuls → 1, ~5-10%
    /// attention speedup. Requires `prequantize_base=true` since stacking
    /// quantized weights is meaningless (each block-quantized independently).
    /// Bias and LoRA adapter deltas are still applied per-projection.
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        targets: &HashSet<TargetModule>,
        lora_cfg: &LoRAConfig,
        vb_lora: VarBuilder,
        device: &Device,
        prequantize_base: bool,
        fuse_qkv: bool,
    ) -> Result<Self> {
        if fuse_qkv && !prequantize_base {
            candle::bail!(
                "fuse_qkv requires prequantize_base=true (cannot stack quantized weights)"
            );
        }
        // Helper: build a QMatMul from a raw QTensor, optionally pre-
        // dequantizing to f16. Centralizing this makes the prequantize
        // semantics impossible to forget for any individual weight.
        let make_base = |qt: candle::quantized::QTensor| -> Result<QMatMul> {
            if prequantize_base {
                let t_f16 = qt.dequantize_f16(device)?;
                Ok(QMatMul::TensorF16(t_f16))
            } else {
                QMatMul::from_qtensor(qt)
            }
        };
        let md = |s: &str| {
            ct.metadata
                .get(s)
                .ok_or_else(|| candle::Error::Msg(format!("missing GGUF metadata: {s}")))
        };
        let head_count = md("qwen2.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md("qwen2.attention.head_count_kv")?.to_u32()? as usize;
        let embedding_length = md("qwen2.embedding_length")?.to_u32()? as usize;
        let context_length = md("qwen2.context_length")?.to_u32()? as usize;
        let block_count = md("qwen2.block_count")?.to_u32()? as usize;
        let rms_eps = md("qwen2.attention.layer_norm_rms_epsilon")?.to_f32()?;
        let rope_freq = md("qwen2.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);
        let head_dim = embedding_length / head_count;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_deq = tok.dequantize(device)?;
        let tok_embeddings = candle_nn::Embedding::new(tok_deq, embedding_length);

        let norm = DiffRmsNorm::from_qtensor(
            std::sync::Arc::new(ct.tensor(reader, "output_norm.weight", device)?),
            rms_eps,
        )?;
        // Output layer (lm_head): pre-dequantize to fp16 at load time.
        // Rationale: this is the largest single weight (hidden × vocab ≈
        // 2.2GB fp32 for 7B). If left quantized, every backward pass
        // through it would transiently allocate a full fp32 dequant tensor
        // and OOM an 8GB card. Pre-dequantizing to F16 costs 1.1GB
        // persistent VRAM but avoids the transient spike — net win.
        let output = {
            let qt = ct
                .tensor(reader, "output.weight", device)
                .or_else(|_| ct.tensor(reader, "token_embd.weight", device))?;
            let t_f16 = qt.dequantize_f16(device)?;
            QMatMul::TensorF16(t_f16)
        };
        let (cos, sin) = precomput_freqs_cis(head_dim, rope_freq, context_length, device)?;

        let vb_layers = vb_lora.pp("model").pp("layers");
        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let prefix = format!("blk.{i}");
            let wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let wo = ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            let bq = ct
                .tensor(reader, &format!("{prefix}.attn_q.bias"), device)?
                .dequantize(device)?;
            let bk = ct
                .tensor(reader, &format!("{prefix}.attn_k.bias"), device)?
                .dequantize(device)?;
            let bv = ct
                .tensor(reader, &format!("{prefix}.attn_v.bias"), device)?
                .dequantize(device)?;

            let vb_l = vb_layers.pp(i).pp("self_attn");
            let q = QLoRAProjection::new(
                make_base(wq)?,
                Some(bq),
                embedding_length,
                head_count * head_dim,
                TargetModule::QProj,
                targets,
                lora_cfg,
                vb_l.pp("q_proj"),
            )?;
            let k = QLoRAProjection::new(
                make_base(wk)?,
                Some(bk),
                embedding_length,
                head_count_kv * head_dim,
                TargetModule::KProj,
                targets,
                lora_cfg,
                vb_l.pp("k_proj"),
            )?;
            let v = QLoRAProjection::new(
                make_base(wv)?,
                Some(bv),
                embedding_length,
                head_count_kv * head_dim,
                TargetModule::VProj,
                targets,
                lora_cfg,
                vb_l.pp("v_proj"),
            )?;
            let o = QLoRAProjection::new(
                make_base(wo)?,
                None,
                head_count * head_dim,
                embedding_length,
                TargetModule::OProj,
                targets,
                lora_cfg,
                vb_l.pp("o_proj"),
            )?;
            // Build the fused QKV weight when --fuse-qkv is on. Requires
            // prequantize_base (already enforced at function entry); we
            // pull the f16 base tensors from each projection and stack
            // them along the output (row) dim to form a single
            // (q_out + k_out + v_out, hidden) weight that the attention
            // forward dispatches against in one matmul instead of three.
            let qkv_fused = if fuse_qkv {
                let q_w = q.base_f16_weight().ok_or_else(|| {
                    candle::Error::Msg("fuse_qkv: q.base is not TensorF16; \
                        prequantize_base must produce f16 bases".into())
                })?;
                let k_w = k.base_f16_weight().ok_or_else(|| {
                    candle::Error::Msg("fuse_qkv: k.base is not TensorF16".into())
                })?;
                let v_w = v.base_f16_weight().ok_or_else(|| {
                    candle::Error::Msg("fuse_qkv: v.base is not TensorF16".into())
                })?;
                let stacked = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;
                Some(QMatMul::TensorF16(stacked))
            } else {
                None
            };

            let attn = QLoRAAttention {
                q,
                k,
                v,
                o,
                qkv_fused,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                hidden: embedding_length,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
            };

            // MLP
            let w_gate = ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
            let w_up = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
            let w_down = ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
            let inter_sz = w_gate.shape().dims2()?.0; // QTensor shape is (N, K)
            let vb_mlp = vb_layers.pp(i).pp("mlp");
            let gate = QLoRAProjection::new(
                make_base(w_gate)?,
                None,
                embedding_length,
                inter_sz,
                TargetModule::GateProj,
                targets,
                lora_cfg,
                vb_mlp.pp("gate_proj"),
            )?;
            let up = QLoRAProjection::new(
                make_base(w_up)?,
                None,
                embedding_length,
                inter_sz,
                TargetModule::UpProj,
                targets,
                lora_cfg,
                vb_mlp.pp("up_proj"),
            )?;
            let down = QLoRAProjection::new(
                make_base(w_down)?,
                None,
                inter_sz,
                embedding_length,
                TargetModule::DownProj,
                targets,
                lora_cfg,
                vb_mlp.pp("down_proj"),
            )?;
            let mlp = QLoRAMlp { gate, up, down };

            let attn_norm = DiffRmsNorm::from_qtensor(
                std::sync::Arc::new(ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?),
                rms_eps,
            )?;
            let ffn_norm = DiffRmsNorm::from_qtensor(
                std::sync::Arc::new(ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?),
                rms_eps,
            )?;

            layers.push(DecoderLayer {
                attn,
                mlp,
                attn_norm,
                ffn_norm,
            });
        }

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            masks: HashMap::new(),
            device: device.clone(),
            dtype: DType::F32, // dequantized norm weights are f32; activations follow.
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(m) = self.masks.get(&(seq_len, kv_len)) {
            return Ok(m.clone());
        }
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > i + index_pos)))
            .collect();
        let mask = Tensor::from_slice(&mask, (seq_len, kv_len), &self.device)?;
        self.masks.insert((seq_len, kv_len), mask.clone());
        Ok(mask)
    }

    /// Inference forward (single or full seq, kv cache-aware).
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos)?)
        };
        let mut xs = self.tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, mask.as_ref(), index_pos, false)?;
        }
        let xs = self.norm.forward(&xs)?;
        let xs = xs.i((.., seq_len - 1, ..))?;
        self.output.forward(&xs)
    }

    /// Training forward — returns post-norm hidden `[B, L, H]`.
    pub fn forward_train(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(self.mask(seq_len, 0)?)
        };
        let mut xs = self.tok_embeddings.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, mask.as_ref(), 0, true)?;
        }
        self.norm.forward(&xs)
    }

    /// Training forward with layer-boundary checkpointing — same
    /// detach-all-but-last pattern as the non-quantized variant.
    pub fn forward_train_with_checkpoint(
        &mut self,
        input_ids: &Tensor,
        ctx: &mut crate::checkpoint::CheckpointContext,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(self.mask(seq_len, 0)?)
        };
        let xs = self.tok_embeddings.forward(input_ids)?;
        let mut current = xs;
        let n = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let input_for_layer = current;
            let out = layer.forward(&input_for_layer, mask.as_ref(), 0, true)?;
            if i + 1 < n {
                current = ctx.record_boundary(input_for_layer, out)?;
            } else {
                ctx.record_last_layer_input(input_for_layer.detach())?;
                current = out;
            }
        }
        let hidden = self.norm.forward(&current)?;
        Ok((hidden, mask))
    }

    pub fn backward_through_checkpoints(
        &mut self,
        ctx: &crate::checkpoint::CheckpointContext,
        grads: &mut candle::backprop::GradStore,
        mask: Option<&Tensor>,
    ) -> Result<()> {
        let layers = &mut self.layers;
        ctx.backward_through_segments(grads, |i, saved_input, upstream, grads| {
            let fresh_out = layers[i].forward(saved_input, mask, 0, true)?;
            fresh_out.backward_into(grads, Some(upstream))?;
            Ok(())
        })
    }

    pub fn output_matmul(&self) -> &QMatMul {
        &self.output
    }

    pub fn clear_kv_cache(&mut self) {
        for l in self.layers.iter_mut() {
            l.clear_kv_cache();
        }
    }

    /// Fold every LoRA adapter into its base projection weight, then drop
    /// the adapters. After this, `forward` does plain `base_matmul + bias`
    /// per projection — no per-token adapter compute. Used at inference
    /// deployment time after loading the trained adapter weights.
    ///
    /// Trade-offs:
    ///   + 10-20% faster decode (one matmul per projection instead of
    ///     base_matmul + A_matmul + B_matmul + scaling + add).
    ///   + Frees adapter VRAM (~10-20 MB at typical ranks).
    ///   + Quantized base bytes get replaced by f16 base bytes — this
    ///     INCREASES VRAM since the merged weight can no longer be
    ///     represented as a Q4_K_M tensor without quality loss. For 7B
    ///     this is ~+10 GB persistent. If memory is tight, leave this
    ///     OFF and accept the slight per-token overhead.
    ///
    /// Mirrors PEFT's `model.merge_and_unload()` and llama.cpp's
    /// `llama-export-lora` for parity.
    pub fn merge_adapters_into_base(&mut self) -> Result<()> {
        let device = self.device.clone();
        for layer in self.layers.iter_mut() {
            layer.attn.q.merge_adapter(&device)?;
            layer.attn.k.merge_adapter(&device)?;
            layer.attn.v.merge_adapter(&device)?;
            layer.attn.o.merge_adapter(&device)?;
            layer.mlp.gate.merge_adapter(&device)?;
            layer.mlp.up.merge_adapter(&device)?;
            layer.mlp.down.merge_adapter(&device)?;

            // If this layer was using fused-QKV, the stacked weight was
            // built from the un-merged q/k/v bases. After merge, the
            // per-projection bases now include the adapter delta; the
            // stacked weight would be stale. Rebuild it from the
            // post-merge per-projection f16 weights.
            if layer.attn.qkv_fused.is_some() {
                let q_w = layer.attn.q.base_f16_weight().ok_or_else(|| {
                    candle::Error::Msg(
                        "merge+fuse_qkv: q.base is not TensorF16 after merge".into(),
                    )
                })?;
                let k_w = layer.attn.k.base_f16_weight().ok_or_else(|| {
                    candle::Error::Msg(
                        "merge+fuse_qkv: k.base is not TensorF16 after merge".into(),
                    )
                })?;
                let v_w = layer.attn.v.base_f16_weight().ok_or_else(|| {
                    candle::Error::Msg(
                        "merge+fuse_qkv: v.base is not TensorF16 after merge".into(),
                    )
                })?;
                let stacked = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;
                layer.attn.qkv_fused = Some(QMatMul::TensorF16(stacked));
            }
        }
        Ok(())
    }
}
