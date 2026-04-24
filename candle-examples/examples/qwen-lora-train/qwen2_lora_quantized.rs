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
        rms_norm_slow(xs, &self.weight, self.eps)
    }
}

struct QLoRAAttention {
    q: QLoRAProjection,
    k: QLoRAProjection,
    v: QLoRAProjection,
    o: QLoRAProjection,
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
        let q = self.q.forward(xs, training)?;
        let k = self.k.forward(xs, training)?;
        let v = self.v.forward(xs, training)?;

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
            let sm = candle_nn::ops::softmax(&att32, D::Minus1)?;
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
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        targets: &HashSet<TargetModule>,
        lora_cfg: &LoRAConfig,
        vb_lora: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
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
                QMatMul::from_qtensor(wq)?,
                Some(bq),
                embedding_length,
                head_count * head_dim,
                TargetModule::QProj,
                targets,
                lora_cfg,
                vb_l.pp("q_proj"),
            )?;
            let k = QLoRAProjection::new(
                QMatMul::from_qtensor(wk)?,
                Some(bk),
                embedding_length,
                head_count_kv * head_dim,
                TargetModule::KProj,
                targets,
                lora_cfg,
                vb_l.pp("k_proj"),
            )?;
            let v = QLoRAProjection::new(
                QMatMul::from_qtensor(wv)?,
                Some(bv),
                embedding_length,
                head_count_kv * head_dim,
                TargetModule::VProj,
                targets,
                lora_cfg,
                vb_l.pp("v_proj"),
            )?;
            let o = QLoRAProjection::new(
                QMatMul::from_qtensor(wo)?,
                None,
                head_count * head_dim,
                embedding_length,
                TargetModule::OProj,
                targets,
                lora_cfg,
                vb_l.pp("o_proj"),
            )?;
            let attn = QLoRAAttention {
                q,
                k,
                v,
                o,
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
                QMatMul::from_qtensor(w_gate)?,
                None,
                embedding_length,
                inter_sz,
                TargetModule::GateProj,
                targets,
                lora_cfg,
                vb_mlp.pp("gate_proj"),
            )?;
            let up = QLoRAProjection::new(
                QMatMul::from_qtensor(w_up)?,
                None,
                embedding_length,
                inter_sz,
                TargetModule::UpProj,
                targets,
                lora_cfg,
                vb_mlp.pp("up_proj"),
            )?;
            let down = QLoRAProjection::new(
                QMatMul::from_qtensor(w_down)?,
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
}
