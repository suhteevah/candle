//! JSONL dataset loader matching the matt-voice corpus schema.
//!
//! Each line is:
//!   {"context": "...", "matt": "...", "source": "voice-solo" | ..., "sha": "...", ...}
//!
//! We build tokenized training examples by concatenating context + matt with
//! a special separator and a loss-mask that only penalizes the `matt` tokens
//! (standard instruction-tuning pattern — don't train the model to re-emit
//! the prompt).

use anyhow::{Context, Result};
use candle::{Device, Tensor};
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Deserialize)]
pub struct Pair {
    pub context: String,
    pub matt: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub avg_logprob: Option<f32>,
}

pub struct Dataset {
    pairs: Vec<Pair>,
}

impl Dataset {
    pub fn load_jsonl<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).with_context(|| format!("open {path:?}"))?;
        let rdr = std::io::BufReader::new(file);
        let mut pairs = Vec::new();
        use std::io::BufRead;
        for (i, line) in rdr.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let pair: Pair = serde_json::from_str(&line)
                .with_context(|| format!("parse line {} in {path:?}", i + 1))?;
            if pair.matt.trim().is_empty() {
                continue;
            }
            pairs.push(pair);
        }
        Ok(Self { pairs })
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    pub fn get(&self, i: usize) -> &Pair {
        &self.pairs[i]
    }

    pub fn iter(&self) -> impl Iterator<Item = &Pair> {
        self.pairs.iter()
    }
}

/// A single tokenized training example ready for the training loop.
#[derive(Debug, Clone)]
pub struct TokenizedExample {
    /// Full token stream: context_tokens + matt_tokens (+ eos).
    pub input_ids: Vec<u32>,
    /// Parallel mask: 1 where we want to compute loss, 0 elsewhere.
    pub loss_mask: Vec<u8>,
}

/// Tokenize a single pair using the qwen2 chat template.
///
/// Qwen2 chat format:
///   <|im_start|>user\n{context}<|im_end|>\n<|im_start|>assistant\n{matt}<|im_end|>
///
/// Loss mask: only the assistant turn (matt tokens + trailing eos) is scored.
pub fn tokenize_pair(tok: &Tokenizer, pair: &Pair, max_len: usize) -> Result<TokenizedExample> {
    let user_block = format!("<|im_start|>user\n{}<|im_end|>\n", pair.context);
    let assistant_prefix = "<|im_start|>assistant\n";
    let assistant_body = format!("{}<|im_end|>", pair.matt);

    let user_ids = tok
        .encode(user_block, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    let prefix_ids = tok
        .encode(assistant_prefix, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    let body_ids = tok
        .encode(assistant_body, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    let mut input_ids = Vec::with_capacity(user_ids.len() + prefix_ids.len() + body_ids.len());
    let mut loss_mask = Vec::with_capacity(input_ids.capacity());

    input_ids.extend_from_slice(&user_ids);
    loss_mask.extend(std::iter::repeat(0u8).take(user_ids.len()));

    input_ids.extend_from_slice(&prefix_ids);
    loss_mask.extend(std::iter::repeat(0u8).take(prefix_ids.len()));

    input_ids.extend_from_slice(&body_ids);
    loss_mask.extend(std::iter::repeat(1u8).take(body_ids.len()));

    // Right-truncate if needed.
    if input_ids.len() > max_len {
        input_ids.truncate(max_len);
        loss_mask.truncate(max_len);
    }

    Ok(TokenizedExample {
        input_ids,
        loss_mask,
    })
}

/// Build a batched (input_ids, loss_mask) tensor pair on the given device.
/// Batches of size 1 skip padding; larger batches right-pad to max length.
pub fn batch_to_tensors(
    examples: &[TokenizedExample],
    pad_token_id: u32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let max_len = examples.iter().map(|e| e.input_ids.len()).max().unwrap_or(0);
    let bsz = examples.len();

    let mut ids_flat = Vec::with_capacity(bsz * max_len);
    let mut mask_flat = Vec::with_capacity(bsz * max_len);

    for ex in examples {
        ids_flat.extend_from_slice(&ex.input_ids);
        mask_flat.extend_from_slice(&ex.loss_mask);
        let pad = max_len - ex.input_ids.len();
        if pad > 0 {
            ids_flat.extend(std::iter::repeat(pad_token_id).take(pad));
            mask_flat.extend(std::iter::repeat(0u8).take(pad));
        }
    }

    let ids = Tensor::from_vec(ids_flat, (bsz, max_len), device)?;
    let mask = Tensor::from_vec(mask_flat, (bsz, max_len), device)?;
    Ok((ids, mask))
}
