//! Background tokenization + batch prefetch.
//!
//! # Why this exists
//!
//! During a training step, the GPU does ~99% of the wall-clock work
//! (forward + backward). The CPU does tokenization, JSONL parsing, and
//! shuffling. Python DataLoader uses multiprocessing workers for this
//! which are slow (IPC overhead, Python pickling) — in Rust we have real
//! threads with zero marshalling cost.
//!
//! We spawn a worker thread that pulls training pairs from the shuffled
//! index stream, tokenizes them, and pushes ready-to-use
//! `TokenizedExample`s into a bounded channel. The main training loop
//! receives from that channel — when it's waiting on the GPU, the worker
//! is already preparing the NEXT example. Overlap = free wall-clock win.
//!
//! # Bounded queue
//!
//! We cap the queue at ~8 examples. Bigger gives more overlap headroom
//! but costs CPU-side memory (each tokenized Qwen example is ~seq_len *
//! 4 bytes ≈ 2KB at seq=512, so 8 examples = 16KB, negligible).
//!
//! # Graceful shutdown
//!
//! When the training loop exits, the receiver is dropped and the channel
//! closes. The worker's next `send` errors, it breaks, and its thread
//! joins when the Prefetcher drops.

use crate::dataset::{tokenize_pair, Dataset, TokenizedExample};
use anyhow::Result;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::thread::JoinHandle;
use tokenizers::Tokenizer;

pub struct Prefetcher {
    rx: Receiver<Result<TokenizedExample>>,
    worker: Option<JoinHandle<()>>,
}

impl Prefetcher {
    /// Spawn a worker thread that produces tokenized examples sampled from
    /// the dataset (with replacement, shuffled per epoch). `max_seq_len`
    /// sets the right-truncation cap. `queue_capacity` bounds the in-flight
    /// backlog — 8 is a good default.
    pub fn new(
        dataset: Arc<Dataset>,
        tokenizer: Arc<Tokenizer>,
        max_seq_len: usize,
        seed: u64,
        queue_capacity: usize,
    ) -> Self {
        let (tx, rx): (SyncSender<Result<TokenizedExample>>, _) =
            sync_channel(queue_capacity.max(1));

        let worker = std::thread::Builder::new()
            .name("mv-prefetch".into())
            .spawn(move || run_prefetch(dataset, tokenizer, max_seq_len, seed, tx))
            .expect("spawn prefetch thread");

        Self {
            rx,
            worker: Some(worker),
        }
    }

    /// Blocking receive of the next tokenized example. Returns `None` if
    /// the channel has been closed (worker exited).
    pub fn next(&self) -> Option<Result<TokenizedExample>> {
        self.rx.recv().ok()
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        // Dropping rx inside self drops the channel receiver; the worker's
        // next send call will fail and the thread exits. Join to ensure
        // clean teardown before the dataset Arc goes out of scope.
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}

fn run_prefetch(
    dataset: Arc<Dataset>,
    tokenizer: Arc<Tokenizer>,
    max_seq_len: usize,
    seed: u64,
    tx: SyncSender<Result<TokenizedExample>>,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    'epoch: loop {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        indices.shuffle(&mut rng);
        for &i in indices.iter() {
            let pair = dataset.get(i);
            let tok = tokenize_pair(&tokenizer, pair, max_seq_len);
            if tx.send(tok).is_err() {
                // Receiver is gone — training loop exited. Done.
                break 'epoch;
            }
        }
    }
}
