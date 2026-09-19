# CTC training trial

User authorized a Google-baseline training trial. Preserve the previous seq2seq
probe and all sources/ files. This is an independent local implementation, not
official training code or a claim of exact reproduction.

1. Verify full official archive checksum; stream InkML into an on-disk SQLite
   feature cache. Use train+synthetic only for fitting; valid for measurement;
   leave test untouched. Preserve pen-up displacement and curve time features.
2. Approximate Carbune's cubic Bézier representation with endpoint-constrained
   least squares, recursive error/length splitting. Document numerical choices.
3. Train 11 encoder layers, 512 model width, 8 heads, Swish, dropout .15, CTC,
   Adam lr .001. Standard 64-dimensional heads and FFN 2048 are explicit local
   choices: cited table says 256 units/head, inconsistent with its stated 35M
   parameters under standard attention. No hidden claim of exact architecture.
   Start with pre-LN plus final normalization: the initial post-LN smoke run
   empirically collapsed (near-zero input gradient after 100 steps).
4. Test CTC repeat/blank behavior, padding isolation, feature invariance and
   finite backward pass. Profile full optimizer update on available GPU.
5. Use microbatches and accumulation for effective batch 256 where practical,
   BF16 forward with FP32 CTC. Run a bounded pilot, report validation token edit
   error and exact-match rather than teacher-forced accuracy. Save restartable
   optimizer/model checkpoints and actual memory/time measurements.

The paper's 100,000 updates are a full training budget, not a promise to finish
that budget in this initial trial. Do not estimate accuracy from a smoke test.
