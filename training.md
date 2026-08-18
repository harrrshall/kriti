# training and compression recipe

kriti starts from
`ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large` at revision
`cd09ba7720f3b17d259f6bfd03e1463bc5ba517d`. the upstream encoder and rnnt
weights remain unchanged in this recipe. attribution and the upstream model
card remain part of the release. the complete lineage is in
[references.md](references.md).

## data profile

the preparation profile contains openslr 54, openslr 43, fleurs nepali, and
indicvoices nepali. the full profile contains 402,905 accepted clips and
630.2761 hours; its training split contains 393,002 clips and 608.9182 hours.
audio is decoded to mono 16 khz, text receives unicode nfc and whitespace
normalization, and speaker or recording groups are kept within one split. raw
audio and row manifests remain in the licensed data environment. see
[data.md](data.md).

## deterministic model construction

1. download the pinned base archive and verify sha-256
   `8c75fee5d82e61d31ddc4e891875617f730b4dfef0f4b226ff1e576bf4143039`.
2. restore it with the pinned ai4bharat nemo revision
   `8dce88cf8e94963e2033c3137f7b9993b51db88a`.
3. delete the ctc decoder and every rnnt joint head except `ne`.
4. save `kriti.nemo`; at inference, replace the 5,633-row prediction embedding
   with the first 257 rows and verify 119,461,121 live asr parameters.

`scripts/export_model.py` implements steps 2 and 3. the runtime compaction stays
in the loader because that is the artifact form that was reproduced and
benchmarked. locally rebuilt archives receive a new byte hash; pass
`verify_release_hashes=False` to `KritiASR.from_files` only for artifacts you
built and trust. hub downloads remain hash-verified by default.

## punctuation head training

the conformer is frozen. training examples are selected deterministically with
seed 20,260,813. fleurs provides the positive punctuation domain, while up to
3,000 deterministic negatives per other source are used for fitting and 1,000
per source for validation. the recorded fit used 10,294 rows and 3,093
validation rows.

for each utterance, the frozen encoder produces a 512-wide sequence. masked
mean and standard deviation pooling gives 1,024 values. a balanced logistic
regression model uses `lbfgs`, `c=1.0`, and at most 2,000 iterations. threshold
search selects the highest recall candidate satisfying precision at least
0.75, recall at least 0.75, and false-positive rate at most 0.005. the selected
threshold is 0.711. `scripts/train_punctuation_head.py` writes portable json.

## hardware

the compression and evaluation workflow used three independent nvidia h200
workers, each with 141 gb gpu memory, 300 gb host memory, and 28 cpu cores.
encoder feature extraction used one gpu. the 1,025-parameter logistic fit ran on
cpu after features were extracted. the recipe covers construction and evaluation
of the derived kriti graph. upstream pretraining remains part of the credited
ai4bharat model lineage.

## commands

```bash
python scripts/export_model.py \
  --source indicconformer_stt_ne_hybrid_rnnt_large.nemo \
  --output kriti.nemo

python scripts/train_punctuation_head.py \
  --model kriti.nemo \
  --train-view train.jsonl \
  --output punctuation_head.json
```

the input jsonl uses `audio`, `sample_id`, and `source_id`. audio paths remain
local to the licensed data environment. freeze output hashes before evaluation.
