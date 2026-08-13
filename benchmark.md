# benchmark

the public snapshot uses a frozen 3,630-utterance nepali development view with
757 openslr 54 rows, 304 fleurs nepali rows, and 2,569 indicvoices nepali rows.
the ordered view sha-256 is
`2374cac54831ce9c69282503763d7f1e12ada0404ae34ed471a7538cdae6c61f`.
each listed system was loaded from scratch twice. a system enters the table only
when both complete prediction files have identical sha-256 values. the primary
metric is punctuation-insensitive word error rate. exact primary error ratios
share a rank; raw wer is descriptive and never breaks a tie.

| rank | model | pi wer | pi cer | raw wer | raw cer |
|---:|---|---:|---:|---:|---:|
| 1 | <mark>kriti</mark> | <mark>24.0773%</mark> | <mark>8.2877%</mark> | <mark>24.6854%</mark> | <mark>8.4042%</mark> |
| 1 | ai4bharat nepali indicconformer hybrid, rnnt | 24.0773% | 8.2877% | 25.1928% | 8.5011% |
| 3 | ai4bharat nepali indicconformer hybrid, ctc | 25.3109% | 8.4515% | 26.4313% | 8.6656% |
| 4 | qwen3 asr nepali fine-tuned | 52.4043% | 24.2176% | 55.5196% | 24.9438% |
| 5 | whisper large v3 nepali, kiranpantha | 55.7059% | 24.9413% | 57.8369% | 25.3969% |
| 6 | whisper large v3 nepali openslr | 55.7678% | 24.6705% | 57.4514% | 25.0608% |
| 7 | whisper medium nepali, paudel et al. | 58.4027% | 27.6389% | 60.8634% | 28.2311% |
| 8 | meta mms 1b all, nepali adapter | 60.1274% | 22.7855% | 61.2749% | 23.0108% |
| 9 | mms 1b nepali, paudel et al. | 60.7661% | 24.7837% | 63.2307% | 25.3560% |
| 10 | xls-r 300m nepali, shniranjan | 62.8900% | 33.4353% | 63.9039% | 33.5927% |

the full snapshot contains 19 fully evaluated systems. `benchmark.json`
contains exact unrounded values for the leading ten. kriti is joint first on the
primary metric and has the lowest raw wer in this evaluated set. this is a
development view used during model selection, not an untouched test claim.

## benchmark protocol

1. obtain the source data under its original terms and construct the exact
   source counts above with the policies in [data.md](data.md).
2. pin every model revision and decoder configuration before loading weights.
3. run `scripts/evaluate.py` twice per model with deterministic torch settings.
4. require all 3,630 ordered sample ids and references to match the frozen view
   sha-256 above and its exact source composition.
5. require the two prediction files for each model to have equal sha-256 values.
6. recompute aggregate metrics with `kriti.metrics`; sort exact
   punctuation-insensitive error ratios and assign competition ranks.

the complete registry, immutable revisions, decoder settings, and replicate
prediction hashes are in `benchmark.json`. the view itself is not redistributed
because one source is gated. results do
not imply performance on every nepali domain, streaming audio, or an untouched
product evaluation set.

## scale efficiency

kriti reaches the leading accuracy tier with 119.46m live parameters. in this
snapshot it ranks ahead of the fine-tuned qwen3 nepali asr model, whisper large v3 variants, meta
mms 1b, and other substantially larger open systems. the official ai4bharat
nepali rnnt shares the same primary score, while kriti records the lower raw
wer through its acoustic danda head.
