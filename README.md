<div align="center">

# kriti

### a new efficiency frontier for open nepali asr

kriti is a 119m-parameter nepali-only rnnt that shares rank 1 on our 19-system development benchmark and records the lowest raw wer in the evaluated field.

[model weights](https://huggingface.co/harrrshall/kriti) · [model card](model-card.md) · [architecture](architecture.md) · [training](training.md) · [benchmark data](benchmark.json)

</div>

| live parameters | primary pi wer | raw wer | evaluated field |
|---:|---:|---:|---:|
| 119m | 24.0773% | 24.6854% | 19 systems |

## benchmark

on the frozen 3,630-utterance development view, kriti shares the primary score with the official ai4bharat nepali rnnt and records the lower raw wer. the table shows the leading 10 of 19 fully evaluated systems.

| rank | model | pi wer | pi cer | raw wer |
|---:|---|---:|---:|---:|
| 1 | <mark>kriti, 119m</mark> | <mark>24.0773%</mark> | <mark>8.2877%</mark> | <mark>24.6854%</mark> |
| 1 | ai4bharat nepali indicconformer, rnnt | 24.0773% | 8.2877% | 25.1928% |
| 3 | ai4bharat nepali indicconformer, ctc | 25.3109% | 8.4515% | 26.4313% |
| 4 | qwen3 asr nepali, fine-tuned | 52.4043% | 24.2176% | 55.5196% |
| 5 | whisper large v3 nepali, kiranpantha | 55.7059% | 24.9413% | 57.8369% |
| 6 | whisper large v3 nepali, openslr | 55.7678% | 24.6705% | 57.4514% |
| 7 | whisper medium nepali, paudel et al. | 58.4027% | 27.6389% | 60.8634% |
| 8 | meta mms 1b all, nepali adapter | 60.1274% | 22.7855% | 61.2749% |
| 9 | mms 1b nepali, paudel et al. | 60.7661% | 24.7837% | 63.2307% |
| 10 | xls-r 300m nepali, shniranjan | 62.8900% | 33.4353% | 63.9039% |

pi wer is punctuation-insensitive word error rate. exact primary error ratios share a rank, and raw wer never breaks that tie. the fine-tuned nepali qwen model is included at rank 4. read the [full benchmark note](benchmark.md) or inspect the [exact model revisions and replicate hashes](benchmark.json).

## architecture

```text
16 khz audio -> log-mel -> conformer encoder -> nepali rnnt -> transcript
                                  |
                                  +-> pooled encoder states -> danda head -> final text
```

| component | role |
|---|---|
| front end | 16 khz audio, 80 log-mel bins, 4x time reduction |
| encoder | 17 conformer blocks at width 512 |
| rnnt | one-layer predictor at width 640 with 257 nepali output rows |
| danda head | 1,024 pooled encoder values, 1,025 trainable parameters, threshold 0.711 |
| live graph | 119m parameters after nepali-only runtime pruning |

the rnnt creates the transcript while the tiny acoustic head decides whether terminal devanagari danda is needed. see the [deep architecture note](architecture.md).

## training recipe

the public asr graph starts from the mit-licensed [ai4bharat nepali indicconformer](https://huggingface.co/ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large) at pinned revision `cd09ba7720f3b17d259f6bfd03e1463bc5ba517d`. its encoder and nepali rnnt weights are retained; this release trains only the terminal punctuation head.

1. verify the pinned base archive and ai4bharat nemo runtime revision.
2. retain the conformer encoder and nepali rnnt, then remove ctc and 21 other language heads.
3. reduce the prediction embedding from 5,633 rows to 257 rows at load time.
4. freeze the encoder and pool each sequence into 1,024 mean and standard deviation values.
5. fit a balanced logistic head with seed `20260813`, then freeze threshold `0.711`.
6. reload the final artifacts and require exact model, head, parameter, and prediction hashes.

model construction and evaluation used nvidia h200 gpu nodes. encoder feature extraction used one gpu, while the 1,025-parameter logistic fit ran on cpu. the full command and export detail is in the [training recipe](training.md).

## about the dataset

the benchmark and punctuation-head recipe use the `ne-commercial-v1` nepali profile. it contains 402,905 accepted clips and 630.2761 hours, with 393,002 clips and 608.9182 hours in train. the upstream asr weights are not retrained by this release. the speech files and row manifests are not redistributed here.

| source | clips | hours | terms |
|---|---:|---:|---|
| openslr 54 | 153,694 | 150.6178 | cc by-sa 4.0 |
| openslr 43 | 2,064 | 2.7960 | cc by-sa 4.0 |
| fleurs nepali | 4,351 | 14.3423 | cc by 4.0 |
| indicvoices nepali | 242,796 | 462.5200 | cc by 4.0, gated |

audio is normalized to mono 16 khz, text uses unicode nfc and conservative space cleanup, and speaker or recording groups stay within one split. obtain every source under its original terms. read the [data note](data.md).

## how the benchmark was run

1. freeze the ordered 3,630-row view and verify sha-256 `2374cac54831ce9c69282503763d7f1e12ada0404ae34ed471a7538cdae6c61f`.
2. require exactly 304 fleurs, 2,569 indicvoices, and 757 openslr 54 rows.
3. pin every model revision, decoder, language value, and batch rule before loading weights.
4. load each model from scratch twice and decode every row with no batch fallback.
5. accept a model only when both full prediction files have the same sha-256.
6. compute pi wer, pi cer, raw wer, and raw cer, then rank by the exact primary error ratio.

the public evaluator enforces the kriti view hash, row count, and source counts. this command also fixes the benchmark batch size at 32:

```bash
python scripts/evaluate.py \
  --view dev.jsonl \
  --output-dir run-1 \
  --revision 762d1c17edaff0a548f3483e37e491fe8cc77971 \
  --batch-size 32
```

run it again into a new output directory and compare `predictions.jsonl` hashes. the view contains gated data and is not redistributed. this is a development view used during model selection, not an untouched test result or a universal nepali claim.

## use kriti

```bash
git clone https://github.com/harrrshall/kriti
cd kriti
python -m venv .venv
source .venv/bin/activate
pip install -e '.[runtime]'
kriti audio.wav
```

```python
from kriti import load_model

model = load_model()
print(model.transcribe("audio.wav"))
```

the loader fetches the public artifacts, verifies their hashes, creates the 119m live graph, and runs nepali rnnt decoding with terminal danda recovery. kriti is released under mit with upstream ai4bharat attribution in the repository notice.
