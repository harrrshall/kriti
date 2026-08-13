---
language:
- ne
license: mit
library_name: nemo
pipeline_tag: automatic-speech-recognition
tags:
- nepali
- rnnt
- conformer
- speech
base_model: ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large
---

# kriti

kriti sets a new efficiency frontier for open nepali asr with 119,462,146 live
parameters. it is joint first in our 19-system development benchmark, records
the best raw wer in that evaluated field, and leads the fine-tuned qwen3 nepali
asr model, whisper large v3,
and mms 1b variants while being far smaller. it is derived from the
mit-licensed ai4bharat nepali indicconformer and retains its attribution.

## results

| rank | model | punctuation-insensitive wer | raw wer |
|---:|---|---:|---:|
| 1 | <mark>kriti</mark> | <mark>24.0773%</mark> | <mark>24.6854%</mark> |
| 1 | ai4bharat nepali indicconformer rnnt | 24.0773% | 25.1928% |
| 3 | ai4bharat nepali indicconformer ctc | 25.3109% | 26.4313% |
| 4 | qwen3 asr nepali fine-tuned | 52.4043% | 55.5196% |
| 5 | whisper large v3 nepali | 55.7059% | 57.8369% |
| 6 | whisper large v3 nepali openslr | 55.7678% | 57.4514% |
| 7 | whisper medium nepali | 58.4027% | 60.8634% |
| 8 | meta mms 1b all, nepali adapter | 60.1274% | 61.2749% |
| 9 | mms 1b nepali | 60.7661% | 63.2307% |
| 10 | xls-r 300m nepali | 62.8900% | 63.9039% |

the benchmark uses a frozen 3,630-utterance development view. each listed
system was loaded and evaluated twice with identical prediction hashes. kriti
is joint first on the primary punctuation-insensitive metric and has the lower
raw wer through its acoustic danda head. this is model-selection evidence, not
an untouched test claim.

## architecture

```text
16 khz audio -> 80-bin log-mel -> 17 conformer blocks -> 257-token nepali rnnt
             -> 1,024 pooled encoder features -> 1,025-parameter danda head
```

the conformer has width 512, eight attention heads, four-times subsampling, and
kernel width 31. the prediction and joint networks have width 640. the public
loader retains only the nepali joint head and compacts the prediction embedding
at load time.

## usage

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

input must be mono 16 khz audio. artifact hashes and the exact live parameter
count are verified by the loader. see the
[github repository](https://github.com/harrrshall/kriti) for architecture,
training, data provenance, benchmark protocol, and metric code.

## license and data

code and derived weights are released under mit with ai4bharat attribution.
the training sources retain their original terms. no raw audio, transcripts,
speaker metadata, or gated source rows are redistributed.
