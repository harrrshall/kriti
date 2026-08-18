<div align="center">

# kriti 🎙️

### a compact acoustic punctuation architecture for nepali asr

kriti couples a nepali-only rnnt transcription graph with a 1,025-parameter acoustic terminal-danda head. the released live graph contains exactly 119,462,146 parameters.

[model weights](https://huggingface.co/harrrshall/kriti) · [architecture](architecture.md) · [training](training.md) · [benchmark](benchmark.md) · [data](data.md) · [usage](usage.md) · [references](references.md)

</div>

## the contribution

kriti turns a multilingual hybrid checkpoint into a focused nepali inference graph and adds an acoustic punctuation branch that shares the speech encoder. the architecture has four released contributions:

1. a nepali-only rnnt path with the ctc branch and 21 additional language heads removed
2. a runtime-compacted prediction embedding with 257 nepali output rows
3. a 1,025-parameter acoustic head for terminal devanagari danda recovery
4. content-bound loading that verifies artifact hashes, graph shape, parameter count, and prediction identity

the pretrained conformer encoder and nepali rnnt weights come from the mit-licensed [ai4bharat nepali indicconformer](https://huggingface.co/ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large) at revision `cd09ba7720f3b17d259f6bfd03e1463bc5ba517d`. kriti retains those upstream weights, applies deterministic deployment pruning, and trains the acoustic danda head. the full technical lineage is recorded in [references.md](references.md) and [notice](NOTICE).

## architecture

```text
16 khz audio → log-mel → conformer encoder → nepali rnnt → transcript
                                  │
                                  └→ pooled encoder states → danda head → final text
```

the transcription graph contains 119,461,121 live parameters. the punctuation branch pools the same encoder states into 1,024 mean and standard-deviation features, then applies one logistic layer and a frozen threshold of `0.711`. the combined graph reaches 119,462,146 live parameters and uses one acoustic network.

the full layer dimensions, pruning contract, artifact hashes, and loader invariants are in [architecture.md](architecture.md).

## measured result

the current evidence is a frozen 3,630-utterance development snapshot containing 19 fully evaluated open systems. every included system completed two fresh-load replicates with matching prediction hashes.

| rank | system | pi wer | pi cer | raw wer |
|---:|---|---:|---:|---:|
| 1 | **kriti** | **24.0773%** | **8.2877%** | **24.6854%** |
| 1 | ai4bharat nepali indicconformer, rnnt | 24.0773% | 8.2877% | 25.1928% |
| 3 | ai4bharat nepali indicconformer, ctc | 25.3109% | 8.4515% | 26.4313% |
| 4 | qwen3 asr nepali, fine-tuned | 52.4043% | 24.2176% | 55.5196% |

pi wer means punctuation-insensitive word error rate. exact primary error ratios share a competition rank, with raw wer reported as a separate descriptive measure. kriti shares exact rank 1 with the official ai4bharat nepali rnnt and records the lowest raw wer within the evaluated snapshot.

the snapshot uses a development view that also guided kriti model selection. the planned campaign ended after 19 complete systems, and the published table contains results only for that completed field. stronger claims require a separately frozen evaluation covering real microphones, streaming behavior, domain shift, latency, and product punctuation.

read [benchmark.md](benchmark.md) for the protocol and scope. [benchmark.json](benchmark.json) carries all 19 systems, immutable revisions, decoder settings, exact metrics, and replicate hashes.

## data record

the punctuation-head recipe and benchmark use the `ne-commercial-v1` profile: 402,905 accepted clips and 630.2761 decoded hours, including 393,002 training clips and 608.9182 training hours.

| source | accepted clips | accepted hours | terms |
|---|---:|---:|---|
| [openslr 54](https://www.openslr.org/54/) | 153,694 | 150.6178 | cc by-sa 4.0 |
| [openslr 43](https://www.openslr.org/43/) | 2,064 | 2.7960 | cc by-sa 4.0 |
| [fleurs nepali](https://huggingface.co/datasets/google/fleurs) | 4,351 | 14.3423 | cc by 4.0 |
| [indicvoices nepali](https://huggingface.co/datasets/ai4bharat/IndicVoices) | 242,796 | 462.5200 | cc by 4.0 and gated access |

each source retains its license, access terms, and attribution requirements. audio, transcripts, speaker metadata, and row manifests stay with their licensed data environment. [data.md](data.md) documents provenance, preparation, split policy, privacy, and redistribution boundaries.

## quick start

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

the loader fetches the public artifacts, verifies both sha-256 values, builds the compact nepali graph, checks the exact live parameter count, and runs rnnt decoding with terminal danda recovery. additional examples are in [usage.md](usage.md).

## reproduce the research record

the public repository keeps each claim close to its evidence:

* [architecture.md](architecture.md) defines the live graph and its invariants
* [training.md](training.md) records deterministic pruning and punctuation-head fitting
* [data.md](data.md) records source provenance, licenses, and preparation policy
* [benchmark.md](benchmark.md) defines ranking, replication, and evidence scope
* [benchmark.json](benchmark.json) provides the machine-readable 19-system snapshot
* [model.json](model.json) binds the released graph to artifact and prediction hashes
* [references.md](references.md) credits upstream models, software, datasets, and comparison systems

## license and credit

kriti source code is released under mit. the released weights preserve the ai4bharat indicconformer lineage and its mit terms. datasets and benchmark checkpoints remain governed by their respective authors, licenses, model cards, and access conditions. [notice](NOTICE) and [references.md](references.md) provide the attribution record.
