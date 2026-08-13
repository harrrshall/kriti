# data

kriti publishes data provenance and preparation logic, not speech recordings,
transcripts, speaker metadata, or row manifests. users must obtain each source
under its original terms.

| source | pinned version | accepted clips | accepted hours | license or access |
|---|---|---:|---:|---|
| openslr 54 | slr54 | 153,694 | 150.6178 | cc by-sa 4.0 |
| openslr 43 | slr43 | 2,064 | 2.7960 | cc by-sa 4.0 |
| fleurs `ne_np` | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | 4,351 | 14.3423 | cc by 4.0 |
| indicvoices nepali | `c96f9088f138cf89d419da7e8e643e1f05c00a87` | 242,796 | 462.5200 | cc by 4.0, gated |

the aggregate profile has 402,905 clips and 630.2761 hours. preparation uses
16 khz mono flac, unicode nfc, conservative whitespace cleanup, duration and
signal checks, exact decoded-audio hashes, and deterministic grouping so a
speaker or recording cannot cross train, development, and test splits.

fleurs official evaluation material remains evaluation-only. indicvoices
requires access approval. source licenses, attribution, share-alike terms, and
gated conditions continue to apply. this repository does not grant rights to
the source recordings.
