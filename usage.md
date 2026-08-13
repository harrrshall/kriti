# usage

kriti requires python 3.10 or newer, ffmpeg, pytorch, and the pinned ai4bharat
nemo fork.

```bash
git clone https://github.com/harrrshall/kriti
cd kriti
python -m venv .venv
source .venv/bin/activate
pip install -e '.[runtime]'
```

provide mono 16 khz audio:

```bash
kriti audio.wav
```

or use python:

```python
from kriti import load_model

model = load_model()
print(model.transcribe("audio.wav"))
```

the loader downloads `kriti.nemo` and `punctuation_head.json` from
`harrrshall/kriti`, verifies both sha-256 values, compacts the prediction
embedding to 257 rows, verifies the exact live parameter count, and performs
nepali rnnt decoding plus terminal danda restoration.
