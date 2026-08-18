# references and credit

kriti is a composed research system with an original deployment architecture built from credited upstream work. this record identifies the role of each external model, runtime, dataset, method, and comparison artifact.

## model lineage

* [ai4bharat nepali indicconformer hybrid](https://huggingface.co/ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large), revision `cd09ba7720f3b17d259f6bfd03e1463bc5ba517d`, supplies the pretrained conformer encoder, prediction network, and nepali rnnt weights. the upstream model card declares mit licensing.
* [ai4bharat nemo](https://github.com/AI4Bharat/NeMo/tree/8dce88cf8e94963e2033c3137f7b9993b51db88a), revision `8dce88cf8e94963e2033c3137f7b9993b51db88a`, supplies the runtime used to restore and execute the upstream checkpoint.
* [conformer](https://arxiv.org/abs/2005.08100) by gulati et al. supplies the convolution-augmented transformer encoder design used by the upstream model.
* [sequence transduction with recurrent neural networks](https://arxiv.org/abs/1211.3711) by graves supplies the rnnt formulation used by the upstream model.
* [sentencepiece](https://arxiv.org/abs/1808.06226) by kudo and richardson supplies the subword tokenization method used by the upstream model.

kriti retains the pretrained ai4bharat encoder and nepali rnnt weights. our released work adds deterministic language-specific pruning, runtime embedding compaction, the shared-encoder danda branch, artifact verification, and reproducible packaging. the public graph follows this pruning lineage. the earlier qwen3 asr campaign is preserved as a separate historical baseline in the private research record.

## software used in the released workflow

* [pytorch](https://pytorch.org/) executes the neural graph.
* [nvidia nemo](https://github.com/NVIDIA/NeMo) provides the broader toolkit lineage inherited by the ai4bharat fork.
* [scikit-learn](https://scikit-learn.org/) supplies the balanced logistic regression implementation used for the 1,025-parameter danda head.
* [hugging face hub](https://huggingface.co/) hosts immutable model revisions and the public kriti artifacts.
* [jiwer](https://github.com/jitsi/jiwer) supports wer and cer computation in the public evaluator.

## training and evaluation data

* [openslr 54](https://www.openslr.org/54/) supplies diverse nepali read speech under cc by-sa 4.0.
* [openslr 43](https://www.openslr.org/43/) supplies a small clean nepali read-speech supplement under cc by-sa 4.0.
* [google fleurs](https://huggingface.co/datasets/google/fleurs), configuration `ne_np`, supplies multilingual benchmark speech under cc by 4.0.
* [ai4bharat indicvoices](https://huggingface.co/datasets/ai4bharat/IndicVoices), nepali partition, supplies read, extempore, and conversational speech under cc by 4.0 with gated access.

the source recordings, transcripts, speaker metadata, and row manifests remain under their upstream terms and stay outside the public kriti repositories. [data.md](data.md) records exact accepted counts, preparation policy, and redistribution boundaries.

## benchmark systems

the 19-system snapshot credits every compared checkpoint through its immutable repository identifier in [benchmark.json](benchmark.json). the evaluated field includes work published by:

* [ai4bharat](https://huggingface.co/ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large)
* [sidskarki](https://huggingface.co/sidskarki/Qwen3-ASR-Nepali)
* [kiranpantha](https://huggingface.co/kiranpantha)
* [dragneel](https://huggingface.co/Dragneel/whisper-large-v3-nepali-openslr)
* [suman paudel and collaborators](https://huggingface.co/sumanpaudel1997)
* [meta](https://huggingface.co/facebook)
* [shniranjan](https://huggingface.co/shniranjan/wav2vec2-large-xlsr-300m-nepali)
* [spktsagar](https://huggingface.co/spktsagar/wav2vec2-large-xls-r-300m-nepali-openslr)
* [openai](https://huggingface.co/openai)
* [vakyansh and harveen chadha](https://huggingface.co/Harveenchadha/vakyansh-wav2vec2-nepali-nem-130)
* [qwen](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)

each benchmark result belongs to its named checkpoint, authors, and maintainers. inclusion provides a reproduced comparison on the frozen kriti development view and carries no transfer of authorship or ownership.

## artifact and claim record

* [architecture.md](architecture.md) documents the graph derived from the credited base model.
* [training.md](training.md) separates retained upstream weights from kriti-trained parameters.
* [benchmark.md](benchmark.md) states the selection-view protocol and evidence boundary.
* [model.json](model.json) binds the release to exact artifact and prediction hashes.
* [notice](NOTICE) preserves the required upstream attribution beside the source license.
