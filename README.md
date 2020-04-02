# Neural Adobe-UCSD Parser

This is a PyTorch implementation of the parser described in ["Rethinking Self-Attention: An Interpretable Self-Attentive Encoder-Decoder Parser" (Mrini et al., 2019)](https://arxiv.org/abs/1911.03875).

## Contents
1. [Requirements](#Requirements)
2. [Pre-trained models](#Pre-trained-models)
3. [Training](#Training)
4. [Inference](#Inference)
5. [Label Attention](#Label-Attention)
6. [Citation](#Citation)
7. [Credits](#Credits)

## Requirements

* Python 3.6 or higher.
* The Python package requirements can be installed through the `requirements.sh` file.

## Pre-trained models

Our best model is available for download [here](https://drive.google.com/file/d/1LC5iVcvgksQhNVJ-CbMigqXnPAaquiA2/view?usp=sharing). It uses XLNet embeddings, HSPG tree representation has 3 layers of self-attention and 1 final 128-dimensional Label Attention Layer with a position-wise feed-forward layer and no residual dropout. On the English Penn Treebank benchmark dataset, our best parser reaches 96.38 F1 score for Constituency Parsing, and 97.42 UAS and 96.26 LAS for Dependency Parsing.

Pre-trained BERT and XLNet weights will be automatically downloaded as needed by the `pytorch-transformers` package.

## Training

The English PTB data files for Dependency Parsing and Constituency Parsing are in the `data/` folder. Note that we provide data with predicted Part-of-Speech tags. We used predicted PoS tags in training. If needed, the gold tags should be obtained separately.

We provide the training script in `best_parser_training_script.sh`.

Check the function `make_hparams` in `src_joint/main.py` for hyperparameters specific to the Label Attention Layer. For more training and evaluation instructions, see [the HPSG parser repository](https://github.com/DoodleJZ/HPSG-Neural-Parser).

As an example, after extracting the pre-trained model, you can evaluate it on the test set using the following command:

```
sh test.sh
```

## Inference

To parse sentences, first place the sentences in the [input file](example_sentences.txt) and download a [pre-trained model](#Pre-trained-models) (or train it yourself), then run following command:
```
sh parse.sh
```

Words should be POS-tagged before use. If your input is not tagged, it will be tagged for you by the program. If you prefer to use your own tagging, provide input as ``tag1_word1 tag2_word2`` and set the flag ``--pos-tag 0`` in the command line arguments.

The output files of the command ``sh parse.sh`` are the following:

* The file in the path in the ``--output-path-syndep`` argument contains the indices of the dependency head of each word in the sequence.
* The file in the path in the ``--output-path-synlabel`` argument contains the dependency labels.
* The file in the path in the ``--output-path-synconst`` argument contains the linearized constituency tree.

## Label Attention

The implementation for the Label Attention Layer is in the Python Class `LabelAttention` in the file [`src_joint/KM_parser.py`](src_joint/KM_parser.py).

## Citation

If you use the Neural Adobe-UCSD Parser, please cite our [paper](https://arxiv.org/abs/1911.03875) as follows:
```
@article{mrini2019rethinking,
  title={Rethinking Self-Attention: An Interpretable Self-Attentive Encoder-Decoder Parser},
  author={Mrini, Khalil and Dernoncourt, Franck and Bui, Trung and Chang, Walter and Nakashole, Ndapa},
  journal={arXiv preprint arXiv:1911.03875},
  year={2019}
}
```

## Credits

The code in this repository, the dataset and portions of this README are based on [the Self-Attentive Parser](https://github.com/nikitakit/self-attentive-parser) and [the HPSG Parser](https://github.com/DoodleJZ/HPSG-Neural-Parser).
