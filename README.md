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
* Run `make` in ./EVALB. 

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

To parse sentences, first place the sentences in the [input file](example_sentences.txt) and download a [pre-trained model](#Pre-trained-models) (or train it yourself), then run following command **to parse with a print of the head contributions**. A warning that the following **will be slow:**
```
sh parse.sh
```

**For a quicker parsing, use the following command.** It does not compute head contributions:
```
sh parse_quick.sh
```

Words should be POS-tagged before use. If your input is not tagged, it will be tagged for you by the program. If you prefer to use your own tagging, provide input as ``tag1_word1 tag2_word2`` and set the flag ``--pos-tag 0`` in the command line arguments.

The output files of the command ``sh parse.sh`` are the following:

* The file in the path in the ``--output-path-syndep`` argument contains the indices of the dependency head of each word in the sequence.
* The file in the path in the ``--output-path-synlabel`` argument contains the dependency labels.
* The file in the path in the ``--output-path-synconst`` argument contains the linearized constituency tree.

Example running the inference code in a Python virtual environment:

```
mkdir neural-parser
cd neural-parser
virtualenv -p python3.6 ./pyenv/neural-parser
source ./pyenv/neural-parser/bin/activate
git clone https://github.com/KhalilMrini/LAL-Parser
cd LAL-Parser/
alias pip=pip3; source requirements.sh

# Testing the Neural Adobe-UCSD Parser inference
sh parse.sh
```

Example running the inference code in a Docker container:

```
docker run --interactive --tty ubuntu:18.04 bash
apt update; apt install -y git nano wget htop python3 python3-pip unzip; git clone https://github.com/KhalilMrini/LAL-Parser
cd LAL-Parser/
alias pip=pip3; source requirements.sh
apt-get install -y libhdf5-serial-dev=1.8.16+docs-4ubuntu1.1

# Testing the Neural Adobe-UCSD Parser inference
alias python=python3 
source parse.sh
```

To generate a diagram of a parse tree based on the output `output_synconst_0.txt`, see [this list of tools to draw parse trees](https://stackoverflow.com/q/4972571/395857).

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
