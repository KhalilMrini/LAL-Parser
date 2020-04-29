#!/usr/bin/env bash
python src_joint/main.py parse \
--dataset ptb \
--save-per-sentences 1000 \
--eval-batch-size 50 \
--contributions 0 \
--input-path example_sentences.txt \
--output-path-synconst output_synconst \
--output-path-syndep output_syndephead \
--output-path-synlabel output_syndeplabel \
--embedding-path data/glove.gz \
--model-path-base best_parser.pt
