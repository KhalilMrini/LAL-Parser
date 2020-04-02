#!/usr/bin/env bash
python src_joint/main.py test \
--dataset ptb \
--consttest-ptb-path data/23.auto.clean \
--deptest-ptb-path data/ptb_test_3.3.0.sd \
--embedding-path data/glove.gz \
--model-path-base best_parser.pt
