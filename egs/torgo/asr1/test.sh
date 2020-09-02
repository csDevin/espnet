#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
score_sclite.sh --bpe 5000 --bpemodel data/lang_char/trainset_unigram5000.model --wer true \
exp/trainset_pytorch_train_specaug/decode_test_clean_model.acc.best_decode_lm data/lang_char/trainset_unigram5000_units.txt