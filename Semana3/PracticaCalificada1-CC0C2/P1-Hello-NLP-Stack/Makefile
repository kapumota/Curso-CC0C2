# Makefile (datos + pipeline)
PY ?= python3
RAW ?= data/raw
PROC ?= data/processed
EXT ?= data/external
EVID ?= evidencias
OUT ?= out
REPORT ?= docs

N ?= 5000
N_WIKI ?= 20000
N_TWEETS ?= 8000

.PHONY: help data preprocess train eval report checks checksums verify-data dirs clean-data
help:
	@echo "Targets: data, preprocess, train, eval, report, checks"
dirs:
	mkdir -p $(RAW) $(PROC) $(EXT) $(EVID) $(OUT) $(REPORT)
data: dirs
	$(PY) scripts/get_data.py --hf-id iberbench/iberbench_all --subset tass-tass-sentiment_analysis-2020-spanish-spain --split train --out $(RAW)/p1_tass_es_spain_train.csv --n $(N) || \
	$(PY) scripts/get_data.py --hf-id mteb/tweet_sentiment_multilingual --split train --out $(RAW)/p1_tweets_es_train.csv --n $(N_TWEETS)
checks: checksums verify-data
checksums: dirs
	$(PY) scripts/checksums.py --root data --out $(EVID)/data_sha256.txt
verify-data:
	$(PY) scripts/checksums.py --root data --out $(EVID)/data_sha256_verify.txt
	@diff -u $(EVID)/data_sha256.txt $(EVID)/data_sha256_verify.txt || true

preprocess: dirs
	$(PY) src/preprocess.py --in $(RAW) --out $(PROC)
train: preprocess
	$(PY) src/train.py --data $(PROC) --out $(OUT)
eval: train
	$(PY) src/eval.py --data $(PROC) --model $(OUT) --out $(OUT)
report: eval
	$(PY) src/make_report.py --metrics $(OUT) --figs $(OUT) --out $(REPORT)

clean-data:
	rm -rf $(RAW)/* $(PROC)/*
