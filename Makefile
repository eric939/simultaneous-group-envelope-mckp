PYTHON ?= python3
RELEASE_DIR ?= results/release/2026-08-09-paper-b-final-r5
LOCAL_DIR ?= results/local/paper-b-reproduction
CALIBRATION_DIR ?= $(RELEASE_DIR)/uci_calibration
EXTERNAL_ARCHIVE ?= data_cache/RobustKnapsack.zip

.PHONY: install test verify reproduce exact-audit clean

install:
	uv sync --frozen --extra experiments --extra validation --extra dev

test:
	$(PYTHON) -m pytest -q

verify: test
	$(PYTHON) scripts/verify_release.py --release-dir $(RELEASE_DIR)

reproduce: test
	env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
		VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
		$(PYTHON) scripts/run_v4_publication_campaign.py \
		--output-dir $(LOCAL_DIR) \
		--calibration-dir $(CALIBRATION_DIR) \
		--external-archive $(EXTERNAL_ARCHIVE)
	$(MAKE) exact-audit PYTHON=$(PYTHON) LOCAL_DIR=$(LOCAL_DIR)

exact-audit:
	env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
		VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
		$(PYTHON) research/exact_integration_campaign.py \
		--output-dir $(LOCAL_DIR)/exact_integration \
		--sizes 30 --seeds 0,1,2 --repetitions 2 --time-limit 5

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; p=Path('results/local'); shutil.rmtree(p) if p.exists() else None"
