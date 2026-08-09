PYTHON ?= python3
TECTONIC ?= tectonic

# Human-facing canonical locations. papers/current points to papers/v4.
PAPER ?= papers/current
RELEASE_RESULTS ?= results/release
LOCAL_RESULTS ?= results/local
LOCAL_PAPER ?= tmp/reproduction-paper
CALIBRATION ?= $(RELEASE_RESULTS)/uci_calibration
EXTERNAL_ARCHIVE ?= data_cache/RobustKnapsack.zip

.PHONY: install test check verify evidence exact-audit reproduce paper package \
	anonymous clean-preview clean v4-verify v4-evidence v4-exact-audit \
	v4-reproduce v4-paper v4-package v4-anonymous-package

install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ".[experiments,validation,dev]"

test:
	$(PYTHON) -m pytest -q

check: verify

verify: test
	MPLCONFIGDIR="$$HOME/.cache/matplotlib" $(PYTHON) scripts/verify_v4_release.py \
		--results $(RELEASE_RESULTS) \
		--paper $(PAPER)

evidence:
	MPLCONFIGDIR="$$HOME/.cache/matplotlib" $(PYTHON) research/generate_v4_publication_artifacts.py \
		--results $(RELEASE_RESULTS) \
		--paper $(PAPER)

exact-audit:
	env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
		VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
		$(PYTHON) research/exact_integration_campaign.py \
		--output-dir $(RELEASE_RESULTS)/exact_integration \
		--sizes 30 --seeds 0,1,2 --repetitions 2 --time-limit 5

reproduce: test
	env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
	$(PYTHON) scripts/run_v4_publication_campaign.py \
		--output-dir $(LOCAL_RESULTS) \
		--calibration-dir $(CALIBRATION) \
		--external-archive $(EXTERNAL_ARCHIVE)
	env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
		$(PYTHON) research/exact_integration_campaign.py \
		--output-dir $(LOCAL_RESULTS)/exact_integration \
		--sizes 30 --seeds 0,1,2 --repetitions 2 --time-limit 5
	MPLCONFIGDIR="$$HOME/.cache/matplotlib" $(PYTHON) research/generate_v4_publication_artifacts.py \
		--results $(LOCAL_RESULTS) \
		--paper $(LOCAL_PAPER)

paper:
	cd $(PAPER) && $(TECTONIC) main.tex
	cd $(PAPER) && $(TECTONIC) journal.tex
	cd $(PAPER) && $(TECTONIC) journal-blind.tex
	cd $(PAPER) && $(TECTONIC) companion.tex
	cd $(PAPER) && $(TECTONIC) companion-blind.tex
	cd $(PAPER) && $(TECTONIC) executive-summary.tex

package: paper
	mkdir -p $(PAPER)/pdf
	cp $(PAPER)/main.pdf $(PAPER)/pdf/full.pdf
	cp $(PAPER)/journal.pdf $(PAPER)/pdf/paper.pdf
	cp $(PAPER)/journal-blind.pdf $(PAPER)/pdf/paper-blind.pdf
	cp $(PAPER)/companion.pdf $(PAPER)/pdf/companion.pdf
	cp $(PAPER)/companion-blind.pdf $(PAPER)/pdf/companion-blind.pdf
	cp $(PAPER)/executive-summary.pdf $(PAPER)/pdf/executive-summary.pdf

anonymous: verify paper
	$(PYTHON) scripts/build_v4_anonymous_supplement.py \
		--results $(RELEASE_RESULTS) \
		--paper $(PAPER) \
		--output $(PAPER)/anonymous-supplement.zip

clean-preview:
	$(PYTHON) scripts/clean_workspace.py

clean:
	$(PYTHON) scripts/clean_workspace.py --apply

# Backward-compatible aliases for existing automation.
v4-verify: verify
v4-evidence: evidence
v4-exact-audit: exact-audit
v4-reproduce: reproduce
v4-paper: paper
v4-package: package
v4-anonymous-package: anonymous
