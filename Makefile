PYTHON ?= python3

.PHONY: verify evaluate

verify:
	$(PYTHON) -m evaluation_v1_2 verify

evaluate:
	$(PYTHON) -m evaluation_v1_2 evaluate
