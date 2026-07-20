.PHONY: setup data benchmark demo test poc

PYTHON := python3
RUN := PYTHONPATH=src $(PYTHON) -m vulsight_guard.cli

setup:
	$(PYTHON) -m pip install -e .

data:
	$(RUN) data

benchmark:
	$(RUN) benchmark

demo:
	$(RUN) demo

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

poc: data benchmark demo test
