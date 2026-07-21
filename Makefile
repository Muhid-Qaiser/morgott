.PHONY: setup data benchmark demo test

RUN := uv run morgott

setup:
	uv sync

data:
	$(RUN) data

benchmark:
	$(RUN) benchmark

demo:
	$(RUN) demo

test:
	uv run python -m unittest discover -s tests -v
