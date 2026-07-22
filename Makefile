.PHONY: setup hooks format lint check data benchmark demo test

UV_RUN := uv run --locked
RUN := $(UV_RUN) morgott

setup:
	uv sync --locked

hooks: setup
	$(UV_RUN) pre-commit install

format:
	$(UV_RUN) ruff check --fix src tests
	$(UV_RUN) ruff format src tests

lint:
	$(UV_RUN) ruff format --check src tests
	$(UV_RUN) ruff check src tests

check: lint test

data:
	$(RUN) data

benchmark:
	$(RUN) benchmark

demo:
	$(RUN) demo

test:
	$(UV_RUN) python -m unittest discover -s tests -v
