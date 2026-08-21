.PHONY: setup hooks format lint check data benchmark routing-baseline demo test test-experiments

UV_RUN := uv run --locked
RUN := $(UV_RUN) morgott

setup:
	uv sync --locked

hooks: setup
	$(UV_RUN) pre-commit install

format:
	$(UV_RUN) ruff check --fix src tests scripts examples
	$(UV_RUN) ruff format src tests scripts examples

lint:
	$(UV_RUN) ruff format --check src tests scripts examples
	$(UV_RUN) ruff check src tests scripts examples

check: lint test test-experiments

data:
	$(RUN) data

benchmark:
	$(RUN) benchmark

routing-baseline:
	$(RUN) routing-baseline

demo:
	$(RUN) demo

test:
	$(UV_RUN) --extra azure python -m unittest discover -s tests -v

# One process per module: several experiment tests import a sibling `run`
# module by bare name, so they cannot share an interpreter.
test-experiments:
	@set -e; files=$$(find experiments -name 'test_*.py' | sort); \
	test -n "$$files"; \
	for f in $$files; do \
		m=$$(echo "$$f" | sed 's/\.py$$//; s|/|.|g'); \
		echo "$$m"; \
		$(UV_RUN) --extra cascade --extra encoder python -m unittest "$$m"; \
	done
