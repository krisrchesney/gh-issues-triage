.PHONY: install test lint demo train drift serve docker

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

demo:
	python -m ghtriage.cli init-demo --rows 5000

train:
	python -m ghtriage.cli train

drift:
	python -m ghtriage.cli drift

serve:
	python -m ghtriage.cli serve

docker:
	docker build -t ghtriage .
