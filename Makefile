SHELL := /bin/bash
ROOT := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

.PHONY: nothing all clean test
nothing:

help:
	@echo "no help info"

all:
	echo hello world

clean:
	rm -fvr \#* *~ *.exe out
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete
	find . -name ".pytest_cache" -type d -exec rm -rf {} +
	find . -name "*.egg-info" -type d -exec rm -rf {} +
	find . -name "build" -type d -exec rm -rf {} +
	find . -name "dist" -type d -exec rm -rf {} +

test:
	python -m pytest tests/