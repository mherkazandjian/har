SHELL := /bin/bash
ROOT := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

RUST_DIR := $(ROOT)/src/rust
RUST_SANDBOX := $(RUST_DIR)/rust_sandbox
RUST_STATIC_SANDBOX := $(RUST_DIR)/rust_static_sandbox
RUST_BIN := $(RUST_DIR)/har-rust
RUST_BIN_STATIC := $(RUST_DIR)/har-rust-static

# ---------------------------------------------------------------------------
# Container backend: apptainer (default) | docker | native
# Usage: make build-rust-release container=docker
# ---------------------------------------------------------------------------
container ?= apptainer

DOCKER_IMAGE := har-rust-build
DOCKER_IMAGE_STATIC := har-rust-build-static

ifeq ($(container),apptainer)
  DYNAMIC_TARGET := /tmp/har_target
  STATIC_TARGET  := /tmp/har_static_target
  RUN_DYNAMIC := apptainer exec \
      --bind $(RUST_DIR):/work \
      --bind /tmp/har_cargo_home:/tmp/cargo_home \
      --bind $(DYNAMIC_TARGET):/tmp/target_dir \
      --env CARGO_HOME=/tmp/cargo_home \
      --env CARGO_TARGET_DIR=/tmp/target_dir \
      --env HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial \
      $(RUST_SANDBOX) bash -c
  RUN_STATIC := apptainer exec \
      --bind $(RUST_DIR):/work \
      --bind /tmp/har_static_cargo:/tmp/cargo_home \
      --bind $(STATIC_TARGET):/tmp/target_dir \
      --env CARGO_HOME=/tmp/cargo_home \
      --env CARGO_TARGET_DIR=/tmp/target_dir \
      --env RUSTUP_HOME=/work/.rustup_home \
      $(RUST_STATIC_SANDBOX) bash -c
  WORKDIR := /work
  NEED_DYNAMIC_ENV := rust-sandbox-apptainer
  NEED_STATIC_ENV := rust-static-sandbox-apptainer
  ENSURE_DYNAMIC_DIRS := @mkdir -p /tmp/har_cargo_home $(DYNAMIC_TARGET)
  ENSURE_STATIC_DIRS  := @mkdir -p /tmp/har_static_cargo $(STATIC_TARGET)
else ifeq ($(container),docker)
  DYNAMIC_TARGET := $(RUST_DIR)/target
  STATIC_TARGET  := $(RUST_DIR)/target
  RUN_DYNAMIC := docker run --rm \
      -v $(RUST_DIR):/work \
      -e CARGO_HOME=/work/.cargo_home \
      -e HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial \
      $(DOCKER_IMAGE) bash -c
  RUN_STATIC := docker run --rm \
      -v $(RUST_DIR):/work \
      -e CARGO_HOME=/work/.cargo_home \
      $(DOCKER_IMAGE_STATIC) bash -c
  WORKDIR := /work
  NEED_DYNAMIC_ENV := rust-sandbox-docker
  NEED_STATIC_ENV := rust-static-sandbox-docker
  ENSURE_DYNAMIC_DIRS :=
  ENSURE_STATIC_DIRS  :=
else ifeq ($(container),native)
  DYNAMIC_TARGET := $(RUST_DIR)/target
  STATIC_TARGET  := $(RUST_DIR)/target
  RUN_DYNAMIC := bash -c
  RUN_STATIC := bash -c
  WORKDIR := $(RUST_DIR)
  NEED_DYNAMIC_ENV :=
  NEED_STATIC_ENV :=
  ENSURE_DYNAMIC_DIRS :=
  ENSURE_STATIC_DIRS  :=
else
  $(error container= must be one of: apptainer, docker, native)
endif

.PHONY: nothing all clean test test-python test-bagit test-mpi test-rust help \
	rust-sandbox rust-static-sandbox \
	rust-sandbox-apptainer rust-sandbox-docker rust-sandbox-native \
	rust-static-sandbox-apptainer rust-static-sandbox-docker rust-static-sandbox-native \
	build-rust build-rust-debug build-rust-release \
	build-rust-static build-rust-release-static \
	clean-rust install-rust lint lint-rust lint-python

nothing:

help:
	@echo "Container backend (default: apptainer):"
	@echo "  container=apptainer  - build inside Apptainer sandbox"
	@echo "  container=docker     - build inside Docker container"
	@echo "  container=native     - build directly on host (no container)"
	@echo ""
	@echo "Rust build targets:"
	@echo "  build-rust                - alias for build-rust-release"
	@echo "  build-rust-debug          - debug build (dynamic linking)"
	@echo "  build-rust-release        - release build (dynamic linking)"
	@echo "  build-rust-static         - debug build (static musl)"
	@echo "  build-rust-release-static - release build (static musl)"
	@echo ""
	@echo "Rust environment setup:"
	@echo "  rust-sandbox              - create dynamic-build environment"
	@echo "  rust-static-sandbox       - create static/musl-build environment"
	@echo ""
	@echo "Rust utility targets:"
	@echo "  test-rust                 - run Rust tests"
	@echo "  clean-rust                - remove target/ and cargo/rustup homes"
	@echo "  install-rust              - copy release binaries to DESTDIR"
	@echo ""
	@echo "General targets:"
	@echo "  test                      - run all tests"
	@echo "  test-python               - run Python tests"
	@echo "  clean                     - remove Python build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make build-rust-release                        # apptainer (default)"
	@echo "  make build-rust-release container=docker       # docker"
	@echo "  make build-rust-release container=native       # no container"

all: build-rust-release build-rust-release-static

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

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

test: test-python test-rust

test-python:
	python -m pytest tests/test_h5.py tests/test_bagit.py

test-bagit:
	python -m pytest tests/test_bagit.py -v

test-mpi:
	module load h5py/3.14.0-foss-2025a && \
	mpirun -np 4 python -m pytest tests/test_h5_mpi.py -v

# ---------------------------------------------------------------------------
# Rust environment setup (dispatches to backend-specific target)
# ---------------------------------------------------------------------------

rust-sandbox: rust-sandbox-$(container)
rust-static-sandbox: rust-static-sandbox-$(container)

# --- Apptainer sandboxes ---

rust-sandbox-apptainer:
	@if [ ! -d "$(RUST_SANDBOX)" ]; then \
		echo "Building Rust Apptainer sandbox (glibc/dynamic)..."; \
		cd $(RUST_DIR) && apptainer build --fakeroot --sandbox rust_sandbox docker://rust:1.86-bookworm; \
		apptainer exec --fakeroot --writable $(RUST_SANDBOX) bash -c \
			"apt-get update && apt-get install -y libhdf5-dev pkg-config cmake && rm -rf /var/lib/apt/lists/*"; \
	else \
		echo "Rust sandbox already exists."; \
	fi

rust-static-sandbox-apptainer:
	@if [ ! -d "$(RUST_STATIC_SANDBOX)" ]; then \
		echo "Building Rust Apptainer sandbox (musl/static)..."; \
		cd $(RUST_DIR) && apptainer build --fakeroot --sandbox rust_static_sandbox docker://rust:1.86-alpine; \
		apptainer exec --fakeroot --writable $(RUST_STATIC_SANDBOX) bash -c \
			"apk add --no-cache hdf5-dev hdf5-static musl-dev pkgconf cmake make perl zlib-static && rustup target add x86_64-unknown-linux-musl"; \
	else \
		echo "Rust static sandbox already exists."; \
	fi

# --- Docker images ---

rust-sandbox-docker:
	@if ! docker image inspect $(DOCKER_IMAGE) >/dev/null 2>&1; then \
		echo "Building Docker image $(DOCKER_IMAGE) (glibc/dynamic)..."; \
		printf 'FROM rust:1.86-bookworm\nRUN apt-get update && apt-get install -y libhdf5-dev pkg-config cmake && rm -rf /var/lib/apt/lists/*\nWORKDIR /work\n' \
			| docker build -t $(DOCKER_IMAGE) -; \
	else \
		echo "Docker image $(DOCKER_IMAGE) already exists."; \
	fi

rust-static-sandbox-docker:
	@if ! docker image inspect $(DOCKER_IMAGE_STATIC) >/dev/null 2>&1; then \
		echo "Building Docker image $(DOCKER_IMAGE_STATIC) (musl/static)..."; \
		printf 'FROM rust:1.86-alpine\nRUN apk add --no-cache bash hdf5-dev hdf5-static musl-dev pkgconf cmake make perl zlib-static && rustup target add x86_64-unknown-linux-musl\nWORKDIR /work\n' \
			| docker build -t $(DOCKER_IMAGE_STATIC) -; \
	else \
		echo "Docker image $(DOCKER_IMAGE_STATIC) already exists."; \
	fi

# --- Native (no-op) ---

rust-sandbox-native:
	@echo "Native mode: no container (ensure libhdf5-dev, pkg-config, cmake, cargo are installed)"

rust-static-sandbox-native:
	@echo "Native mode: no container (ensure hdf5-dev, hdf5-static, musl target are installed)"

# ---------------------------------------------------------------------------
# Rust builds
# ---------------------------------------------------------------------------

build-rust: build-rust-release

build-rust-debug: $(NEED_DYNAMIC_ENV)
	$(ENSURE_DYNAMIC_DIRS)
	$(RUN_DYNAMIC) "cd $(WORKDIR) && cargo build"
	cp $(DYNAMIC_TARGET)/debug/har $(RUST_BIN)
	@echo "Built: $(RUST_BIN) (debug, dynamic, $(container))"

build-rust-release: $(NEED_DYNAMIC_ENV)
	$(ENSURE_DYNAMIC_DIRS)
	$(RUN_DYNAMIC) "cd $(WORKDIR) && cargo build --release"
	cp $(DYNAMIC_TARGET)/release/har $(RUST_BIN)
	@echo "Built: $(RUST_BIN) (release, dynamic, $(container))"

build-rust-static: $(NEED_STATIC_ENV)
	$(ENSURE_STATIC_DIRS)
	$(RUN_STATIC) "cd $(WORKDIR) && RUSTFLAGS='-C target-feature=+crt-static' cargo build --features static-hdf5 --target x86_64-unknown-linux-musl"
	cp $(STATIC_TARGET)/x86_64-unknown-linux-musl/debug/har $(RUST_BIN_STATIC)
	@echo "Built: $(RUST_BIN_STATIC) (debug, static, $(container))"

build-rust-release-static: $(NEED_STATIC_ENV)
	$(ENSURE_STATIC_DIRS)
	$(RUN_STATIC) "cd $(WORKDIR) && RUSTFLAGS='-C target-feature=+crt-static' cargo build --features static-hdf5 --release --target x86_64-unknown-linux-musl"
	cp $(STATIC_TARGET)/x86_64-unknown-linux-musl/release/har $(RUST_BIN_STATIC)
	@echo "Built: $(RUST_BIN_STATIC) (release, static, $(container))"

# ---------------------------------------------------------------------------
# Rust utility
# ---------------------------------------------------------------------------

test-rust: $(NEED_DYNAMIC_ENV)
	$(ENSURE_DYNAMIC_DIRS)
	$(RUN_DYNAMIC) "cd $(WORKDIR) && cargo test"

lint: lint-rust lint-python

lint-rust: $(NEED_DYNAMIC_ENV)
	$(ENSURE_DYNAMIC_DIRS)
	$(RUN_DYNAMIC) "cd $(WORKDIR) && cargo clippy -- -D warnings"

lint-python:
	python3 -m py_compile src/har.py
	python3 -m py_compile src/har_bagit.py

clean-rust:
	rm -rf $(RUST_DIR)/target $(RUST_DIR)/.cargo_home $(RUST_DIR)/.rustup_home
	rm -f $(RUST_BIN) $(RUST_BIN_STATIC)

DESTDIR ?= /usr/local/bin
install-rust:
	@test -f $(RUST_BIN) || test -f $(RUST_BIN_STATIC) || { echo "No binaries found. Run a build target first."; exit 1; }
	@test -f $(RUST_BIN) && install -m 755 $(RUST_BIN) $(DESTDIR)/har-rust && echo "Installed $(DESTDIR)/har-rust" || true
	@test -f $(RUST_BIN_STATIC) && install -m 755 $(RUST_BIN_STATIC) $(DESTDIR)/har-rust-static && echo "Installed $(DESTDIR)/har-rust-static" || true
