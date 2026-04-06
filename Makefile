SHELL := /bin/bash
ROOT := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

RUST_DIR := $(ROOT)/src/rust
RUST_SANDBOX := $(RUST_DIR)/rust_sandbox
RUST_STATIC_SANDBOX := $(RUST_DIR)/rust_static_sandbox
RUST_BIN := $(RUST_DIR)/har-rust
RUST_BIN_STATIC := $(RUST_DIR)/har-rust-static

APPTAINER_RUN := apptainer exec \
	--bind $(RUST_DIR):/work \
	--env CARGO_HOME=/work/.cargo_home \
	--env HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial \
	$(RUST_SANDBOX) bash -c

APPTAINER_RUN_STATIC := apptainer exec \
	--bind $(RUST_DIR):/work \
	--env CARGO_HOME=/work/.cargo_home \
	--env RUSTUP_HOME=/work/.rustup_home \
	$(RUST_STATIC_SANDBOX) bash -c

.PHONY: nothing all clean test test-python test-rust help \
	rust-sandbox rust-static-sandbox \
	build-rust build-rust-debug build-rust-release \
	build-rust-static build-rust-release-static \
	clean-rust install-rust

nothing:

help:
	@echo "Rust build targets:"
	@echo "  build-rust                - alias for build-rust-release"
	@echo "  build-rust-debug          - debug build (dynamic linking)"
	@echo "  build-rust-release        - release build (dynamic linking)"
	@echo "  build-rust-static         - debug build (static musl)"
	@echo "  build-rust-release-static - release build (static musl)"
	@echo ""
	@echo "Rust utility targets:"
	@echo "  rust-sandbox              - create dynamic-build sandbox"
	@echo "  rust-static-sandbox       - create static/musl-build sandbox"
	@echo "  test-rust                 - run Rust tests"
	@echo "  clean-rust                - remove target/ and cargo/rustup homes"
	@echo "  install-rust              - copy release binaries to DESTDIR"
	@echo ""
	@echo "General targets:"
	@echo "  test                      - run all tests"
	@echo "  test-python               - run Python tests"
	@echo "  clean                     - remove Python build artifacts"

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
# Rust sandboxes
# ---------------------------------------------------------------------------

rust-sandbox:
	@if [ ! -d "$(RUST_SANDBOX)" ]; then \
		echo "Building Rust Apptainer sandbox (glibc/dynamic)..."; \
		cd $(RUST_DIR) && apptainer build --fakeroot --sandbox rust_sandbox docker://rust:1.86-bookworm; \
		apptainer exec --fakeroot --writable $(RUST_SANDBOX) bash -c \
			"apt-get update && apt-get install -y libhdf5-dev pkg-config cmake && rm -rf /var/lib/apt/lists/*"; \
	else \
		echo "Rust sandbox already exists."; \
	fi

rust-static-sandbox:
	@if [ ! -d "$(RUST_STATIC_SANDBOX)" ]; then \
		echo "Building Rust Apptainer sandbox (musl/static)..."; \
		cd $(RUST_DIR) && apptainer build --fakeroot --sandbox rust_static_sandbox docker://rust:1.86-alpine; \
		apptainer exec --fakeroot --writable $(RUST_STATIC_SANDBOX) bash -c \
			"apk add --no-cache hdf5-dev hdf5-static musl-dev pkgconf cmake make perl zlib-static && rustup target add x86_64-unknown-linux-musl"; \
	else \
		echo "Rust static sandbox already exists."; \
	fi

# ---------------------------------------------------------------------------
# Rust builds
# ---------------------------------------------------------------------------

build-rust: build-rust-release

build-rust-debug: rust-sandbox
	$(APPTAINER_RUN) "cd /work && cargo build"
	cp $(RUST_DIR)/target/debug/har $(RUST_BIN)
	@echo "Built: $(RUST_BIN) (debug, dynamic)"

build-rust-release: rust-sandbox
	$(APPTAINER_RUN) "cd /work && cargo build --release"
	cp $(RUST_DIR)/target/release/har $(RUST_BIN)
	@echo "Built: $(RUST_BIN) (release, dynamic)"

build-rust-static: rust-static-sandbox
	$(APPTAINER_RUN_STATIC) "cd /work && RUSTFLAGS='-C target-feature=+crt-static' cargo build --target x86_64-unknown-linux-musl"
	cp $(RUST_DIR)/target/x86_64-unknown-linux-musl/debug/har $(RUST_BIN_STATIC)
	@echo "Built: $(RUST_BIN_STATIC) (debug, static)"

build-rust-release-static: rust-static-sandbox
	$(APPTAINER_RUN_STATIC) "cd /work && RUSTFLAGS='-C target-feature=+crt-static' cargo build --release --target x86_64-unknown-linux-musl"
	cp $(RUST_DIR)/target/x86_64-unknown-linux-musl/release/har $(RUST_BIN_STATIC)
	@echo "Built: $(RUST_BIN_STATIC) (release, static)"

# ---------------------------------------------------------------------------
# Rust utility
# ---------------------------------------------------------------------------

test-rust: rust-sandbox
	$(APPTAINER_RUN) "cd /work && cargo test"

clean-rust:
	rm -rf $(RUST_DIR)/target $(RUST_DIR)/.cargo_home $(RUST_DIR)/.rustup_home
	rm -f $(RUST_BIN) $(RUST_BIN_STATIC)

DESTDIR ?= /usr/local/bin
install-rust:
	@test -f $(RUST_BIN) || test -f $(RUST_BIN_STATIC) || { echo "No binaries found. Run a build target first."; exit 1; }
	@test -f $(RUST_BIN) && install -m 755 $(RUST_BIN) $(DESTDIR)/har-rust && echo "Installed $(DESTDIR)/har-rust" || true
	@test -f $(RUST_BIN_STATIC) && install -m 755 $(RUST_BIN_STATIC) $(DESTDIR)/har-rust-static && echo "Installed $(DESTDIR)/har-rust-static" || true