"""Regression guards for two MPI-path bugs that were invisible to CI.

These tests do NOT require mpi4py / a real MPI runtime. They exercise the
code paths that used to raise at module-load or argument-dispatch time:

  1. har_mpi.py imported the symbol `_sha256_bytes` from har_bagit, which does
     not exist. The first `import har_mpi` raised ImportError.
  2. har.py's CLI dispatcher called `mpi_pack_bagit(..., batch_size=bs, ...)`
     before assigning `bs = parse_batch_size(args.batch_size)`. The first MPI
     create raised NameError.

Both bugs live outside the actual MPI runtime — they're plain Python
scoping/import errors — so we catch them here with stdlib + mock.
"""

import os
import sys
from unittest import mock

import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, SRC_DIR)


def test_har_mpi_imports_cleanly():
    # Forces a fresh import of har_mpi so a cached module can't mask a
    # regression. If har_mpi ever re-imports a missing symbol from har_bagit,
    # this test fails with ImportError at collection time.
    for mod in ("har_mpi", "har_bagit"):
        sys.modules.pop(mod, None)
    import har_mpi  # noqa: F401


def test_har_mpi_create_dispatch_passes_batch_size():
    # Invokes har.main() with --mpi --bagit -c and verifies the dispatcher
    # calls mpi_pack_bagit without NameError, and forwards the parsed
    # batch_size. We patch mpi_pack_bagit so no real MPI work happens.
    for mod in ("har", "har_mpi", "har_bagit"):
        sys.modules.pop(mod, None)
    import har
    import har_mpi

    called = {}

    def fake_pack(sources, output_h5, batch_size=None, verbose=False):
        called["sources"] = sources
        called["batch_size"] = batch_size
        # Create a minimal stub output file so later code paths don't choke.
        open(output_h5, 'wb').close()

    with mock.patch.object(har_mpi, 'mpi_pack_bagit', fake_pack), \
         mock.patch.object(har_mpi, 'mpi_extract_bagit', lambda *a, **k: None), \
         mock.patch.object(har_mpi, 'mpi_list_bagit', lambda *a, **k: None):
        src_dir = os.path.dirname(__file__)
        argv = [
            'har', '--mpi', '--bagit', '-cf', '/tmp/_mpi_regression_test.h5',
            '--batch-size', '8K', src_dir,
        ]
        with mock.patch.object(sys, 'argv', argv):
            har.main()

    assert called["batch_size"] == 8 * 1024, \
        f"batch_size not forwarded; got {called.get('batch_size')!r}"
    assert called["sources"] == [src_dir]
