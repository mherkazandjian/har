import os
import tempfile
import h5py
import pytest
from io import StringIO
import contextlib

# Import the functions from your h5.py module
from har import pack_or_append_to_h5, extract_h5_to_directory, list_h5_contents

@pytest.fixture
def test_env():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = temp_dir

        # Create source directory with files
        src_dir = os.path.join(base_dir, "src")
        os.makedirs(src_dir)

        # Create file1.txt
        file1 = os.path.join(src_dir, "file1.txt")
        with open(file1, "w") as f:
            f.write("This is file 1.")

        # Create subdirectory and file2.txt
        sub_dir = os.path.join(src_dir, "subdir")
        os.makedirs(sub_dir)
        file2 = os.path.join(sub_dir, "file2.txt")
        with open(file2, "w") as f:
            f.write("This is file 2 in subdir.")

        # Define archive and extract paths
        archive_path = os.path.join(base_dir, "archive.h5")
        extract_dir = os.path.join(base_dir, "extract")
        os.makedirs(extract_dir, exist_ok=True)

        yield {
            'src_dir': src_dir,
            'file1': file1,
            'file2': file2,
            'sub_dir': sub_dir,
            'archive_path': archive_path,
            'extract_dir': extract_dir
        }

def test_archive_creation(test_env):
    pack_or_append_to_h5([test_env['src_dir']], test_env['archive_path'], 'w')
    assert os.path.exists(test_env['archive_path'])

    with h5py.File(test_env['archive_path'], 'r') as h5f:
        assert "file1.txt" in h5f
        assert "subdir" in h5f
        assert "file2.txt" in h5f["subdir"]

def test_append(test_env):
    pack_or_append_to_h5([test_env['src_dir']], test_env['archive_path'], 'w')

    file3 = os.path.join(test_env['src_dir'], "file3.txt")
    with open(file3, "w") as f:
        f.write("This is file 3.")

    pack_or_append_to_h5([file3], test_env['archive_path'], 'a')

    with h5py.File(test_env['archive_path'], 'r') as h5f:
        assert "file3.txt" in h5f

def test_extract_all(test_env):
    pack_or_append_to_h5([test_env['src_dir']], test_env['archive_path'], 'w')
    extract_h5_to_directory(test_env['archive_path'], test_env['extract_dir'])

    extracted_file1 = os.path.join(test_env['extract_dir'], "file1.txt")
    assert os.path.exists(extracted_file1)
    with open(extracted_file1, "r") as f:
        assert f.read() == "This is file 1."

    extracted_file2 = os.path.join(test_env['extract_dir'], "subdir", "file2.txt")
    assert os.path.exists(extracted_file2)
    with open(extracted_file2, "r") as f:
        assert f.read() == "This is file 2 in subdir."

def test_extract_single(test_env):
    pack_or_append_to_h5([test_env['src_dir']], test_env['archive_path'], 'w')
    extract_h5_to_directory(test_env['archive_path'], test_env['extract_dir'], file_key="file1.txt")

    extracted_file1 = os.path.join(test_env['extract_dir'], "file1.txt")
    assert os.path.exists(extracted_file1)
    with open(extracted_file1, "r") as f:
        assert f.read() == "This is file 1."

    extracted_file2 = os.path.join(test_env['extract_dir'], "subdir", "file2.txt")
    assert not os.path.exists(extracted_file2)

def test_list_contents(test_env):
    pack_or_append_to_h5([test_env['src_dir']], test_env['archive_path'], 'w')

    output = StringIO()
    with contextlib.redirect_stdout(output):
        list_h5_contents(test_env['archive_path'])
    out_str = output.getvalue()

    assert "file1.txt" in out_str
    assert "subdir" in out_str
