# python -m unittest test_h5.py
import os
import tempfile
import unittest
import h5py
import contextlib
from io import StringIO

# Import the functions from your h5.py module.
# Adjust the import if your script/module is named differently.
from h5 import pack_or_append_to_h5, extract_h5_to_directory, list_h5_contents

class TestH5Operations(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = self.temp_dir.name

        # Create a source directory with some files and a subdirectory.
        self.src_dir = os.path.join(self.base_dir, "src")
        os.makedirs(self.src_dir)
        # Create file1.txt
        self.file1 = os.path.join(self.src_dir, "file1.txt")
        with open(self.file1, "w") as f:
            f.write("This is file 1.")
        # Create a subdirectory and file2.txt inside it.
        self.sub_dir = os.path.join(self.src_dir, "subdir")
        os.makedirs(self.sub_dir)
        self.file2 = os.path.join(self.sub_dir, "file2.txt")
        with open(self.file2, "w") as f:
            f.write("This is file 2 in subdir.")

        # Define the path for the HDF5 archive and an extraction directory.
        self.archive_path = os.path.join(self.base_dir, "archive.h5")
        self.extract_dir = os.path.join(self.base_dir, "extract")
        os.makedirs(self.extract_dir, exist_ok=True)

    def tearDown(self):
        # Clean up the temporary directory.
        self.temp_dir.cleanup()

    def test_archive_creation(self):
        # Archive (create) the src directory.
        pack_or_append_to_h5([self.src_dir], self.archive_path, 'w')
        self.assertTrue(os.path.exists(self.archive_path))
        # Verify that file1.txt and file2.txt (within subdir) are in the archive.
        with h5py.File(self.archive_path, 'r') as h5f:
            self.assertIn("file1.txt", list(h5f.keys()))
            self.assertIn("subdir", list(h5f.keys()))
            # Check that file2.txt is inside the "subdir" group.
            self.assertIn("file2.txt", list(h5f["subdir"].keys()))

    def test_append(self):
        # Create archive first.
        pack_or_append_to_h5([self.src_dir], self.archive_path, 'w')
        # Create a new file that was not in the original archive.
        file3 = os.path.join(self.src_dir, "file3.txt")
        with open(file3, "w") as f:
            f.write("This is file 3.")
        # Append the new file.
        pack_or_append_to_h5([file3], self.archive_path, 'a')
        # Verify that file3.txt was added.
        with h5py.File(self.archive_path, 'r') as h5f:
            self.assertIn("file3.txt", list(h5f.keys()))

    def test_extract_all(self):
        # Archive the source directory.
        pack_or_append_to_h5([self.src_dir], self.archive_path, 'w')
        # Extract the entire archive.
        extract_h5_to_directory(self.archive_path, self.extract_dir)
        # Check that file1.txt was extracted.
        extracted_file1 = os.path.join(self.extract_dir, "file1.txt")
        self.assertTrue(os.path.exists(extracted_file1))
        with open(extracted_file1, "r") as f:
            self.assertEqual(f.read(), "This is file 1.")
        # Check that file2.txt (in subdir) was extracted.
        extracted_file2 = os.path.join(self.extract_dir, "subdir", "file2.txt")
        self.assertTrue(os.path.exists(extracted_file2))
        with open(extracted_file2, "r") as f:
            self.assertEqual(f.read(), "This is file 2 in subdir.")

    def test_extract_single(self):
        # Archive the source directory.
        pack_or_append_to_h5([self.src_dir], self.archive_path, 'w')
        # Extract only file1.txt.
        extract_h5_to_directory(self.archive_path, self.extract_dir, file_key="file1.txt")
        extracted_file1 = os.path.join(self.extract_dir, "file1.txt")
        self.assertTrue(os.path.exists(extracted_file1))
        with open(extracted_file1, "r") as f:
            self.assertEqual(f.read(), "This is file 1.")
        # Ensure file2.txt was not extracted.
        extracted_file2 = os.path.join(self.extract_dir, "subdir", "file2.txt")
        self.assertFalse(os.path.exists(extracted_file2))

    def test_list_contents(self):
        # Archive the source directory.
        pack_or_append_to_h5([self.src_dir], self.archive_path, 'w')
        # Capture the output from list_h5_contents.
        output = StringIO()
        with contextlib.redirect_stdout(output):
            list_h5_contents(self.archive_path)
        out_str = output.getvalue()
        # Check that the output lists expected file keys.
        self.assertIn("file1.txt", out_str)
        # Note: depending on how the keys are stored, the subdirectory might appear as
        # "subdir/file2.txt" or as separate groups. Adjust the test if necessary.
        self.assertTrue("subdir" in out_str)

if __name__ == '__main__':
    unittest.main()
