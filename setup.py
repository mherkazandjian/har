from setuptools import setup, find_packages

setup(
    name='hdf5-archive-utility',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'h5py',
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'har=h5:main',
            'h5=h5:main',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A utility for creating, appending, extracting, and listing HDF5 archives.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/hdf5-archive-utility',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)