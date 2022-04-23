from setuptools import setup

with open('README.txt', 'r') as f:
    long_description = f.read()

setup(
    name='torusgrid',
    version='1.0',
    description='periodic fields with definite dimensions equipped with FFT operations',
    author='Michael Wang',
    author_email='mike.wang96029@gmail.com',
    packages=['src'],
    install_requires=['numpy', 'scipy', 'pyfftw', 'matplotlib', 'tqdm'],
    long_description=long_description
)
