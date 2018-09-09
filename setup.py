from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='torch-baker',
    version='0.0.0',
    description='A wrapper for PyTorch that use chain-like syntax to custom training/testing process',
    license="MIT",
    long_description=long_description,
    author='huisedenanhai',
    packages=['torchbaker']
)
