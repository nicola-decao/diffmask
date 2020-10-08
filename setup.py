import os
from setuptools import setup
from setuptools import find_packages

setup(
    name="diffmask",
    version="0.1.0",
    author="Nicola De Cao",
    author_email="nicola.decao@gmail.com",
    description="Pytorch implementation of DiffMask",
    license="MIT",
    keywords="pytorch machine-learning deep-learning interpretability explainability",
    url="https://nicola-decao.github.io",
    download_url="https://github.com/nicola-decao/diffmask",
    long_description=open(os.path.join(os.path.dirname(__file__), "README.md")).read(),
    install_requires=[
        "torch>=1.5.0",
        "pytorch-lightning==0.7.5",
        "transformers>=2.9.0",
        "spacy==2.2.4",
        "torch-optimizer==0.0.1a9",
    ],
    packages=find_packages(),
)
