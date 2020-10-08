# DiffMask

## Overview
This library contains a Pytorch implementation of Differentiable Masking Explainer (Diffmask), as presented in [[1]](#citation)(https://arxiv.org/abs/2004.14992).

## Dependencies

* **python>=3.6**
* **pytorch>=1.5**: https://pytorch.org
* **pytorch-lightning==0.7.5**: https://pytorch-lightning.readthedocs.io
* **transformers>=2.9.0**: https://github.com/huggingface/transformers
* **torch-optimizer>=0.0.1a9**: https://github.com/jettify/pytorch-optimizer
* **matplotlib>=3.1.1**: https://matplotlib.org

*Notice that older or newer version could work but they were not tested.*

## Installation

To install, run

```bash
$ python setup.py install
```

To donwload datasets run
```bash
$ ./scripts/download_datasets.sh
```

To download models, use the following [link](https://mega.nz/folder/19QwELDK#CxKo4UO2P8KDt5TwYWfhmA). **Note that these are not the exaclt same models used for the paper.**


## Structure
* [diffmask](https://github.com/nicola-decao/diffmask/tree/master/diffmask): Contains the source code for DiffMask.

We have 5 jupyter notebbok with the code for reproducing some of the results from our work [[1](#citation)]. **Note that since i) the code was refactored and ii) we were not able to realise the exact same models used for the paper, re-generated plots and tables might differ from the ones in our work.**

* [QuestionAnsweringSquadAnalysis](https://github.com/nicola-decao/diffmask/tree/master/QuestionAnsweringSquadAnalysis.ipynb) Plots and analysis using other methods of a BERT large model for question answering.
* [QuestionAnsweringSquadDiffMaskAnalysis](https://github.com/nicola-decao/diffmask/tree/master/QuestionAnsweringSquadDiffMaskAnalysis.ipynb) Plots and analysis using **DiffMask** of a BERT large model for question answering.
* [SentimentClassificationSSTAnalysis](https://github.com/nicola-decao/diffmask/tree/master/SentimentClassificationSSTAnalysis.ipynb) Plots and analysis using other methods of a BERT base model for sentiment analisys.
* [SentimentClassificationSSTDiffMaskAnalysis](https://github.com/nicola-decao/diffmask/tree/master/SentimentClassificationSSTDiffMaskAnalysis.ipynb) Plots and analysis using **DiffMask** of a BERT base model for sentiment analisys.
* [ToyTask](https://github.com/nicola-decao/diffmask/tree/master/ToyTask.ipynb) Training, analysis and plots for the artificial toy task using both other methods and **DiffMask**.

Please cite [[1](#citation)] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [Nicola De Cao](mailto:nicola.decao@gmail.com).

## License
MIT

## Citation
```
[1] De Cao, N., Schlichtkrull, M., Aziz, W., Titov, I. (2020).
How do Decisions Emerge across Layers in Neural Models? 
Interpretation with Differentiable Masking
In Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing.
```

BibTeX format:
```
@article{decao2020decisions,
  title={How do Decisions Emerge across Layers in Neural Models?},
  author={
    De Cao, Nicola and
    Aziz, Wilker and
    Titov, Ivan},
  journal={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  year={2020}
}
```