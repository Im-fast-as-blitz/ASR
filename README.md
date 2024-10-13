# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository is a system for training the [`deepspeech2`](http://proceedings.mlr.press/v48/amodei16.pdf) model for an ASR task.

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=deepspeech2
```

Where the model will learn 50 epochs on all datasets from leebspeech

To run inference (evaluate the model or save predictions):

Dowload model:

```bash
python3 download_model.py
```

For predicts on test-clean dataset:

```bash
python3 inference.py -cn=inference
```

For predicts on test-other dataset:

```bash
python3 inference.py -cn=inference_other
```

To calc cer/wer

```bash
python3 calc_wer_cer.py --dir_path dir
```

Where dir is path to your dir (example "/ASR/data/saved/predict/test")

## About work

All my graphs with experiments on obtaining my solution can be found [`here`](https://wandb.ai/rodion-chernomordin/pytorch_template_asr_example/overview) (there are also separate conclusions for each of the augmentation)

I will keep my course of action in the same order as the graphs are arranged. First of all, I made baseline and one batch test (which will be better later), changed max lr to 1-e3, added log-scaling to spectrograms (at least better perception) and a self-written beam search (in the corresponding graph you can see how it works - goes through all possible options. As proof that my beam search is working correctly, I have displayed it in every training).

You can see that all the graphs give out a strange loss and bad metrics - the mistake was that I incorrectly calculated the length of the output sequences of probabilities.

At this moment, my learning model has the following hyperparameters:

- start lr 1e-4
- max lr 1e-3
- num epochs 50 (200 iter)
- batch size 10
- train dataset: clean-100
- beam size 10
- model parametrs 28086844

I added 4 augmentations: LowPassFilter, HighPassFilter, Color Noise, BandPassFilter. The probability of each one being triggered is about 1/4. 
The result on clean data turned out to be slightly worse than without it, but I was ready to do it, because then my model would work a little better with "other" data and there would be no overfiting in the future.

My next and final step was to expand the amount of data (use all three datasets) and increase the batch size to 64.

[`Final model`](https://drive.google.com/file/d/1LoU_kCzl20hM5p709teRPB0a4Jib8VK-/view?usp=sharing)

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
