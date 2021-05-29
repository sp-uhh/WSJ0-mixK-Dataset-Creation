# WSJ0-mixK Dataset Creation

### Date : 28/05/2021
### Author : Leroy Bartel : *leroy.bartel@outlook.com*

## 1- Description

This package can be used to generate a dataset suited for training, testing, and evaluating
neural networks on speaker count agnostic speech separation with an unknown number of simultaneous speakers. 

This work aims to extend the WSJ0-mix2 and WSJ0-mix3 datasets proposed in [1] for an arbitrary number of speakers.
The provided Python script can be used to generate a WSJ0-mix-k dataset of mixtures with k simultaneous speakers according to the method proposed in [1].
In order to obtain the full WSJ0-mixK dataset that consists of K many WSJ0-mix-k subsets (k in {1, 2, ..., K}), the provided script has to be run for each k in {1, 2, ..., K} with the argument --k set accordingly. The Python code is an adaption of the scripts provided by [2] and is used to generate the dataset employed in [3] that consists of four WSJ0-mix-k subsets with k in {1, 2, 3, 4}, i.e. K=4.

## 2- Requirements

- **Matlab R2018** or later


- **Python 3** with packages:
    - numpy, scipy, soundfile, pandas, matlab.engine
    

- A directory containing the WSJ0 dataset (containing the wsj0/ folder)

## 3- Usage

- *Set the following arguments when running the script:*
    - *--output-dir: The target output directory for the WSJ0-mixK dataset*
    - *--wsj0-root: The path to the folder containing the dataset wsj0/*
    - *--sr-str: Whether to generate the dataset with utterances sampled at 8 kHz and/or 16 kHz (8k / 16k / both)*
    - *--data-length: Whether to use the maximum or minimum length of the selected utterances (min / max / both)*
    - *--k: The number of speakers to mix in each mixture*
    
    
- Run the script in a command line: 
    `python3 create_wsj0_mix_k_subset.py --output-dir=../path/to/dir/of/choice --wsj0-root=/path/to/wsj0/ --sr-str=8k --data-length=min --k=2`
  

## 4- References

 [1] J. R. Hershey, Z. Chen, J. Le Roux, and S. Watanabe, "Deep Clustering: Discriminative Embeddings for Segmentation and Separation,"
 in 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 31–35, 2016.

 [2] Wichern, Gordon, et al. "WHAM!: Extending speech separation to noisy environments." 
    arXiv preprint arXiv:1907.01160 (2019). [https://wham.whisper.ai/](https://wham.whisper.ai/)

 [3] L. Bartel, "Deep Learning based Speaker Count Estimation for Single-Channel Speech Separation." Master's Thesis. University of Hamburg. March 2021.
