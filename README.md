# Jax implementation of 'Capturing Label Characteristics in VAEs'

## Introduction
This project is a reimplementation of the following papers

- Semi-supervised Learning with Deep Generative Models
- Capturing Label Characteristics in VAEs

This branch in particular deals with the experiments we led. All the models were lauched 4 times with different seeds on GPU from the P2200 or RTX A2000 families (freely available computer with GPU for Polytechnique's students). All models should run in less than 5 minutes on a single GPU for MNIST. For other datasets, the training time per epoch varies significantly from one GPU to another (and from one dataset to another).

All the models are checkpointed at the end of the training and they are then analysed in the following notebooks:
- test.ipynb: for M2VAE
- test_ccvae.ipynb: for CCVAE
- analyze_experiments.ipynb: for both models in term of accuracy and Expected Calibration Error (ECE)

in Jax/Numpyro.

## Installation

### Prerequisites
Python version 3.9 or higher.

### Setup
To set up the project environment:

Clone the repository:
```
git clone https://github.com/MattX20/CCVAE_Jax
```

Navigate to the cloned directory:
```
cd CCVAE_Jax
```

Install the required packages using:
```
pip install -r requirements.txt
```
If you want a GPU-compatible jax install, please refer to the [jax installation page](https://jax.readthedocs.io/en/latest/installation.html).
A GPU-compatible installation of torch is not usefull, as torch is only used to load the datasets.

## References
#### M2 semi-supervised VAE model
```
@article{kingma2014semi,
  title={Semi-supervised learning with deep generative models},
  author={Kingma, Durk P and Mohamed, Shakir and Jimenez Rezende, Danilo and Welling, Max},
  journal={Advances in neural information processing systems},
  volume={27},
  year={2014}
}
```
#### CCVAE model
```
@article{joy2020capturing,
  title={Capturing label characteristics in vaes},
  author={Joy, Tom and Schmon, Sebastian M and Torr, Philip HS and Siddharth, N and Rainforth, Tom},
  journal={arXiv preprint arXiv:2006.10102},
  year={2020}
}
```
## License

This project is licensed under the [MIT license](LICENSE).