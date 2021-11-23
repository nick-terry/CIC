# CIC (Cross-Entropy Information Criterion)

The CIC is a statistic which can be used to design algorithms for the <a href="https://en.wikipedia.org/wiki/Boltzmann_distribution"> Boltzmann approximation problem </a>. In <a href="https://arxiv.org/abs/1704.04315"> our paper </a>, the CIC is theoretically developed and an algorithm is provided for using the CIC to solve Boltzmann approximation problems.

<div><img src="https://i.imgur.com/dYqp8oF.png" alt="Successive approximation of a probability distribution using our algorithm."/></div>

This repository contains:
1. An implementation for an algorithm described in the paper which solves the Boltzmann approximation problem using the CIC.
2. Code for reproducing all experimental results in the paper.

Please note that code for running the experiments utilizes multi-core processing functionality which is specific to Unix-based operating systems. We cannot guarantee that the code will work for you without slight modification.

<h> Installation and Use </h>

To install the CIC implementation, we recommend cloning this repository and using the pip package manager with the included wheel file i.e.

```
pip install CIC-1.0-py3-none-any.whl
```

The dependencies for the CIC implementation included with the package will cover any dependencies for experiment scripts as well.

To see how to use the CIC implementation, you can refer to example scripts in the ```Experiments\Bias Tests``` folder.
