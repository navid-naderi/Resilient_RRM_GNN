# Learning Resilient Radio Resource Management Policies with Graph Neural Networks

This repository contains the source code for learning resilient resource management policies in wireless networks, including power control and user selection, via graph neural network (GNN) parameterizations. In particular, the aforementioned radio resource management (RRM) policies are trained in such a way to maximize a network-wide objective, while satisfying per-user minimum-capacity constraints that *adapt* to the underlying network conditions. For example, if a user is in poor network conditions (due to a poor signal-to-noise ratio (SNR) or a strong incoming interference-to-noise ratio (INR)), then the minimum-capacity constraint for that user is relaxed just enough to make the optimization problem feasible, hence allowing fair allocation of wireless resources across the network. We use a shared GNN architecture to parameterize the RRM policies, and we train the GNN parameters, alongside the rest of the optimization variables, using an unsupervised primal-dual approach. Please refer to [the accompanying paper](https://arxiv.org/abs/2203.ABCDE) for more details.

## Training and Evaluation

To train the RRM policies for networks with `m` transmitters and `n` receivers, run the following command:

```
python3 main_train.py --m m --n n
```

Furthermore, to evaluate the RRM policies, which were trained for networks with `m_train` transmitters and `n_train` receivers, on networks with `m` transmitters and `n` receivers, run the following command:

```
python3 main_eval.py --m m --n n --m_train m_train --n_train n_train
```

Check `main_train.py` and `main_eval.py` for other optional arguments for the training and evaluation procedures, respectively, such as the number of samples (i.e., network configurations), GNN size, number of training epochs, learning rates, etc.

The created datasets during the training/evaluation procedures get saved under `./data`. Moreover, the training/evaluation results, as well as the best models, get saved under `./results`.

## Dependencies

* [PyTorch](https://pytorch.org/)
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
* [Scatter](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html)
* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [tqdm](https://tqdm.github.io/)
* [Matplotlib](https://matplotlib.org/)

## Citation

Please use the following BibTeX citation if you use this repository in your work:

```
@article{Resilient_RRM_GNN_naderializadeh2022,
  title={Learning Resilient Radio Resource Management Policies with Graph Neural Networks},
  author={Navid Naderializadeh and Mark Eisen and Alejandro Ribeiro},
  journal={arXiv preprint arXiv:2203.ABCDE},
  year={2022}
}
```
