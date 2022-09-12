# tdlda

Python implementation of Latent Dirichlet Allocation based on *Scalable Tensor Orthogonal Decomposition* (STOD) from:

> Scalable Moment-Based Inference for Latent Dirichlet Allocation 
> > C. Wang, X. Liu, Y. Song, and J. Han, 2014

and https://github.com/UIUC-data-mining/STOD.

## Examples

The `tests/test_lda.py` file demonstrates the use of STOD-based LDA to perform topic analysis on the *NYT dataset* (included in the `tests/assets` folder) and the *20 newsgroups* dataset (from *sklearn*).
It also implements the algorithm's evaluation using synthetic data from Want et. al. 2014. 