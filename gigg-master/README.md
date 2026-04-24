
<!-- README.md is generated from README.Rmd. Please edit that file -->

# R package `gigg`

# Group Inverse-Gamma Gamma Shrinkage for Sparse Regression with Grouping Structure

[![](https://img.shields.io/badge/devel%20version-0.2.1-blue.svg)](https://github.com/umich-cphds/gigg)
[![](https://img.shields.io/github/languages/code-size/umich-cphds/gigg.svg)](https://github.com/umich-cphds/gigg)

## Overview

This package implements a Gibbs sampler corresponding to a Group
Inverse-Gamma Gamma (GIGG) regression model with adjustment covariates.
Hyperparameters in the GIGG prior specification can either be fixed by
the user or can be estimated via Marginal Maximum Likelihood Estimation.

## Installation

If the devtools package is not yet installed, install it first:

``` r
install.packages('devtools')
```

``` r
# install the package from Github:
devtools::install_github('umich-cphds/gigg') 
```

Once installed, load the package:

``` r
library(gigg)
```

## Examples

GIGG regression Gibbs sampler with fixed hyperparameters:

``` r
X = concentrated$X
C = concentrated$C
Y = as.vector(concentrated$Y)
grp_idx = concentrated$grps

gf = gigg(X, C, Y, method = "fixed", grp_idx, n_burn_in = 500, n_samples = 1000, 
          n_thin = 1, verbose = TRUE, btrick = FALSE, stable_solve = TRUE)
```

GIGG regression Gibbs sampler with hyperparameter estimation via
Marginal Maximum Likelihood Estimation:

``` r
X = concentrated$X
C = concentrated$C
Y = as.vector(concentrated$Y)
grp_idx = concentrated$grps

gf_mmle = gigg(X, C, Y, method = "mmle", grp_idx, n_burn_in = 500, 
               n_samples = 1000, n_thin = 1, verbose = TRUE, btrick = FALSE, 
               stable_solve = TRUE)
```

### Current Suggested Citation

Boss, J., Datta, J., Wang, X., Park, S.K., Kang, J., & Mukherjee, B.
(2021). Group Inverse-Gamma Gamma Shrinkage for Sparse Regression with
Block-Correlated Predictors. arXiv preprint arXiv:2102.10670.
