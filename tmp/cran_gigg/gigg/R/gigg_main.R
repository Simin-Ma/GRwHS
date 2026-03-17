#' GIGG regression
#' 
#' Perform GIGG (Group Inverse-Gamma Gamma) regression.
#' This package implements a Gibbs sampler corresponding to a Group 
#' Inverse-Gamma Gamma (GIGG) regression model with adjustment covariates. 
#' Hyperparameters in the GIGG prior specification can either be fixed by the 
#' user or can be estimated via Marginal Maximum Likelihood Estimation.
#'
#' @param X A (n x p) matrix of covariates that to apply GIGG shrinkage on.
#' @param C A (n x k) matrix of covariates that to apply no shrinkage on (typically intercept + adjustment covariates).
#' @param Y A length n vector of responses.
#' @param method Either `fixed` for GIGG regression with fixed hyperparameters or `mmle` for GIGG regression with MMLE.
#' Defaults to method = "mmle".
#' @param grp_idx A length p integer vector indicating which group of the G groups the p covariates in X belong to.
#' The `grp_idx` vector must be a sequence from 1 to G with no skips. A valid example is 1,1,1,2,2,3,3,3,4,5,5.
#' @param alpha_inits A length k vector containing initial values for the regression coefficients corresponding to C.
#' @param beta_inits A length p vector containing initial values for the regression coefficients corresponding to X.
#' @param a A length G vector of shape parameters for the prior on the group shrinkage parameters.
#' The `a` parameter is only used if the user selects `method = 'fixed'`. If `method = 'mmle'`,
#' then `a = rep(1/n, length(unique(grp_idx)))`.
#' @param b A length G vector of shape parameters for the prior on the individual shrinkage parameters. If `method = 'mmle'`,
#' then the `b` is used as an inital value for the MMLE procedure.
#' @param sigma_sq_init Initial value for the residual error variance (double).
#' @param tau_sq_init Initial value for the global shrinkage parameter (double).
#' @param n_burn_in The number of burn-in samples (integer).
#' @param n_samples The number of posterior draws (integer).
#' @param n_thin The thinning interval (integer).
# #' @param stable_const Parameter that controls numerical stability of the algorithm (double).
#' @param verbose Boolean value which indicates whether or not to print the progress of the Gibbs sampler.
#' @param btrick Boolean value which indicates whether or not to use the computational trick in Bhattacharya et al. (2016). Only recommended if number of covariates is much larger than the number of observations.
#' @param stable_solve Boolean value which indicates whether or not to use Cholesky decomposition during the update of the regression coefficients corresponding to X. In our experience, `stable_solve = TRUE` is slightly slower, but more stable.
#' @return A list containing 
#' \itemize{
#'  \item{"draws"}{ - A list containing the posterior draws of \cr
#' (1) the regression coefficients (alphas and betas) \cr
#' (2) the individual shrinkage parameters (lambda_sqs) \cr
#' (3) the group shrinkage parameters (gamma_sqs) \cr
#' (4) the global shrinkage parameter (tau_sqs) and \cr
#' (5) the residual error variance (sigma_sqs). \cr The list also contains details
#' regarding the dataset (X, C, Y, grp_idx) and Gibbs sampler details
#' (n_burn_in, n_samples, and n_thin).}
#'  \item{"beta.hat"}{ - Posterior mean of betas}
#'  \item{"beta.lcl.95"}{ - 95% credible interval lower bound of betas}
#'  \item{"beta.ucl.95"}{ - 95% credible interval upper bound of betas}
#'  \item{"alpha.hat"}{ - Posterior mean of alpha}
#'  \item{"alpha.lcl.95"}{ - 95% credible interval lower bound of alphas}
#'  \item{"alpha.ucl.95"}{ - 95% credible interval upper bound of alphas}
#'  \item{"sigma_sq.hat"}{ - Posterior mean of sigma squared}
#'  \item{"sigma_sq.lcl.95"}{ - 95% credible interval lower bound of sigma sq.}
#'  \item{"sigma_sq.ucl.95"}{ - 95% credible interval upper bound of sigma sq.}
#' }
#' @references Boss, J., Datta, J., Wang, X., Park, S.K., Kang, J., & Mukherjee, B. 
#' (2021). Group Inverse-Gamma Gamma Shrinkage for Sparse Regression with Block-Correlated Predictors. 
#' \href{https://arxiv.org/abs/2102.10670}{arXiv}
#' @examples 
#' X = concentrated$X
#' C = concentrated$C
#' Y = as.vector(concentrated$Y)
#' grp_idx = concentrated$grps
#' alpha_inits = concentrated$alpha
#' beta_inits = concentrated$beta
#' 
#' gf = gigg(X, C, Y, method = "fixed", grp_idx, alpha_inits, beta_inits,
#'           n_burn_in = 200, n_samples = 500, n_thin = 1,  
#'           verbose = TRUE, btrick = FALSE, stable_solve = FALSE)
#' \donttest{
#' gf_mmle = gigg(X, C, Y, method = "mmle", grp_idx, alpha_inits, beta_inits,
#'                 n_burn_in = 200, n_samples = 500, n_thin = 1, 
#'                 verbose = TRUE, btrick = FALSE, 
#'                 stable_solve = FALSE)
#' }
#' 
gigg = function(X, C, Y, method = "mmle", grp_idx, alpha_inits = rep(0, ncol(C)), beta_inits = rep(0, ncol(X)), a = rep(0.5, length(unique(grp_idx))), b = rep(0.5, length(unique(grp_idx))),
                sigma_sq_init = 1, tau_sq_init = 1, n_burn_in = 500, n_samples = 1000, n_thin = 1, verbose = TRUE, btrick = FALSE, stable_solve = TRUE) {
  
  lambda_sq_inits = rep(1, ncol(X))
  gamma_sq_inits = rep(1.0, length(unique(grp_idx)))
  nu_init = 1
  stable_const = 1e-07
  
  #Store useful quantites
  grp_size <- as.vector(table(grp_idx))
  grp_size_cs <- cumsum(grp_size)
  
  if(!is.matrix(X) | !is.matrix(C)) {
    stop("X and C must be matrices. At least one is of a different type.")
  }
  
  if(!is.vector(Y)) {
    stop("Y must be a vector.")
  }
  
  n = length(Y)
  p = ncol(X)
  k = ncol(C)
  group_ids = unique(grp_idx)
  G = length(group_ids)
  
  if(n != nrow(X) | n != nrow(C)) {
    stop("X and C matrices must have the same number of rows and match the length of the Y vector.")
  }
  
  if(length(grp_idx) != p) {
    stop("The argument grp_idx must have length p, where p in the number of columns in X.")
  }
  
  if(length(alpha_inits) != k) {
    stop("The argument alpha_inits must have length k, where k is the number of columns in C.")
  }
  
  if(length(beta_inits) != p) {
    stop("The argument beta_inits must have length p, where p in the number of columns in X.")
  }
  
  if(length(a) != G) {
    stop("The argument a must have length G.")
  }
  
  if(length(b) != G) {
    stop("The argument b must have length G.")
  }
  
  if(all.equal(grp_idx, sort(grp_idx)) != TRUE) {
    stop("Groups are out of order. Ensure that grp_idx is an ordered sequence
         from 1 to p with no skips. A valid example is 1,1,1,2,2,3,3,3,4,5,5.")
  }
  
  if(any(diff(group_ids) > 1)) {
    stop("Groups skip a number somewhere in grp_idx. Ensure that grp_idx is an ordered sequence
         from 1 to p with no skips. A valid example is 1,1,1,2,2,3,3,3,4,5,5.")
  }
  
  if(method == "fixed") {
    #Fit GIGG regression with fixed hyperparameters
    gigg <- gigg_fixed_gibbs_sampler(X = X, C = C, Y = Y, grp_idx = grp_idx, grp_size = grp_size, grp_size_cs = grp_size_cs,
                                     alpha_inits = alpha_inits, beta_inits = beta_inits, lambda_sq_inits = lambda_sq_inits, gamma_sq_inits = gamma_sq_inits,
                                     eta_inits = rep(1.0, length(unique(grp_idx))), p = a, q = b, tau_sq_init = tau_sq_init, sigma_sq_init = sigma_sq_init,
                                     nu_init = nu_init, n_burn_in = n_burn_in, n_samples = n_samples, n_thin = n_thin, stable_const = stable_const,
                                     verbose = verbose, btrick = btrick, stable_solve = stable_solve)
    
  } else if(method == "mmle") {
    #Fit GIGG regression with MMLE
    gigg <- gigg_mmle_gibbs_sampler(X = X, C = C, Y = Y, grp_idx = grp_idx, grp_size = grp_size, grp_size_cs = grp_size_cs,
                                    alpha_inits = alpha_inits, beta_inits = beta_inits, lambda_sq_inits = lambda_sq_inits, gamma_sq_inits = gamma_sq_inits,
                                    eta_inits = rep(1.0, length(unique(grp_idx))), p_inits = rep(1/n, length(unique(grp_idx))), q_inits = b, tau_sq_init = tau_sq_init, sigma_sq_init = sigma_sq_init,
                                    nu_init = nu_init, n_burn_in = n_burn_in, n_samples = n_samples, n_thin = n_thin, stable_const = stable_const,
                                    verbose = verbose, btrick = btrick, stable_solve = stable_solve)
    
  } else {
    
    stop("Method must be either 'fixed' for GIGG regression with fixed hyperparameters 
         or 'mmle' for GIGG regression with MMLE.")
    
  }
  
  alpha.draws = .post_summary(gigg$alphas)
  beta.draws = .post_summary(gigg$betas)
  sigma_sq.draws = .post_summary(gigg$sigma_sqs, dimension = 2)
  
  return(list(draws = gigg, 
              beta.hat = beta.draws["mean",], 
              beta.lcl.95 = beta.draws["lcl.95.2.5%",],
              beta.ucl.95 = beta.draws["ucl.95.97.5%",], 
              alpha.hat = alpha.draws["mean",], 
              alpha.lcl.95 = alpha.draws["lcl.95.2.5%",],
              alpha.ucl.95 = alpha.draws["ucl.95.97.5%",], 
              sigma_sq.hat = sigma_sq.draws["mean",], 
              sigma_sq.lcl.95 = sigma_sq.draws["lcl.95.2.5%",],
              sigma_sq.ucl.95 = sigma_sq.draws["ucl.95.97.5%",]))
  
}
