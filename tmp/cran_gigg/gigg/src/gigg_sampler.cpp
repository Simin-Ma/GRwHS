#include <RcppArmadillo.h>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]
using namespace Rcpp;


// //' Randomly generate a generalized inverse gaussian random variable.
// //'
// //' Randomly generates one draw from a generalized inverse gaussian distribution.
// //' @param chi A positive double.
// //' @param psi A positive double.
// //' @param lambda A non-negative double.
// //' @return A random draw from the generalized inverse gaussian distribution with parameters chi, psi, and lambda (double).
// // [[Rcpp::export]]
// double rgig_cpp(double chi, double psi, double lambda) {
//   double final_draw = 0;
//   double alpha = sqrt(psi / chi);
//   double beta = sqrt(chi*psi);
//   if ((lambda > 1) || (beta > 1)) {
//     double m = (sqrt(pow(lambda - 1.0, 2) + pow(beta, 2)) + (lambda - 1.0)) / beta;
//     double a = -2.0*(lambda + 1.0) / beta - m;
//     double b = 2.0*(lambda - 1.0)*m / beta - 1.0;
//     double c = m;
//     double p = b - pow(a, 2) / 3.0;
//     double q = 2.0*pow(a, 3) / 27.0 - a*b / 3.0 + c;
//     double phi = acos(-(q / 2.0)*sqrt(-27.0 / pow(p, 3)));
//     double x_minus = sqrt(-(4.0 / 3.0)*p)*cos(phi / 3.0 + (4.0 / 3.0)*M_PI) - a / 3.0;
//     double x_plus = sqrt(-(4.0 / 3.0)*p)*cos(phi / 3.0) - a / 3.0;
//     double v_plus = sqrt(pow(m, lambda - 1.0)*exp(-(beta / 2.0)*(m + 1.0 / m)));
//     double u_minus = (x_minus - m)*sqrt(pow(x_minus, lambda - 1.0)*exp(-(beta / 2.0)*(x_minus + 1.0 / x_minus)));
//     double u_plus = (x_plus - m)*sqrt(pow(x_plus, lambda - 1.0)*exp(-(beta / 2.0)*(x_plus + 1.0 / x_plus)));
//     bool keep_looping = true;
//     double u_draw; double v_draw; double x_draw;
//     while (keep_looping) {
//       u_draw = R::runif(u_minus, u_plus);
//       v_draw = R::runif(0, v_plus);
//       x_draw = u_draw / v_draw + m;
//       if ((pow(v_draw, 2) <= pow(x_draw, lambda - 1.0)*exp(-(beta / 2.0)*(x_draw + 1.0 / x_draw))) && (x_draw > 0)) {
//         final_draw = x_draw;
//         keep_looping = false;
//       }
//     }
//   }
//   else if (lambda >= 0 && lambda <= 1 && beta >= std::min(1.0 / 2.0, (2.0 / 3.0)*sqrt(1.0 - lambda)) && beta <= 1) {
//     double m = beta / ((1.0 - lambda) + sqrt(pow(1.0 - lambda, 2) + pow(beta, 2)));
//     double x_plus = ((1.0 + lambda) + sqrt(pow(1 + lambda, 2) + pow(beta, 2))) / beta;
//     double v_plus = sqrt(pow(m, lambda - 1.0)*exp(-(beta / 2.0)*(m + 1.0 / m)));
//     double u_plus = x_plus*sqrt(pow(x_plus, lambda - 1.0)*exp(-(beta / 2.0)*(x_plus + 1.0 / x_plus)));
//     bool keep_looping = true;
//     double u_draw; double v_draw; double x_draw;
//     while (keep_looping) {
//       u_draw = R::runif(0, u_plus);
//       v_draw = R::runif(0, v_plus);
//       x_draw = u_draw / v_draw;
//       if (pow(v_draw, 2) <= pow(x_draw, lambda - 1.0)*exp(-(beta / 2.0)*(x_draw + 1.0 / x_draw))) {
//         final_draw = x_draw;
//         keep_looping = false;
//       }
//     }
//   }
//   else if (lambda >= 0 && lambda < 1 && beta > 0 && beta <= (2.0 / 3.0)*sqrt(1.0 - lambda)) {
//     double m = beta / ((1.0 - lambda) + sqrt(pow(1.0 - lambda, 2) + pow(beta, 2)));
//     double x0 = beta / (1.0 - lambda);
//     double x_star = std::max(x0, 2.0 / beta);
//     double k1 = pow(m, lambda - 1.0)*exp(-(beta / 2.0)*(m + 1.0 / m));
//     double A1 = k1*x0;
//     double A2; double k2;
//     if (x0 < 2.0 / beta) {
//       k2 = exp(-beta);
//       if (lambda == 0) {
//         A2 = k2*log(2.0 / pow(beta, 2));
//       }
//       else {
//         A2 = k2*(pow(2.0 / beta, lambda) - pow(x0, lambda)) / lambda;
//       }
//     }
//     else {
//       k2 = 0;
//       A2 = 0;
//     }
//     double k3 = pow(x_star, lambda - 1.0);
//     double A3 = 2.0*k3*exp(-x_star*beta / 2.0) / beta;
//     double A = A1 + A2 + A3;
//     bool keep_looping = true;
//     double u_draw; double v_draw; double x_draw; double h;
//     while (keep_looping) {
//       u_draw = R::runif(0, 1);
//       v_draw = R::runif(0, A);
//       if (v_draw <= A1) {
//         x_draw = x0*v_draw / A1;
//         h = k1;
//       }
//       else if (v_draw <= A1 + A2) {
//         v_draw = v_draw - A1;
//         if (lambda == 0) {
//           x_draw = beta*exp(v_draw*exp(beta));
//         }
//         else {
//           x_draw = pow(pow(x0, lambda) + v_draw*lambda / k2, 1.0 / lambda);
//         }
//         h = k2*pow(x_draw, lambda - 1.0);
//       }
//       else {
//         v_draw = v_draw - (A1 + A2);
//         x_draw = -2.0 / beta*log(exp(-x_star*beta / 2.0) - v_draw*beta / (2.0*k3));
//         h = k3*exp(-x_draw*beta / 2.0);
//       }
//       if (u_draw*h <= pow(x_draw, lambda - 1.0)*exp(-(beta / 2.0)*(x_draw + 1.0 / x_draw))) {
//         final_draw = x_draw;
//         keep_looping = false;
//       }
//     }
//   }
//   return final_draw / alpha;
// }

//' Solve function with Cholesky decomposition.
//'
//' An Rcpp function that solves M*U = V.
//' @param M A (M x M) symmetric positive definite matrix.
//' @param V A (M x 1) vector.
//' @return The solution to M*U = V.
// [[Rcpp::export]]
arma::colvec chol_solve(arma::mat& M, arma::colvec& V) {
  arma::mat R = arma::chol(M);
  arma::colvec b_star = arma::solve(trimatl(R.t()), V);
  return(arma::solve(trimatu(R), b_star));
}

//' Randomly generate a generalized inverse gaussian random variable.
//'
//' Randomly generates one draw from a generalized inverse gaussian distribution.
//' @param chi A positive double.
//' @param psi A positive double.
//' @param lambda A non-negative double.
//' @return A random draw from the generalized inverse gaussian distribution with parameters chi, psi, and lambda (double).
// [[Rcpp::export]]
double rgig_cpp(double chi, double psi, double lambda) {
  double final_draw = 0;
  double alpha = sqrt(psi / chi);
  double beta = sqrt(chi*psi);
  if ((lambda > 1) || (beta > 1)) {
    double m = (sqrt(pow(lambda - 1.0, 2) + pow(beta, 2)) + (lambda - 1.0)) / beta;
    double a = -2.0*(lambda + 1.0) / beta - m;
    double b = 2.0*(lambda - 1.0)*m / beta - 1.0;
    double c = m;
    double p = b - pow(a, 2) / 3.0;
    double q = 2.0*pow(a, 3) / 27.0 - a*b / 3.0 + c;
    double phi = acos(-(q / 2.0)*sqrt(-27.0 / pow(p, 3)));
    double x_minus = sqrt(-(4.0 / 3.0)*p)*cos(phi / 3.0 + (4.0 / 3.0)*M_PI) - a / 3.0;
    double x_plus = sqrt(-(4.0 / 3.0)*p)*cos(phi / 3.0) - a / 3.0;
    double v_plus = sqrt(pow(m, lambda - 1.0)*exp(-(beta / 2.0)*(m + 1.0 / m)));
    double u_minus = (x_minus - m)*sqrt(pow(x_minus, lambda - 1.0)*exp(-(beta / 2.0)*(x_minus + 1.0 / x_minus)));
    double u_plus = (x_plus - m)*sqrt(pow(x_plus, lambda - 1.0)*exp(-(beta / 2.0)*(x_plus + 1.0 / x_plus)));
    bool keep_looping = true;
    double u_draw; double v_draw; double x_draw;
    while (keep_looping) {
      u_draw = R::runif(u_minus, u_plus);
      v_draw = R::runif(0, v_plus);
      x_draw = u_draw / v_draw + m;
      if ((pow(v_draw, 2) <= pow(x_draw, lambda - 1.0)*exp(-(beta / 2.0)*(x_draw + 1.0 / x_draw))) && (x_draw > 0)) {
        final_draw = x_draw;
        keep_looping = false;
      }
    }
  }
  else if (lambda >= 0 && lambda <= 1 && beta >= std::min(1.0 / 2.0, (2.0 / 3.0)*sqrt(1.0 - lambda)) && beta <= 1) {
    double m = beta / ((1.0 - lambda) + sqrt(pow(1.0 - lambda, 2) + pow(beta, 2)));
    double x_plus = ((1.0 + lambda) + sqrt(pow(1 + lambda, 2) + pow(beta, 2))) / beta;
    double v_plus = sqrt(pow(m, lambda - 1.0)*exp(-(beta / 2.0)*(m + 1.0 / m)));
    double u_plus = x_plus*sqrt(pow(x_plus, lambda - 1.0)*exp(-(beta / 2.0)*(x_plus + 1.0 / x_plus)));
    bool keep_looping = true;
    double u_draw; double v_draw; double x_draw;
    while (keep_looping) {
      u_draw = R::runif(0, u_plus);
      v_draw = R::runif(0, v_plus);
      x_draw = u_draw / v_draw;
      if (pow(v_draw, 2) <= pow(x_draw, lambda - 1.0)*exp(-(beta / 2.0)*(x_draw + 1.0 / x_draw))) {
        final_draw = x_draw;
        keep_looping = false;
      }
    }
  }
  else if (lambda >= 0 && lambda < 1 && beta > 0 && beta <= (2.0 / 3.0)*sqrt(1.0 - lambda)) {
    double m = beta / ((1.0 - lambda) + sqrt(pow(1.0 - lambda, 2) + pow(beta, 2)));
    double x0 = beta / (1.0 - lambda);
    double x_star = std::max(x0, 2.0 / beta);
    double k1 = pow(m, lambda - 1.0)*exp(-(beta / 2.0)*(m + 1.0 / m));
    double A1 = k1*x0;
    double A2; double k2;
    if (x0 < 2.0 / beta) {
      k2 = exp(-beta);
      if (lambda == 0) {
        A2 = k2*log(2.0 / pow(beta, 2));
      }
      else {
        A2 = k2*(pow(2.0 / beta, lambda) - pow(x0, lambda)) / lambda;
      }
    }
    else {
      k2 = 0;
      A2 = 0;
    }
    double k3 = pow(x_star, lambda - 1.0);
    double A3 = 2.0*k3*exp(-x_star*beta / 2.0) / beta;
    double A = A1 + A2 + A3;
    bool keep_looping = true;
    double u_draw; double v_draw; double x_draw; double h;
    while (keep_looping) {
      u_draw = R::runif(0, 1);
      v_draw = R::runif(0, A);
      if (v_draw <= A1) {
        x_draw = x0*v_draw / A1;
        h = k1;
      }
      else if (v_draw <= A1 + A2) {
        v_draw = v_draw - A1;
        if (lambda == 0) {
          x_draw = beta*exp(v_draw*exp(beta));
        }
        else {
          x_draw = pow(pow(x0, lambda) + v_draw*lambda / k2, 1.0 / lambda);
        }
        h = k2*pow(x_draw, lambda - 1.0);
      }
      else {
        v_draw = v_draw - (A1 + A2);
        x_draw = -2.0 / beta*log(exp(-x_star*beta / 2.0) - v_draw*beta / (2.0*k3));
        h = k3*exp(-x_draw*beta / 2.0);
      }
      if (u_draw*h <= pow(x_draw, lambda - 1.0)*exp(-(beta / 2.0)*(x_draw + 1.0 / x_draw))) {
        final_draw = x_draw;
        keep_looping = false;
      }
    }
  }
  return final_draw / alpha;
}

//' Iterative one rank update for matrix inverse.
//'
//' An Rcpp function that computes the matrix inverse of XtX + D_pos.
//' @param XtX_inv A precomputed (M x M) matrix inverse.
//' @param D_pos A (M x 1) vector of the square root of the diagonal entries in the D matrix.
//' @param vec_draw A (M x 1) vector drawn from a multivariate normal distribution.
//' @return The solution to (XtX + D)*U = vec_draw.
// [[Rcpp::export]]
arma::colvec quick_solve(arma::mat& XtX_inv, arma::colvec& D_pos, arma::colvec& vec_draw) {
  int p = XtX_inv.n_cols;
  arma::mat alpha_mat = arma::zeros(p, p);
  arma::colvec store_k_vec(p);
  arma::mat beta_mat(p, p);
  beta_mat.col(0) = XtX_inv * vec_draw;

  //Calculate alphas and betas
  for (int j = 0; j < p; ++j) {
	  alpha_mat.col(j) = D_pos[j] * XtX_inv.col(j);
  }

  for (int k = 0; k < p - 1; ++k) {
	  store_k_vec = (D_pos[k] / (1.0 + D_pos[k] * alpha_mat(k, k))) * alpha_mat.col(k);
	  for (int j = p - k - 1; j > 0; --j) {
		  alpha_mat.col(j + k) = alpha_mat.col(j + k) - alpha_mat(k, j + k) * store_k_vec;
	  }
	  beta_mat.col(k + 1) = beta_mat.col(k) - beta_mat(k, k) * store_k_vec;
  }

  return(beta_mat.col(p-1) - (D_pos[p-1] * beta_mat(p-1, p-1) / (1.0 + D_pos[p-1] * alpha_mat(p-1, p-1))) * alpha_mat.col(p-1));
}

// //' Solve function with Cholesky decomposition.
// //'
// //' An Rcpp function that solves M*U = V.
// //' @param M A (M x M) symmetric positive definite matrix.
// //' @param V A (M x 1) vector.
// //' @return The solution to M*U = V.
// // [[Rcpp::export]]
// arma::colvec chol_solve(arma::mat& M, arma::colvec& V) {
//   arma::mat R = arma::chol(M);
//   arma::colvec b_star = arma::solve(trimatl(R.t()), V);
//   return(arma::solve(trimatu(R), b_star));
// }


//' Gibbs sampler for GIGG regression with fixed hyperparameters.
//'
//' An Rcpp function that implements a Gibbs sampler for GIGG regression with fixed hyperparameters.
//' @param X A (n x M) matrix of covariates that we want to apply GIGG shrinkage on.
//' @param C A (n x K) matrix of covariates that we want to apply no shrinkage on (typically intercept + adjustment covariates).
//' @param Y A (n x 1) column vector of responses.
//' @param grp_idx A (1 x M) row vector indicating which group of the J groups the M covariates in X belong to.
//' @param grp_size A (1 x J) row vector indicating the number of covariates in each group.
//' @param grp_size_cs A (1 x J) row vector that is the cumulative sum of grp_size (indicating the indicies where each group ends).
//' @param alpha_inits A (K x 1) column vector containing initial values for the regression coefficients corresponding to C.
//' @param beta_inits A (M x 1) column vector containing initial values for the regression coefficients corresponding to X.
//' @param lambda_sq_inits A (M x 1) column vector containing initial values for the local shrinkage parameters.
//' @param gamma_sq_inits A (J x 1) column vector containing initial values for the group shrinkage parameters.
//' @param eta_inits A (J x 1) column vector containing initial values for the mixing parameters.
//' @param p A (J x 1) column vector of shape parameter for the prior on the group shrinkage parameters.
//' @param q A (J x 1) column vector of shape parameter for the prior on the individual shrinkage parameters.
//' @param tau_sq_init Initial value for the global shrinkage parameter (double).
//' @param sigma_sq_init Initial value for the residual variance (double).
//' @param nu_init Initial value for the augmentation variable (double).
//' @param n_burn_in The number of burn-in samples (integer).
//' @param n_samples The number of posterior draws (integer).
//' @param n_thin The thinning interval (integer).
//' @param stable_const Parameter that controls numerical stability of the algorithm (double).
//' @param verbose Boolean value which indicates whether or not to print the progress of the Gibbs sampler.
//' @param btrick Boolean value which indicates whether or not to use the computational trick in Bhattacharya et al. (2016). Only recommended if number of covariates is much larger than the number of observations.
//' @param stable_solve default to FALSE
//' @return A list containing the posterior draws of (1) the regression coefficients (alphas and betas) (2) the individual shrinkage parameters (lambda_sqs) (3) the group shrinkage parameters (gamma_sqs) (4) the global shrinkage parameter (tau_sqs) and (5) the residual error variance (sigma_sqs). The list also contains details regarding the dataset (X, C, Y, grp_idx) and Gibbs sampler details (n_burn_in, n_samples, and n_thin).
// [[Rcpp::export]]
List gigg_fixed_gibbs_sampler(arma::mat& X, arma::mat& C, arma::colvec& Y, arma::rowvec& grp_idx, arma::rowvec& grp_size, arma::rowvec& grp_size_cs,
                              arma::colvec& alpha_inits, arma::colvec& beta_inits, arma::colvec& lambda_sq_inits, arma::colvec& gamma_sq_inits, arma::colvec& eta_inits,
                              arma::colvec& p, arma::colvec& q, double tau_sq_init = 1, double sigma_sq_init = 1,
                              double nu_init = 1, int n_burn_in = 500, int n_samples = 1000, int n_thin = 1, double stable_const = 1e-07, bool verbose = true, bool btrick = false, bool stable_solve = false) {
  
  //Pre-compute and store useful quantities
  int n = X.n_rows;
  int K = C.n_cols;
  int J = eta_inits.n_elem;
  int M = X.n_cols;
  arma::mat tX = X.t();
  arma::mat tC = C.t();
  arma::mat XtX = tX * X;
  arma::mat CtCinv = inv(tC * C);
  arma::mat CtCinvtC = CtCinv*tC;
  arma::mat alpha_term1 = CtCinvtC * Y;
  arma::mat alpha_term2 = CtCinvtC * X;
  
  //Initialize
  arma::colvec alpha = alpha_inits;
  arma::colvec beta = beta_inits;
  arma::colvec lambda_sq = lambda_sq_inits;
  arma::colvec gamma_sq = gamma_sq_inits;
  arma::colvec eta = eta_inits;
  double tau_sq = tau_sq_init;
  double sigma_sq = sigma_sq_init;
  double nu = nu_init;
  
  //Store Gibbs sampler output
  //arma::mat alpha_store = arma::zeros(n_samples, K);
  //arma::mat beta_store = arma::zeros(n_samples, M);
  //arma::mat lambda_store = arma::zeros(n_samples, M);
  //arma::mat gamma_store = arma::zeros(n_samples, J);
  //arma::colvec tau_store = arma::zeros(n_samples);
  //arma::colvec sigma_store = arma::zeros(n_samples);
  //arma::mat eta_store = arma::zeros(n_samples, J);
  //arma::colvec nu_store = arma::zeros(n_samples);
  
  arma::mat alpha_store = arma::zeros(K, n_samples);
  arma::mat beta_store = arma::zeros(M, n_samples);
  arma::mat lambda_store = arma::zeros(M, n_samples);
  arma::mat gamma_store = arma::zeros(J, n_samples);
  arma::colvec tau_store = arma::zeros(n_samples);
  arma::colvec sigma_store = arma::zeros(n_samples);
  arma::mat eta_store = arma::zeros(J, n_samples);
  arma::colvec nu_store = arma::zeros(n_samples);
  
  //Calculate constants for updating sigma and tau
  double tau_shape_const = ((double)M + 1.0) / 2.0;
  double tau_rate_const = 0;
  double sigma_shape_const = ((double)n + 1.0) / 2.0;
  
  //Prevent repetative initializations by initializing here
  //arma::mat local_param_inv = arma::zeros(M, M);
  //arma::colvec local_param_expand_inv = arma::zeros(M);
  
  //if (btrick == true) {
  //	arma::mat gl_param_expand = arma::zeros(M, M);
  //	arma::colvec gl_param_expand_diag = arma::zeros(M);
  //	arma::colvec beta_tmp_u = arma::zeros(M);
  //	arma::colvec beta_tmp_delta = arma::zeros(n);
  //	arma::colvec beta_tmp_zeros_M = arma::zeros(M);
  //	arma::colvec beta_tmp_zeros_n = arma::zeros(n);
  //	arma::mat beta_tmp_zeros_identity_n = arma::eye(n, n);
  //	arma::colvec beta_tmp_v = arma::zeros(n);
  //	arma::colvec beta_tmp_w = arma::zeros(n);
  //}
  //else {
  //	arma::mat beta_tmp = arma::zeros(M, M);
  //}
  
  arma::mat gl_param_expand = arma::zeros(M, M);
  arma::colvec gl_param_expand_diag = arma::zeros(M);
  arma::colvec gl_param_expand_diag_inv = arma::zeros(M);
  
  arma::colvec beta_tmp_u = arma::zeros(M);
  arma::colvec beta_tmp_delta = arma::zeros(n);
  arma::colvec beta_tmp_zeros_M = arma::zeros(M);
  arma::colvec beta_tmp_zeros_n = arma::zeros(n);
  arma::mat beta_tmp_zeros_identity_n = arma::eye(n, n);
  arma::mat beta_tmp_matrix_theta = arma::zeros(M, n);
  arma::colvec beta_tmp_v = arma::zeros(n);
  arma::colvec beta_tmp_w = arma::zeros(n);
  
  arma::mat beta_tmp = arma::zeros(M, M);
  
  double stable_psi = 0;
  double sum_inv_lambda_sq = 0;
  double sum_log_lambda_sq = 0;
  
  int cnt = 0;
  while (cnt < n_burn_in) {
    
    //Draw alpha
    alpha = arma::mvnrnd(alpha_term1 - alpha_term2 * beta, sigma_sq*CtCinv);
    
    //Draw beta
    if (btrick == true) {
      for (int g = 0; g < M; ++g) {
        gl_param_expand_diag[g] = tau_sq * gamma_sq[grp_idx[g] - 1] * lambda_sq[g];
        gl_param_expand_diag_inv[g] = 1.0 / gl_param_expand_diag[g];
      }
      gl_param_expand.diag() = gl_param_expand_diag;
      //local_param_inv.diag() = local_param_expand_inv;
      
      for (int j = 0; j < M; ++j) {
        beta_tmp_matrix_theta.row(j) = gl_param_expand_diag[j] * tX.row(j);
      }
      
      beta_tmp_u = arma::mvnrnd(beta_tmp_zeros_M, gl_param_expand);
      beta_tmp_delta = arma::mvnrnd(beta_tmp_zeros_n, beta_tmp_zeros_identity_n);
      beta_tmp_v = (1.0 / sqrt(sigma_sq)) * (X * beta_tmp_u) + beta_tmp_delta;
      beta_tmp_w = arma::solve(((1.0 / sigma_sq) * X * beta_tmp_matrix_theta) + beta_tmp_zeros_identity_n, ((1.0 / sqrt(sigma_sq)) * (Y - C * alpha)) - beta_tmp_v);
      beta = beta_tmp_u + (1 / sqrt(sigma_sq)) * (beta_tmp_matrix_theta * beta_tmp_w);
    }
    else {
      //for (int g = 0; g < M; ++g) {
      //	local_param_expand_inv[g] = 1.0 / (gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
      //}
      //local_param_inv.diag() = local_param_expand_inv;
      //beta_tmp = inv((1.0 / sigma_sq) * XtX + (1.0 / tau_sq) * local_param_inv);
      ////beta_tmp = (1.0 / sigma_sq) * XtX + (1.0 / tau_sq) * local_param_inv;
      ////beta_tmp = inv(arma::chol(beta_tmp));
      ////beta_tmp = beta_tmp * beta_tmp.t();
      //beta = arma::mvnrnd((1.0 / sigma_sq) * beta_tmp * tX * (Y - C * alpha), beta_tmp);
      
      for (int g = 0; g < M; ++g) {
        gl_param_expand_diag_inv[g] = 1.0 / (tau_sq * gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
      }
      //local_param_inv.diag() = local_param_expand_inv;
      beta_tmp = (1.0 / sigma_sq) * XtX;
      beta_tmp.diag() = beta_tmp.diag() + gl_param_expand_diag_inv;
      beta = arma::mvnrnd((1.0 / sigma_sq) * tX * (Y - C * alpha), beta_tmp);
      if (stable_solve) {
        beta = chol_solve(beta_tmp, beta);
      }
      else {
        beta = arma::solve(beta_tmp, beta);
      }
    }
    
    
    //Draw tau^2
    for (int j = 0; j < M; ++j) {
      tau_rate_const += beta[j] * gl_param_expand_diag_inv[j] * beta[j];
    }
    tau_sq = 1.0 / R::rgamma(tau_shape_const, 1.0 / (tau_sq * tau_rate_const / 2.0 + 1.0 / nu));
    tau_rate_const = 0;
    
    //Draw sigma^2
    sigma_sq = 1.0 / R::rgamma(sigma_shape_const, 1.0 / (((Y - C * alpha - X * beta).t()*(Y - C * alpha - X * beta)) / 2.0 + 1.0 / nu).eval()(0, 0));
    
    for (int j = 0; j < J; ++j) {
      //Draw gamma^2
      if (j != 0) {
        stable_psi = 0;
        for (int l = grp_size_cs[j - 1]; l < grp_size_cs[j]; ++l) {
          stable_psi += pow(beta[l], 2) / lambda_sq[l];
        }
        stable_psi *= (1.0 / tau_sq);
        stable_psi = std::max(stable_psi, stable_const);
        //gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
        if (((double)grp_size[j]) / 2.0 < p[j]) {
          //gamma_sq[j] = rgig_cpp(2.0 * eta[j], stable_psi, p[j] - ((double)grp_size[j]) / 2.0);
          gamma_sq[j] = rgig_cpp(stable_psi, 2.0 * eta[j], p[j] - ((double)grp_size[j]) / 2.0);
        }
        else {
          //gamma_sq[j] = 1.0 / rgig_cpp(stable_psi, 2.0 * eta[j], ((double)grp_size[j]) / 2.0 - p[j]);
          gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
        }
      }
      else {
        stable_psi = 0;
        for (int l = 0; l < grp_size_cs[j]; ++l) {
          stable_psi += pow(beta[l], 2) / lambda_sq[l];
        }
        stable_psi *= (1.0 / tau_sq);
        stable_psi = std::max(stable_psi, stable_const);
        //gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
        if (((double)grp_size[j]) / 2.0 < p[j]) {
          //gamma_sq[j] = rgig_cpp(2.0 * eta[j], stable_psi, p[j] - ((double)grp_size[j]) / 2.0);
          gamma_sq[j] = rgig_cpp(stable_psi, 2.0 * eta[j], p[j] - ((double)grp_size[j]) / 2.0);
        }
        else {
          //gamma_sq[j] = 1.0 / rgig_cpp(stable_psi, 2.0 * eta[j], ((double)grp_size[j]) / 2.0 - p[j]);
          gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
        }
      }
      
      //Draw lambda^2
      sum_inv_lambda_sq = 0.0;
      sum_log_lambda_sq = 0.0;
      for (int i = 0; i < grp_size[j]; ++i) {
        if (j != 0) {
          lambda_sq[grp_size_cs[j - 1] + i] = 1.0 / R::rgamma(q[j] + 0.5, 1.0 / (eta[j] + pow(beta[grp_size_cs[j - 1] + i], 2) / (2.0 * tau_sq * gamma_sq[j])));
          sum_inv_lambda_sq += (1.0 / lambda_sq[grp_size_cs[j - 1] + i]);
          sum_log_lambda_sq += log(lambda_sq[grp_size_cs[j - 1] + i]);
        }
        else {
          lambda_sq[i] = 1.0 / R::rgamma(q[j] + 0.5, 1.0 / (eta[j] + pow(beta[i], 2) / (2.0 * tau_sq * gamma_sq[j])));
          sum_inv_lambda_sq += (1.0 / lambda_sq[i]);
          sum_log_lambda_sq += log(lambda_sq[i]);
        }
      }
      
      //Draw eta
      //eta[j] = R::rgamma(p[j] + q[j] * (double)grp_size[j], 1.0 / (gamma_sq[j] + sum_inv_lambda_sq));
      eta[j] = 1.0;
    }
    
    //Draw nu
    nu = 1.0 / R::rgamma(1.0, 1.0 / ((1.0 / tau_sq) + (1.0 / sigma_sq)));
    
    ++cnt;
    if (cnt % 500 == 0 && verbose) {
      Rcpp::Rcout << cnt << " Burn-in Draws" << std::endl;
    }
  }
  
  if (verbose) {
    Rcpp::Rcout << "Burn-in Iterations Complete" << std::endl;
  }
  
  cnt = 0;
  int total_saved = 0;
  while (total_saved < n_samples) {
    
    //Draw alpha
    alpha = arma::mvnrnd(alpha_term1 - alpha_term2 * beta, sigma_sq*CtCinv);
    
    //Draw beta
    if (btrick == true) {
      for (int g = 0; g < M; ++g) {
        gl_param_expand_diag[g] = tau_sq * gamma_sq[grp_idx[g] - 1] * lambda_sq[g];
        gl_param_expand_diag_inv[g] = 1.0 / gl_param_expand_diag[g];
      }
      gl_param_expand.diag() = gl_param_expand_diag;
      //local_param_inv.diag() = local_param_expand_inv;
      
      for (int j = 0; j < M; ++j) {
        beta_tmp_matrix_theta.row(j) = gl_param_expand_diag[j] * tX.row(j);
      }
      
      beta_tmp_u = arma::mvnrnd(beta_tmp_zeros_M, gl_param_expand);
      beta_tmp_delta = arma::mvnrnd(beta_tmp_zeros_n, beta_tmp_zeros_identity_n);
      beta_tmp_v = (1.0 / sqrt(sigma_sq)) * (X * beta_tmp_u) + beta_tmp_delta;
      beta_tmp_w = arma::solve(((1.0 / sigma_sq) * X * beta_tmp_matrix_theta) + beta_tmp_zeros_identity_n, ((1.0 / sqrt(sigma_sq)) * (Y - C * alpha)) - beta_tmp_v);
      beta = beta_tmp_u + (1 / sqrt(sigma_sq)) * (beta_tmp_matrix_theta * beta_tmp_w);
    }
    else {
      //for (int g = 0; g < M; ++g) {
      //	local_param_expand_inv[g] = 1.0 / (gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
      //}
      //local_param_inv.diag() = local_param_expand_inv;
      //beta_tmp = inv((1.0 / sigma_sq) * XtX + (1.0 / tau_sq) * local_param_inv);
      ////beta_tmp = (1.0 / sigma_sq) * XtX + (1.0 / tau_sq) * local_param_inv;
      ////beta_tmp = inv(arma::chol(beta_tmp));
      ////beta_tmp = beta_tmp * beta_tmp.t();
      //beta = arma::mvnrnd((1.0 / sigma_sq) * beta_tmp * tX * (Y - C * alpha), beta_tmp);
      
      for (int g = 0; g < M; ++g) {
        gl_param_expand_diag_inv[g] = 1.0 / (tau_sq * gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
      }
      //local_param_inv.diag() = local_param_expand_inv;
      beta_tmp = (1.0 / sigma_sq) * XtX;
      beta_tmp.diag() = beta_tmp.diag() + gl_param_expand_diag_inv;
      beta = arma::mvnrnd((1.0 / sigma_sq) * tX * (Y - C * alpha), beta_tmp);
      if (stable_solve) {
        beta = chol_solve(beta_tmp, beta);
      }
      else {
        beta = arma::solve(beta_tmp, beta);
      }
      
    }
    
    
    //Draw tau^2
    for (int j = 0; j < M; ++j) {
      tau_rate_const += beta[j] * gl_param_expand_diag_inv[j] * beta[j];
    }
    tau_sq = 1.0 / R::rgamma(tau_shape_const, 1.0 / (tau_sq * tau_rate_const / 2.0 + 1.0 / nu));
    tau_rate_const = 0;
    
    //Draw sigma^2
    sigma_sq = 1.0 / R::rgamma(sigma_shape_const, 1.0 / (((Y - C * alpha - X * beta).t()*(Y - C * alpha - X * beta)) / 2.0 + 1.0 / nu).eval()(0, 0));
    
    for (int j = 0; j < J; ++j) {
      //Draw gamma^2
      if (j != 0) {
        stable_psi = 0;
        for (int l = grp_size_cs[j - 1]; l < grp_size_cs[j]; ++l) {
          stable_psi += pow(beta[l], 2) / lambda_sq[l];
        }
        stable_psi *= (1.0 / tau_sq);
        stable_psi = std::max(stable_psi, stable_const);
        //gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
        if (((double)grp_size[j]) / 2.0 < p[j]) {
          //gamma_sq[j] = rgig_cpp(2.0 * eta[j], stable_psi, p[j] - ((double)grp_size[j]) / 2.0);
          gamma_sq[j] = rgig_cpp(stable_psi, 2.0 * eta[j], p[j] - ((double)grp_size[j]) / 2.0);
        }
        else {
          //gamma_sq[j] = 1.0 / rgig_cpp(stable_psi, 2.0 * eta[j], ((double)grp_size[j]) / 2.0 - p[j]);
          gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
        }
      }
      else {
        stable_psi = 0;
        for (int l = 0; l < grp_size_cs[j]; ++l) {
          stable_psi += pow(beta[l], 2) / lambda_sq[l];
        }
        stable_psi *= (1.0 / tau_sq);
        stable_psi = std::max(stable_psi, stable_const);
        //gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
        if (((double)grp_size[j]) / 2.0 < p[j]) {
          //gamma_sq[j] = rgig_cpp(2.0 * eta[j], stable_psi, p[j] - ((double)grp_size[j]) / 2.0);
          gamma_sq[j] = rgig_cpp(stable_psi, 2.0 * eta[j], p[j] - ((double)grp_size[j]) / 2.0);
        }
        else {
          //gamma_sq[j] = 1.0 / rgig_cpp(stable_psi, 2.0 * eta[j], ((double)grp_size[j]) / 2.0 - p[j]);
          gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
        }
      }
      
      //Draw lambda^2
      sum_inv_lambda_sq = 0.0;
      sum_log_lambda_sq = 0.0;
      for (int i = 0; i < grp_size[j]; ++i) {
        if (j != 0) {
          lambda_sq[grp_size_cs[j - 1] + i] = 1.0 / R::rgamma(q[j] + 0.5, 1.0 / (eta[j] + pow(beta[grp_size_cs[j - 1] + i], 2) / (2.0 * tau_sq * gamma_sq[j])));
          sum_inv_lambda_sq += (1.0 / lambda_sq[grp_size_cs[j - 1] + i]);
          sum_log_lambda_sq += log(lambda_sq[grp_size_cs[j - 1] + i]);
        }
        else {
          lambda_sq[i] = 1.0 / R::rgamma(q[j] + 0.5, 1.0 / (eta[j] + pow(beta[i], 2) / (2.0 * tau_sq * gamma_sq[j])));
          sum_inv_lambda_sq += (1.0 / lambda_sq[i]);
          sum_log_lambda_sq += log(lambda_sq[i]);
        }
      }
      
      //Draw eta
      //eta[j] = R::rgamma(p[j] + q[j] * (double)grp_size[j], 1.0 / (gamma_sq[j] + sum_inv_lambda_sq));
      eta[j] = 1.0;
    }
    
    //Draw nu
    nu = 1.0 / R::rgamma(1.0, 1.0 / ((1.0 / tau_sq) + (1.0 / sigma_sq)));
    
    //Save output
    if (cnt % n_thin == 0) {
      //alpha_store.row(total_saved) = alpha.t();
      //beta_store.row(total_saved) = beta.t();
      //lambda_store.row(total_saved) = lambda_sq.t();
      //gamma_store.row(total_saved) = gamma_sq.t();
      tau_store[total_saved] = tau_sq;
      sigma_store[total_saved] = sigma_sq;
      //eta_store.row(total_saved) = eta.t();
      nu_store[total_saved] = nu;
      
      alpha_store.col(total_saved) = alpha;
      beta_store.col(total_saved) = beta;
      lambda_store.col(total_saved) = lambda_sq;
      gamma_store.col(total_saved) = gamma_sq;
      eta_store.col(total_saved) = eta;
      
      ++total_saved;
      if (total_saved % 500 == 0 && verbose) {
        Rcpp::Rcout << total_saved << " Samples Drawn" << std::endl;
      }
    }
    
    ++cnt;
  }
  
  return(List::create(Named("alphas") = alpha_store, Named("betas") = beta_store, Named("lambda_sqs") = lambda_store,
                            Named("gamma_sqs") = gamma_store, Named("tau_sqs") = tau_store, Named("sigma_sqs") = sigma_store,
                            Named("a") = p, Named("b") = q, Named("X") = X, Named("C") = C, Named("Y") = Y,
                            Named("grp_idx") = grp_idx, Named("n_burn_in") = n_burn_in, Named("n_samples") = n_samples, Named("n_thin") = n_thin));
}


//' Inverse digamma function.
//'
//' Evaluate the inverse digamma function.
//' @param y value to evaluate the inverse digamma function at.
//' @param precision default = 1e-08.
//' @return Numeric inverse digamma value.
// [[Rcpp::export]]
double digamma_inv(double y, double precision = 1e-08) {
  double x_old = 1.0;
  if (y >= -2.22) {
    x_old = exp(y) + 0.5;
  }
  else {
    x_old = -1.0/(y- boost::math::digamma(1.0));
  }
  double x_new = x_old - (boost::math::digamma(x_old)-y)/boost::math::trigamma(x_old);
  if (abs(x_new - x_old) >= precision) {
    while (abs(x_new - x_old) >= precision) {
      x_old = x_new;
      x_new = x_old - (boost::math::digamma(x_old) - y) / boost::math::trigamma(x_old);
    }
  }
  return x_new;
}

//' Gibbs sampler for GIGG regression with hyperparameters estimated via MMLE.
//'
//' An Rcpp function that implements a Gibbs sampler for GIGG regression with hyperparameters estimated via MMLE.
//' @param X A (n x M) matrix of covariates that we want to apply GIGG shrinkage on.
//' @param C A (n x K) matrix of covariates that we want to apply no shrinkage on (typically intercept + adjustment covariates).
//' @param Y A (n x 1) column vector of responses.
//' @param grp_idx A (1 x M) row vector indicating which group of the J groups the M covariates in X belong to.
//' @param grp_size A (1 x J) row vector indicating the number of covariates in each group.
//' @param grp_size_cs A (1 x J) row vector that is the cumulative sum of grp_size (indicating the indicies where each group ends).
//' @param alpha_inits A (K x 1) column vector containing initial values for the regression coefficients corresponding to C.
//' @param beta_inits A (M x 1) column vector containing initial values for the regression coefficients corresponding to X.
//' @param lambda_sq_inits A (M x 1) column vector containing initial values for the local shrinkage parameters.
//' @param gamma_sq_inits A (J x 1) column vector containing initial values for the group shrinkage parameters.
//' @param eta_inits A (J x 1) column vector containing initial values for the mixing parameters.
//' @param p_inits A (J x 1) column vector of initial shape parameter for the prior on the group shrinkage parameters.
//' @param q_inits A (J x 1) column vector of inital shape parameter for the prior on the individual shrinkage parameters.
//' @param tau_sq_init Initial value for the global shrinkage parameter (double).
//' @param sigma_sq_init Initial value for the residual variance (double).
//' @param nu_init Initial value for the augmentation variable (double).
//' @param n_burn_in The number of burn-in samples (integer).
//' @param n_samples The number of posterior draws (integer).
//' @param n_thin The thinning interval (integer).
//' @param stable_const Parameter that controls numerical stability of the algorithm (double).
//' @param verbose Boolean value which indicates whether or not to print the progress of the Gibbs sampler.
//' @param btrick Boolean value which indicates whether or not to use the computational trick in Bhattacharya et al. (2016). Only recommended if number of covariates is much larger than the number of observations.
//' @param stable_solve default to FALSE
//' @return A list containing the posterior draws of (1) the regression coefficients (alphas and betas) (2) the individual shrinkage parameters (lambda_sqs) (3) the group shrinkage parameters (gamma_sqs) (4) the global shrinkage parameter (tau_sqs) and (5) the residual error variance (sigma_sqs). The list also contains details regarding the dataset (X, C, Y, grp_idx) and Gibbs sampler details (n_burn_in, n_samples, and n_thin).
// [[Rcpp::export]]
List gigg_mmle_gibbs_sampler(arma::mat& X, arma::mat& C, arma::colvec& Y, arma::rowvec& grp_idx, arma::rowvec& grp_size, arma::rowvec& grp_size_cs,
                             arma::colvec& alpha_inits, arma::colvec& beta_inits, arma::colvec& lambda_sq_inits, arma::colvec& gamma_sq_inits, arma::colvec& eta_inits,
                             arma::colvec& p_inits, arma::colvec& q_inits, double tau_sq_init = 1, double sigma_sq_init = 1,
                             double nu_init = 1, int n_burn_in = 500, int n_samples = 1000, int n_thin = 1, double stable_const = 1e-07, bool verbose = true, bool btrick = false, bool stable_solve = false) {
  
  //Pre-compute and store useful quantities
  int n = X.n_rows;
  int K = C.n_cols;
  int J = eta_inits.n_elem;
  int M = X.n_cols;
  arma::mat tX = X.t();
  arma::mat tC = C.t();
  arma::mat XtX = tX * X;
  arma::mat CtCinv = inv(tC * C);
  arma::mat CtCinvtC = CtCinv*tC;
  arma::mat alpha_term1 = CtCinvtC * Y;
  arma::mat alpha_term2 = CtCinvtC * X;
  
  //Initialize
  arma::colvec alpha = alpha_inits;
  arma::colvec beta = beta_inits;
  arma::colvec lambda_sq = lambda_sq_inits;
  arma::colvec gamma_sq = gamma_sq_inits;
  arma::colvec eta = eta_inits;
  arma::colvec p = p_inits;
  arma::colvec q = q_inits;
  double tau_sq = tau_sq_init;
  double sigma_sq = sigma_sq_init;
  double nu = nu_init;
  
  //Store Gibbs sampler output
  //arma::mat alpha_store = arma::zeros(n_samples, K);
  //arma::mat beta_store = arma::zeros(n_samples, M);
  //arma::mat lambda_store = arma::zeros(n_samples, M);
  //arma::mat gamma_store = arma::zeros(n_samples, J);
  //arma::colvec tau_store = arma::zeros(n_samples);
  //arma::colvec sigma_store = arma::zeros(n_samples);
  //arma::mat eta_store = arma::zeros(n_samples, J);
  //arma::mat p_store = arma::zeros(n_samples, J);
  //arma::mat q_store = arma::zeros(n_samples, J);
  //arma::colvec nu_store = arma::zeros(n_samples);
  
  arma::mat alpha_store = arma::zeros(K, n_samples);
  arma::mat beta_store = arma::zeros(M, n_samples);
  arma::mat lambda_store = arma::zeros(M, n_samples);
  arma::mat gamma_store = arma::zeros(J, n_samples);
  arma::colvec tau_store = arma::zeros(n_samples);
  arma::colvec sigma_store = arma::zeros(n_samples);
  arma::mat eta_store = arma::zeros(J, n_samples);
  //arma::mat p_store = arma::zeros(J, n_samples);
  //arma::mat q_store = arma::zeros(J, n_samples);
  arma::colvec nu_store = arma::zeros(n_samples);
  
  //Empirical Bayes Update Storage
  int mmle_samp_size = 1000;
  //double epsilon = 1e-3;
  double terminate_mmle = 1e-4 * (double)J;
  double overflow_check = 0;
  double delta_mmle = 1.0;
  double log_lambda_mmle_sum = 0.0;
  arma::mat lambda_mmle_store = arma::zeros(mmle_samp_size, M);
  arma::mat gamma_mmle_store = arma::zeros(mmle_samp_size, J);
  arma::mat eta_mmle_store = arma::zeros(mmle_samp_size, J);
  
  //Rcpp::Rcout << terminate_mmle << std::endl;
  
  //Calculate constants for updating sigma and tau
  double tau_shape_const = ((double)M + 1.0) / 2.0;
  double tau_rate_const = 0;
  double sigma_shape_const = ((double)n + 1.0) / 2.0;
  
  //Prevent repetative initializations by initializing here
  arma::mat gl_param_expand = arma::zeros(M, M);
  arma::colvec gl_param_expand_diag = arma::zeros(M);
  arma::colvec gl_param_expand_diag_inv = arma::zeros(M);
  double stable_psi = 0;
  double sum_inv_lambda_sq = 0;
  double sum_log_lambda_sq = 0;
  
  arma::colvec beta_tmp_u = arma::zeros(M);
  arma::colvec beta_tmp_delta = arma::zeros(n);
  arma::colvec beta_tmp_zeros_M = arma::zeros(M);
  arma::colvec beta_tmp_zeros_n = arma::zeros(n);
  arma::mat beta_tmp_zeros_identity_n = arma::eye(n, n);
  arma::mat beta_tmp_matrix_theta = arma::zeros(M, n);
  arma::colvec beta_tmp_v = arma::zeros(n);
  arma::colvec beta_tmp_w = arma::zeros(n);
  
  arma::mat beta_tmp = arma::zeros(M, M);
  
  int cnt = 0;
  while (cnt < n_burn_in) {
    
    //Draw alpha
    alpha = arma::mvnrnd(alpha_term1 - alpha_term2 * beta, sigma_sq*CtCinv);
    
    //Draw beta
    //for (int g = 0; g < M; ++g) {
    //	local_param_expand_inv[g] = 1.0 / (gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
    //}
    //local_param_inv.diag() = local_param_expand_inv;
    //beta_tmp = inv((1.0 / sigma_sq) * XtX + (1.0 / tau_sq) * local_param_inv);
    //beta = arma::mvnrnd((1.0 / sigma_sq) * beta_tmp * tX * (Y - C * alpha), beta_tmp);
    
    //New Stuff
    //if (btrick == true) {
    //	for (int g = 0; g < M; ++g) {
    //		local_param_expand_inv[g] = 1.0 / (gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
    //		gl_param_expand_diag[g] = tau_sq * gamma_sq[grp_idx[g] - 1] * lambda_sq[g];
    //	}
    //	gl_param_expand.diag() = gl_param_expand_diag;
    //	local_param_inv.diag() = local_param_expand_inv;
    //	beta_tmp_u = arma::mvnrnd(beta_tmp_zeros_M, gl_param_expand);
    //	beta_tmp_delta = arma::mvnrnd(beta_tmp_zeros_n, beta_tmp_zeros_identity_n);
    //	beta_tmp_v = (1.0 / sqrt(sigma_sq)) * (X * beta_tmp_u) + beta_tmp_delta;
    //	beta_tmp_w = arma::solve(((1.0 / sigma_sq) * X * gl_param_expand * tX) + beta_tmp_zeros_identity_n, ((1.0 / sqrt(sigma_sq)) * (Y - C * alpha)) - beta_tmp_v);
    //	beta = beta_tmp_u + (1 / sqrt(sigma_sq)) * gl_param_expand * tX * beta_tmp_w;
    //}
    //else {
    //	for (int g = 0; g < M; ++g) {
    //		local_param_expand_inv[g] = 1.0 / (gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
    //	}
    //	local_param_inv.diag() = local_param_expand_inv;
    //	beta_tmp = inv((1.0 / sigma_sq) * XtX + (1.0 / tau_sq) * local_param_inv);
    //	beta = arma::mvnrnd((1.0 / sigma_sq) * beta_tmp * tX * (Y - C * alpha), beta_tmp);
    //}
    //
    
    if (btrick == true) {
      for (int g = 0; g < M; ++g) {
        gl_param_expand_diag[g] = tau_sq * gamma_sq[grp_idx[g] - 1] * lambda_sq[g];
        gl_param_expand_diag_inv[g] = 1.0 / gl_param_expand_diag[g];
      }
      gl_param_expand.diag() = gl_param_expand_diag;
      
      for (int j = 0; j < M; ++j) {
        beta_tmp_matrix_theta.row(j) = gl_param_expand_diag[j] * tX.row(j);
      }
      
      beta_tmp_u = arma::mvnrnd(beta_tmp_zeros_M, gl_param_expand);
      beta_tmp_delta = arma::mvnrnd(beta_tmp_zeros_n, beta_tmp_zeros_identity_n);
      beta_tmp_v = (1.0 / sqrt(sigma_sq)) * (X * beta_tmp_u) + beta_tmp_delta;
      beta_tmp_w = arma::solve(((1.0 / sigma_sq) * X * beta_tmp_matrix_theta) + beta_tmp_zeros_identity_n, ((1.0 / sqrt(sigma_sq)) * (Y - C * alpha)) - beta_tmp_v);
      beta = beta_tmp_u + (1 / sqrt(sigma_sq)) * (beta_tmp_matrix_theta * beta_tmp_w);
    }
    else {
      for (int g = 0; g < M; ++g) {
        gl_param_expand_diag_inv[g] = 1.0 / (tau_sq * gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
      }
      beta_tmp = (1.0 / sigma_sq) * XtX;
      beta_tmp.diag() = beta_tmp.diag() + gl_param_expand_diag_inv;
      beta = arma::mvnrnd((1.0 / sigma_sq) * tX * (Y - C * alpha), beta_tmp);
      if (stable_solve) {
        beta = chol_solve(beta_tmp, beta);
      }
      else {
        beta = arma::solve(beta_tmp, beta);
      }
    }
    
    //Draw tau^2
    //tau_sq = 1.0 / R::rgamma(((double)M + 1.0) / 2.0, 1.0 / ((beta.t() * local_param_inv * beta) / 2.0 + 1.0 / nu).eval()(0, 0));
    for (int j = 0; j < M; ++j) {
      tau_rate_const += beta[j] * gl_param_expand_diag_inv[j] * beta[j];
    }
    tau_sq = 1.0 / R::rgamma(tau_shape_const, 1.0 / (tau_sq * tau_rate_const / 2.0 + 1.0 / nu));
    tau_rate_const = 0;
    
    //Draw sigma^2
    //sigma_sq = 1.0 / R::rgamma(((double)n + 1.0) / 2.0, 1.0 / (((Y - C * alpha - X * beta).t()*(Y - C * alpha - X * beta)) / 2.0 + 1.0 / nu).eval()(0, 0));
    sigma_sq = 1.0 / R::rgamma(sigma_shape_const, 1.0 / (((Y - C * alpha - X * beta).t()*(Y - C * alpha - X * beta)) / 2.0 + 1.0 / nu).eval()(0, 0));
    
    for (int j = 0; j < J; ++j) {
      //Draw gamma^2
      //if (j != 0) {
      //	stable_psi = 0;
      //	for (int l = grp_size_cs[j - 1]; l < grp_size_cs[j]; ++l) {
      //		stable_psi += pow(beta[l], 2) / lambda_sq[l];
      //	}
      //	stable_psi *= (1.0 / tau_sq);
      //	stable_psi = std::max(stable_psi, stable_const);
      //	gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
      //}
      //else {
      //	stable_psi = 0;
      //	for (int l = 0; l < grp_size_cs[j]; ++l) {
      //		stable_psi += pow(beta[l], 2) / lambda_sq[l];
      //	}
      //	stable_psi *= (1.0 / tau_sq);
      //	stable_psi = std::max(stable_psi, stable_const);
      //	gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
      //}
      
      //Draw gamma^2
      if (j != 0) {
        stable_psi = 0;
        for (int l = grp_size_cs[j - 1]; l < grp_size_cs[j]; ++l) {
          stable_psi += pow(beta[l], 2) / lambda_sq[l];
        }
        stable_psi *= (1.0 / tau_sq);
        stable_psi = std::max(stable_psi, stable_const);
        if (((double)grp_size[j]) / 2.0 < p[j]) {
          //gamma_sq[j] = rgig_cpp(2.0 * eta[j], stable_psi, p[j] - ((double)grp_size[j]) / 2.0);
          gamma_sq[j] = rgig_cpp(stable_psi, 2.0 * eta[j], p[j] - ((double)grp_size[j]) / 2.0);
        }
        else {
          //gamma_sq[j] = 1.0 / rgig_cpp(stable_psi, 2.0 * eta[j], ((double)grp_size[j]) / 2.0 - p[j]);
          gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
        }
      }
      else {
        stable_psi = 0;
        for (int l = 0; l < grp_size_cs[j]; ++l) {
          stable_psi += pow(beta[l], 2) / lambda_sq[l];
        }
        stable_psi *= (1.0 / tau_sq);
        stable_psi = std::max(stable_psi, stable_const);
        if (((double)grp_size[j]) / 2.0 < p[j]) {
          //gamma_sq[j] = rgig_cpp(2.0 * eta[j], stable_psi, p[j] - ((double)grp_size[j]) / 2.0);
          gamma_sq[j] = rgig_cpp(stable_psi, 2.0 * eta[j], p[j] - ((double)grp_size[j]) / 2.0);
        }
        else {
          //gamma_sq[j] = 1.0 / rgig_cpp(stable_psi, 2.0 * eta[j], ((double)grp_size[j]) / 2.0 - p[j]);
          gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
        }
      }
      
      //Draw lambda^2
      sum_inv_lambda_sq = 0.0;
      sum_log_lambda_sq = 0.0;
      for (int i = 0; i < grp_size[j]; ++i) {
        if (j != 0) {
          lambda_sq[grp_size_cs[j - 1] + i] = 1.0 / R::rgamma(q[j] + 0.5, 1.0 / (eta[j] + pow(beta[grp_size_cs[j - 1] + i], 2) / (2.0 * tau_sq * gamma_sq[j])));
          sum_inv_lambda_sq += (1.0 / lambda_sq[grp_size_cs[j - 1] + i]);
          sum_log_lambda_sq += log(lambda_sq[grp_size_cs[j - 1] + i]);
        }
        else {
          lambda_sq[i] = 1.0 / R::rgamma(q[j] + 0.5, 1.0 / (eta[j] + pow(beta[i], 2) / (2.0 * tau_sq * gamma_sq[j])));
          sum_inv_lambda_sq += (1.0 / lambda_sq[i]);
          sum_log_lambda_sq += log(lambda_sq[i]);
        }
      }
      
      //Draw eta
      //eta[j] = R::rgamma(p[j] + q[j] * (double)grp_size[j], 1.0 / (gamma_sq[j] + sum_inv_lambda_sq));
      eta[j] = 1.0;
      
    }
    
    //Draw nu
    nu = 1.0 / R::rgamma(1.0, 1.0 / ((1.0 / tau_sq) + (1.0 / sigma_sq)));
    
    //Estimate q
    //eta_mmle_store.row(cnt % mmle_samp_size) = eta.t();
    //gamma_mmle_store.row(cnt % mmle_samp_size) = gamma_sq.t();
    //lambda_mmle_store.row(cnt % mmle_samp_size) = lambda_sq.t();
    //if ((cnt + 1) % mmle_samp_size == 0) {
    //	for (int j = 0; j < J; ++j) {
    //		for (int i = 0; i < grp_size[j]; ++i) {
    //			if (j != 0) {
    //				log_lambda_mmle_sum += sum(log(lambda_mmle_store.col(grp_size_cs[j - 1] + i)));
    //			}
    //			else {
    //				log_lambda_mmle_sum += sum(log(lambda_mmle_store.col(i)));
    //			}
    //		}
    //		overflow_check = sum(log(eta_mmle_store.col(j))) / (double)mmle_samp_size - log_lambda_mmle_sum / ((double)grp_size[j] * (double)mmle_samp_size);
    //		//q[j] = digamma_inv(overflow_check);
    //		//p[j] = digamma_inv(sum(log(eta_mmle_store.col(j))) / (double)mmle_samp_size + sum(log(gamma_mmle_store.col(j))) / (double)mmle_samp_size);
    //		q[j] = q[j];
    //		p[j] = p[j];
    //		log_lambda_mmle_sum = 0;
    //	}
    //}
    
    ++cnt;
    if (cnt % 500 == 0 && verbose) {
      Rcpp::Rcout << cnt << " Burn-in Draws" << std::endl;
    }
  }
  
  if (verbose) {
    Rcpp::Rcout << "Burn-in Iterations Complete" << std::endl;
  }
  
  //Rcpp::Rcout << lambda_mmle_store << std::endl;
  
  cnt = 0;
  arma::colvec q_new = q;
  arma::colvec p_new = p;
  while (delta_mmle >= terminate_mmle) {
    
    //Rcpp::Rcout << "Inside loop" << std::endl;
    
    //Draw alpha
    alpha = arma::mvnrnd(alpha_term1 - alpha_term2 * beta, sigma_sq*CtCinv);
    
    //Draw beta
    //for (int g = 0; g < M; ++g) {
    //	local_param_expand_inv[g] = 1.0 / (gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
    //}
    //local_param_inv.diag() = local_param_expand_inv;
    //beta_tmp = inv((1.0 / sigma_sq) * XtX + (1.0 / tau_sq) * local_param_inv);
    //beta = arma::mvnrnd((1.0 / sigma_sq) * beta_tmp * tX * (Y - C * alpha), beta_tmp);
    
    //New Stuff
    //if (btrick == true) {
    //	for (int g = 0; g < M; ++g) {
    //		local_param_expand_inv[g] = 1.0 / (gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
    //		gl_param_expand_diag[g] = tau_sq * gamma_sq[grp_idx[g] - 1] * lambda_sq[g];
    //	}
    //	gl_param_expand.diag() = gl_param_expand_diag;
    //	local_param_inv.diag() = local_param_expand_inv;
    //	beta_tmp_u = arma::mvnrnd(beta_tmp_zeros_M, gl_param_expand);
    //	beta_tmp_delta = arma::mvnrnd(beta_tmp_zeros_n, beta_tmp_zeros_identity_n);
    //	beta_tmp_v = (1.0 / sqrt(sigma_sq)) * (X * beta_tmp_u) + beta_tmp_delta;
    //	beta_tmp_w = arma::solve(((1.0 / sigma_sq) * X * gl_param_expand * tX) + beta_tmp_zeros_identity_n, ((1.0 / sqrt(sigma_sq)) * (Y - C * alpha)) - beta_tmp_v);
    //	beta = beta_tmp_u + (1 / sqrt(sigma_sq)) * gl_param_expand * tX * beta_tmp_w;
    //}
    //else {
    //	for (int g = 0; g < M; ++g) {
    //		local_param_expand_inv[g] = 1.0 / (gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
    //	}
    //	local_param_inv.diag() = local_param_expand_inv;
    //	beta_tmp = inv((1.0 / sigma_sq) * XtX + (1.0 / tau_sq) * local_param_inv);
    //	beta = arma::mvnrnd((1.0 / sigma_sq) * beta_tmp * tX * (Y - C * alpha), beta_tmp);
    //}
    //
    
    if (btrick == true) {
      for (int g = 0; g < M; ++g) {
        gl_param_expand_diag[g] = tau_sq * gamma_sq[grp_idx[g] - 1] * lambda_sq[g];
        gl_param_expand_diag_inv[g] = 1.0 / gl_param_expand_diag[g];
      }
      gl_param_expand.diag() = gl_param_expand_diag;
      
      for (int j = 0; j < M; ++j) {
        beta_tmp_matrix_theta.row(j) = gl_param_expand_diag[j] * tX.row(j);
      }
      
      beta_tmp_u = arma::mvnrnd(beta_tmp_zeros_M, gl_param_expand);
      beta_tmp_delta = arma::mvnrnd(beta_tmp_zeros_n, beta_tmp_zeros_identity_n);
      beta_tmp_v = (1.0 / sqrt(sigma_sq)) * (X * beta_tmp_u) + beta_tmp_delta;
      beta_tmp_w = arma::solve(((1.0 / sigma_sq) * X * beta_tmp_matrix_theta) + beta_tmp_zeros_identity_n, ((1.0 / sqrt(sigma_sq)) * (Y - C * alpha)) - beta_tmp_v);
      beta = beta_tmp_u + (1 / sqrt(sigma_sq)) * (beta_tmp_matrix_theta * beta_tmp_w);
    }
    else {
      for (int g = 0; g < M; ++g) {
        gl_param_expand_diag_inv[g] = 1.0 / (tau_sq * gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
      }
      beta_tmp = (1.0 / sigma_sq) * XtX;
      beta_tmp.diag() = beta_tmp.diag() + gl_param_expand_diag_inv;
      beta = arma::mvnrnd((1.0 / sigma_sq) * tX * (Y - C * alpha), beta_tmp);
      if (stable_solve) {
        beta = chol_solve(beta_tmp, beta);
      }
      else {
        beta = arma::solve(beta_tmp, beta);
      }
    }
    
    //Draw tau^2
    //tau_sq = 1.0 / R::rgamma(((double)M + 1.0) / 2.0, 1.0 / ((beta.t() * local_param_inv * beta) / 2.0 + 1.0 / nu).eval()(0, 0));
    for (int j = 0; j < M; ++j) {
      tau_rate_const += beta[j] * gl_param_expand_diag_inv[j] * beta[j];
    }
    tau_sq = 1.0 / R::rgamma(tau_shape_const, 1.0 / (tau_sq * tau_rate_const / 2.0 + 1.0 / nu));
    tau_rate_const = 0;
    
    //Draw sigma^2
    //sigma_sq = 1.0 / R::rgamma(((double)n + 1.0) / 2.0, 1.0 / (((Y - C * alpha - X * beta).t()*(Y - C * alpha - X * beta)) / 2.0 + 1.0 / nu).eval()(0, 0));
    sigma_sq = 1.0 / R::rgamma(sigma_shape_const, 1.0 / (((Y - C * alpha - X * beta).t()*(Y - C * alpha - X * beta)) / 2.0 + 1.0 / nu).eval()(0, 0));
    
    for (int j = 0; j < J; ++j) {
      //Draw gamma^2
      //if (j != 0) {
      //	stable_psi = 0;
      //	for (int l = grp_size_cs[j - 1]; l < grp_size_cs[j]; ++l) {
      //		stable_psi += pow(beta[l], 2) / lambda_sq[l];
      //	}
      //	stable_psi *= (1.0 / tau_sq);
      //	stable_psi = std::max(stable_psi, stable_const);
      //	gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
      //}
      //else {
      //	stable_psi = 0;
      //	for (int l = 0; l < grp_size_cs[j]; ++l) {
      //		stable_psi += pow(beta[l], 2) / lambda_sq[l];
      //	}
      //	stable_psi *= (1.0 / tau_sq);
      //	stable_psi = std::max(stable_psi, stable_const);
      //	gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
      //}
      
      //Draw gamma^2
      if (j != 0) {
        stable_psi = 0;
        for (int l = grp_size_cs[j - 1]; l < grp_size_cs[j]; ++l) {
          stable_psi += pow(beta[l], 2) / lambda_sq[l];
        }
        stable_psi *= (1.0 / tau_sq);
        stable_psi = std::max(stable_psi, stable_const);
        if (((double)grp_size[j]) / 2.0 < p[j]) {
          //gamma_sq[j] = rgig_cpp(2.0 * eta[j], stable_psi, p[j] - ((double)grp_size[j]) / 2.0);
          gamma_sq[j] = rgig_cpp(stable_psi, 2.0 * eta[j], p[j] - ((double)grp_size[j]) / 2.0);
        }
        else {
          //gamma_sq[j] = 1.0 / rgig_cpp(stable_psi, 2.0 * eta[j], ((double)grp_size[j]) / 2.0 - p[j]);
          gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
        }
      }
      else {
        stable_psi = 0;
        for (int l = 0; l < grp_size_cs[j]; ++l) {
          stable_psi += pow(beta[l], 2) / lambda_sq[l];
        }
        stable_psi *= (1.0 / tau_sq);
        stable_psi = std::max(stable_psi, stable_const);
        if (((double)grp_size[j]) / 2.0 < p[j]) {
          //gamma_sq[j] = rgig_cpp(2.0 * eta[j], stable_psi, p[j] - ((double)grp_size[j]) / 2.0);
          gamma_sq[j] = rgig_cpp(stable_psi, 2.0 * eta[j], p[j] - ((double)grp_size[j]) / 2.0);
        }
        else {
          //gamma_sq[j] = 1.0 / rgig_cpp(stable_psi, 2.0 * eta[j], ((double)grp_size[j]) / 2.0 - p[j]);
          gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
        }
      }
      
      //Draw lambda^2
      sum_inv_lambda_sq = 0.0;
      sum_log_lambda_sq = 0.0;
      for (int i = 0; i < grp_size[j]; ++i) {
        if (j != 0) {
          lambda_sq[grp_size_cs[j - 1] + i] = 1.0 / R::rgamma(q[j] + 0.5, 1.0 / (eta[j] + pow(beta[grp_size_cs[j - 1] + i], 2) / (2.0 * tau_sq * gamma_sq[j])));
          sum_inv_lambda_sq += (1.0 / lambda_sq[grp_size_cs[j - 1] + i]);
          sum_log_lambda_sq += log(lambda_sq[grp_size_cs[j - 1] + i]);
        }
        else {
          lambda_sq[i] = 1.0 / R::rgamma(q[j] + 0.5, 1.0 / (eta[j] + pow(beta[i], 2) / (2.0 * tau_sq * gamma_sq[j])));
          sum_inv_lambda_sq += (1.0 / lambda_sq[i]);
          sum_log_lambda_sq += log(lambda_sq[i]);
        }
      }
      
      //Draw eta
      //eta[j] = R::rgamma(p[j] + q[j] * (double)grp_size[j], 1.0 / (gamma_sq[j] + sum_inv_lambda_sq));
      eta[j] = 1.0;
      
    }
    
    //Draw nu
    nu = 1.0 / R::rgamma(1.0, 1.0 / ((1.0 / tau_sq) + (1.0 / sigma_sq)));
    
    //Rcpp::Rcout << cnt << " count" << std::endl;
    //Rcpp::Rcout << cnt % mmle_samp_size << " remainder" << std::endl;
    
    //Rcpp::Rcout << gamma_mmle_store << std::endl;
    
    //Estimate q
    eta_mmle_store.row(cnt % mmle_samp_size) = eta.t();
    gamma_mmle_store.row(cnt % mmle_samp_size) = gamma_sq.t();
    lambda_mmle_store.row(cnt % mmle_samp_size) = lambda_sq.t();
    if ((cnt + 1) % mmle_samp_size == 0) {
      for (int j = 0; j < J; ++j) {
        for (int i = 0; i < grp_size[j]; ++i) {
          if (j != 0) {
            log_lambda_mmle_sum += sum(log(lambda_mmle_store.col(grp_size_cs[j - 1] + i)));
          }
          else {
            log_lambda_mmle_sum += sum(log(lambda_mmle_store.col(i)));
          }
        }
        overflow_check = sum(log(eta_mmle_store.col(j))) / (double)mmle_samp_size - log_lambda_mmle_sum / ((double)grp_size[j] * (double)mmle_samp_size);
        //q_new[j] = digamma_inv(overflow_check);
        q_new[j] = std::min(digamma_inv(overflow_check), 4.0);
        //p_new[j] = digamma_inv(sum(log(eta_mmle_store.col(j))) / (double)mmle_samp_size + sum(log(gamma_mmle_store.col(j))) / (double)mmle_samp_size);
        p_new[j] = 1.0 / (double)n;
        log_lambda_mmle_sum = 0;
      }
      
      //Rcpp::Rcout << p_new << " a" << std::endl;
      //Rcpp::Rcout << q_new << " b" << std::endl;
      
      //Check Error Tolerance
      //delta_mmle = sum(abs(q_new - q) + abs(p_new - p));
      delta_mmle = ((q_new - q).t()*(q_new - q)).eval()(0, 0) + ((p_new - p).t()*(p_new - p)).eval()(0, 0);
      
      //Rcpp::Rcout << (q_new - q).t()*(q_new - q) << std::endl;
      
      //Rcpp::Rcout << p << " p" << std::endl;
      //Rcpp::Rcout << q << " q" << std::endl;
      //Rcpp::Rcout << delta_mmle << " d" << std::endl;
      
      //New p and q vectors
      q = q_new;
      p = p_new;
    }
    
    ++cnt;
    if (cnt % 500 == 0 && verbose) {
      Rcpp::Rcout << cnt << " MMLE Draws" << std::endl;
    }
  }
  
  if (verbose) {
    Rcpp::Rcout << "MMLE Estimate Found" << std::endl;
  }
  
  cnt = 0;
  int total_saved = 0;
  while (total_saved < n_samples) {
    
    //Draw alpha
    alpha = arma::mvnrnd(alpha_term1 - alpha_term2 * beta, sigma_sq*CtCinv);
    
    //Draw beta
    //for (int g = 0; g < M; ++g) {
    //	local_param_expand_inv[g] = 1.0 / (gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
    //}
    //local_param_inv.diag() = local_param_expand_inv;
    //beta_tmp = inv((1.0 / sigma_sq) * XtX + (1.0 / tau_sq) * local_param_inv);
    //beta = arma::mvnrnd((1.0 / sigma_sq) * beta_tmp * tX * (Y - C * alpha), beta_tmp);
    
    //New Stuff
    //if (btrick == true) {
    //	for (int g = 0; g < M; ++g) {
    //		local_param_expand_inv[g] = 1.0 / (gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
    //		gl_param_expand_diag[g] = tau_sq * gamma_sq[grp_idx[g] - 1] * lambda_sq[g];
    //	}
    //	gl_param_expand.diag() = gl_param_expand_diag;
    //	local_param_inv.diag() = local_param_expand_inv;
    //	beta_tmp_u = arma::mvnrnd(beta_tmp_zeros_M, gl_param_expand);
    //	beta_tmp_delta = arma::mvnrnd(beta_tmp_zeros_n, beta_tmp_zeros_identity_n);
    //	beta_tmp_v = (1.0 / sqrt(sigma_sq)) * (X * beta_tmp_u) + beta_tmp_delta;
    //	beta_tmp_w = arma::solve(((1.0 / sigma_sq) * X * gl_param_expand * tX) + beta_tmp_zeros_identity_n, ((1.0 / sqrt(sigma_sq)) * (Y - C * alpha)) - beta_tmp_v);
    //	beta = beta_tmp_u + (1 / sqrt(sigma_sq)) * gl_param_expand * tX * beta_tmp_w;
    //}
    //else {
    //	for (int g = 0; g < M; ++g) {
    //		local_param_expand_inv[g] = 1.0 / (gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
    //	}
    //	local_param_inv.diag() = local_param_expand_inv;
    //	beta_tmp = inv((1.0 / sigma_sq) * XtX + (1.0 / tau_sq) * local_param_inv);
    //	beta = arma::mvnrnd((1.0 / sigma_sq) * beta_tmp * tX * (Y - C * alpha), beta_tmp);
    //}
    //
    
    if (btrick == true) {
      for (int g = 0; g < M; ++g) {
        gl_param_expand_diag[g] = tau_sq * gamma_sq[grp_idx[g] - 1] * lambda_sq[g];
        gl_param_expand_diag_inv[g] = 1.0 / gl_param_expand_diag[g];
      }
      gl_param_expand.diag() = gl_param_expand_diag;
      
      for (int j = 0; j < M; ++j) {
        beta_tmp_matrix_theta.row(j) = gl_param_expand_diag[j] * tX.row(j);
      }
      
      beta_tmp_u = arma::mvnrnd(beta_tmp_zeros_M, gl_param_expand);
      beta_tmp_delta = arma::mvnrnd(beta_tmp_zeros_n, beta_tmp_zeros_identity_n);
      beta_tmp_v = (1.0 / sqrt(sigma_sq)) * (X * beta_tmp_u) + beta_tmp_delta;
      beta_tmp_w = arma::solve(((1.0 / sigma_sq) * X * beta_tmp_matrix_theta) + beta_tmp_zeros_identity_n, ((1.0 / sqrt(sigma_sq)) * (Y - C * alpha)) - beta_tmp_v);
      beta = beta_tmp_u + (1 / sqrt(sigma_sq)) * (beta_tmp_matrix_theta * beta_tmp_w);
    }
    else {
      for (int g = 0; g < M; ++g) {
        gl_param_expand_diag_inv[g] = 1.0 / (tau_sq * gamma_sq[grp_idx[g] - 1] * lambda_sq[g]);
      }
      beta_tmp = (1.0 / sigma_sq) * XtX;
      beta_tmp.diag() = beta_tmp.diag() + gl_param_expand_diag_inv;
      beta = arma::mvnrnd((1.0 / sigma_sq) * tX * (Y - C * alpha), beta_tmp);
      if (stable_solve) {
        beta = chol_solve(beta_tmp, beta);
      }
      else {
        beta = arma::solve(beta_tmp, beta);
      }
    }
    
    //Draw tau^2
    //tau_sq = 1.0 / R::rgamma(((double)M + 1.0) / 2.0, 1.0 / ((beta.t() * local_param_inv * beta) / 2.0 + 1.0 / nu).eval()(0, 0));
    for (int j = 0; j < M; ++j) {
      tau_rate_const += beta[j] * gl_param_expand_diag_inv[j] * beta[j];
    }
    tau_sq = 1.0 / R::rgamma(tau_shape_const, 1.0 / (tau_sq * tau_rate_const / 2.0 + 1.0 / nu));
    tau_rate_const = 0;
    
    //Draw sigma^2
    //sigma_sq = 1.0 / R::rgamma(((double)n + 1.0) / 2.0, 1.0 / (((Y - C * alpha - X * beta).t()*(Y - C * alpha - X * beta)) / 2.0 + 1.0 / nu).eval()(0, 0));
    sigma_sq = 1.0 / R::rgamma(sigma_shape_const, 1.0 / (((Y - C * alpha - X * beta).t()*(Y - C * alpha - X * beta)) / 2.0 + 1.0 / nu).eval()(0, 0));
    
    for (int j = 0; j < J; ++j) {
      //Draw gamma^2
      //if (j != 0) {
      //	stable_psi = 0;
      //	for (int l = grp_size_cs[j - 1]; l < grp_size_cs[j]; ++l) {
      //		stable_psi += pow(beta[l], 2) / lambda_sq[l];
      //	}
      //	stable_psi *= (1.0 / tau_sq);
      //	stable_psi = std::max(stable_psi, stable_const);
      //	gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
      //}
      //else {
      //	stable_psi = 0;
      //	for (int l = 0; l < grp_size_cs[j]; ++l) {
      //		stable_psi += pow(beta[l], 2) / lambda_sq[l];
      //	}
      //	stable_psi *= (1.0 / tau_sq);
      //	stable_psi = std::max(stable_psi, stable_const);
      //	gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
      //}
      
      //Draw gamma^2
      if (j != 0) {
        stable_psi = 0;
        for (int l = grp_size_cs[j - 1]; l < grp_size_cs[j]; ++l) {
          stable_psi += pow(beta[l], 2) / lambda_sq[l];
        }
        stable_psi *= (1.0 / tau_sq);
        stable_psi = std::max(stable_psi, stable_const);
        if (((double)grp_size[j]) / 2.0 < p[j]) {
          //gamma_sq[j] = rgig_cpp(2.0 * eta[j], stable_psi, p[j] - ((double)grp_size[j]) / 2.0);
          gamma_sq[j] = rgig_cpp(stable_psi, 2.0 * eta[j], p[j] - ((double)grp_size[j]) / 2.0);
        }
        else {
          //gamma_sq[j] = 1.0 / rgig_cpp(stable_psi, 2.0 * eta[j], ((double)grp_size[j]) / 2.0 - p[j]);
          gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
        }
      }
      else {
        stable_psi = 0;
        for (int l = 0; l < grp_size_cs[j]; ++l) {
          stable_psi += pow(beta[l], 2) / lambda_sq[l];
        }
        stable_psi *= (1.0 / tau_sq);
        stable_psi = std::max(stable_psi, stable_const);
        if (((double)grp_size[j]) / 2.0 < p[j]) {
          //gamma_sq[j] = rgig_cpp(2.0 * eta[j], stable_psi, p[j] - ((double)grp_size[j]) / 2.0);
          gamma_sq[j] = rgig_cpp(stable_psi, 2.0 * eta[j], p[j] - ((double)grp_size[j]) / 2.0);
        }
        else {
          //gamma_sq[j] = 1.0 / rgig_cpp(stable_psi, 2.0 * eta[j], ((double)grp_size[j]) / 2.0 - p[j]);
          gamma_sq[j] = 1.0 / rgig_cpp(2.0 * eta[j], stable_psi, ((double)grp_size[j]) / 2.0 - p[j]);
        }
      }
      
      //Draw lambda^2
      sum_inv_lambda_sq = 0.0;
      sum_log_lambda_sq = 0.0;
      for (int i = 0; i < grp_size[j]; ++i) {
        if (j != 0) {
          lambda_sq[grp_size_cs[j - 1] + i] = 1.0 / R::rgamma(q[j] + 0.5, 1.0 / (eta[j] + pow(beta[grp_size_cs[j - 1] + i], 2) / (2.0 * tau_sq * gamma_sq[j])));
          sum_inv_lambda_sq += (1.0 / lambda_sq[grp_size_cs[j - 1] + i]);
          sum_log_lambda_sq += log(lambda_sq[grp_size_cs[j - 1] + i]);
        }
        else {
          lambda_sq[i] = 1.0 / R::rgamma(q[j] + 0.5, 1.0 / (eta[j] + pow(beta[i], 2) / (2.0 * tau_sq * gamma_sq[j])));
          sum_inv_lambda_sq += (1.0 / lambda_sq[i]);
          sum_log_lambda_sq += log(lambda_sq[i]);
        }
      }
      
      //Draw eta
      //eta[j] = R::rgamma(p[j] + q[j] * (double)grp_size[j], 1.0 / (gamma_sq[j] + sum_inv_lambda_sq));
      eta[j] = 1.0;
    }
    
    //Draw nu
    nu = 1.0 / R::rgamma(1.0, 1.0 / ((1.0 / tau_sq) + (1.0 / sigma_sq)));
    
    //Save output
    if (cnt % n_thin == 0) {
      //alpha_store.row(total_saved) = alpha.t();
      //beta_store.row(total_saved) = beta.t();
      //lambda_store.row(total_saved) = lambda_sq.t();
      //gamma_store.row(total_saved) = gamma_sq.t();
      tau_store[total_saved] = tau_sq;
      sigma_store[total_saved] = sigma_sq;
      //eta_store.row(total_saved) = eta.t();
      //q_store.row(total_saved) = q.t();
      //p_store.row(total_saved) = p.t();
      nu_store[total_saved] = nu;
      
      alpha_store.col(total_saved) = alpha;
      beta_store.col(total_saved) = beta;
      lambda_store.col(total_saved) = lambda_sq;
      gamma_store.col(total_saved) = gamma_sq;
      eta_store.col(total_saved) = eta;
      //q_store.col(total_saved) = q;
      //p_store.col(total_saved) = p;
      
      ++total_saved;
      if (total_saved % 500 == 0 && verbose) {
        Rcpp::Rcout << total_saved << " Samples Drawn" << std::endl;
      }
    }
    
    ++cnt;
  }
  
  return(List::create(Named("alphas") = alpha_store, Named("betas") = beta_store, Named("lambda_sqs") = lambda_store,
                            Named("gamma_sqs") = gamma_store, Named("tau_sqs") = tau_store, Named("sigma_sqs") = sigma_store,
                            Named("a") = p, Named("b") = q, Named("X") = X, Named("C") = C, Named("Y") = Y,
                            Named("grp_idx") = grp_idx, Named("n_burn_in") = n_burn_in, Named("n_samples") = n_samples, Named("n_thin") = n_thin));
}
