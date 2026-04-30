data {
  int<lower=1> n;
  int<lower=1> d;
  matrix[n, d] X;
  array[n] int<lower=0, upper=1> y;
  vector[d] x_center;
  vector<lower=0>[d] x_scale;
  real scale_icept;
  real<lower=0> scale_global;
  real<lower=0> nu_global;
  real<lower=0> nu_local;
  real<lower=0> slab_scale;
  real<lower=0> slab_df;
}

parameters {
  real beta0;
  vector[d] z_beta;
  array[2] real<lower=0> global;
  array[2] vector<lower=0>[d] local;
  real<lower=0> caux;
}

transformed parameters {
  real c = slab_scale * sqrt(caux);
  vector[d] lambda = local[1] .* sqrt(local[2]);
  real tau = global[1] * sqrt(global[2]) * scale_global;
  vector[d] lambda_tilde;
  vector[d] beta;
  vector[d] beta_scaled;
  real beta0_shifted;

  for (j in 1:d) {
    real a = tau * lambda[j];
    real shrink;
    if (a > c) {
      real ratio = c / a;
      shrink = c / sqrt(1 + square(ratio));
    } else if (a > 0) {
      real ratio = a / c;
      shrink = a / sqrt(1 + square(ratio));
    } else {
      shrink = 0;
    }
    lambda_tilde[j] = shrink / fmax(tau, 1e-12);
    beta[j] = z_beta[j] * shrink;
  }
  beta_scaled = beta .* x_scale;
  beta0_shifted = beta0 + dot_product(x_center, beta);
}

model {
  beta0 ~ normal(0, scale_icept);
  z_beta ~ normal(0, 1);
  global[1] ~ normal(0, 1);
  global[2] ~ inv_gamma(0.5 * nu_global, 0.5 * nu_global);
  local[1] ~ normal(0, 1);
  local[2] ~ inv_gamma(0.5 * nu_local, 0.5 * nu_local);
  caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);
  target += bernoulli_logit_glm_lpmf(y | X, beta0_shifted, beta_scaled);
}
