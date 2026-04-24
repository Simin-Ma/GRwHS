data {
  int<lower=1> n;
  int<lower=1> d;
  matrix[n, d] X;
  array[n] int<lower=0, upper=1> y;
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
  real c2 = square(slab_scale) * caux;
  vector[d] lambda = local[1] .* sqrt(local[2]);
  real tau = global[1] * sqrt(global[2]) * scale_global;
  vector[d] lambda_tilde = sqrt(c2 * square(lambda) ./ (c2 + square(tau) * square(lambda)));
  vector[d] beta = z_beta .* lambda_tilde * tau;
}

model {
  beta0 ~ normal(0, scale_icept);
  z_beta ~ normal(0, 1);
  global[1] ~ normal(0, 1);
  global[2] ~ inv_gamma(0.5 * nu_global, 0.5 * nu_global);
  local[1] ~ normal(0, 1);
  local[2] ~ inv_gamma(0.5 * nu_local, 0.5 * nu_local);
  caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);
  y ~ bernoulli_logit(beta0 + X * beta);
}
