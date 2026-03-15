data {
  int<lower=1> n;
  int<lower=1> d;
  vector[n] y;
  matrix[n, d] X;

  real<lower=0> scale_icept;
  real<lower=0> scale_global;
  real<lower=1> nu_global;
  real<lower=1> nu_local;
  real<lower=0> slab_scale;
  real<lower=0> slab_df;
}

transformed data {
  real<lower=1e-9> y_sd = fmax(sd(y), 1e-9);
}

parameters {
  real beta0;
  vector[d] z;

  real<lower=0> aux1_global;
  real<lower=0> aux2_global;

  vector<lower=0>[d] aux1_local;
  vector<lower=0>[d] aux2_local;

  real<lower=0> caux;
  real logsigma;
}

transformed parameters {
  real<lower=0> sigma;
  real<lower=0> tau;
  vector<lower=0>[d] lambda;
  real<lower=0> c;
  vector<lower=0>[d] lambda_tilde;
  vector<lower=0>[d] lambda_tilde_sq;
  vector[d] beta;
  vector[n] mu;

  sigma = fmax(exp(logsigma), 1e-9);

  lambda = aux1_local .* sqrt(aux2_local);
  tau = aux1_global * sqrt(aux2_global) * scale_global * sigma;

  c = slab_scale * sqrt(caux);
  lambda_tilde_sq = c^2 * square(lambda) ./ (c^2 + tau^2 * square(lambda) + 1e-12);
  lambda_tilde = sqrt(lambda_tilde_sq);

  beta = z .* lambda_tilde * tau;
  mu = beta0 + X * beta;
}

model {
  z ~ normal(0, 1);

  aux1_local ~ normal(0, 1);
  aux2_local ~ inv_gamma(0.5 * nu_local, 0.5 * nu_local);

  aux1_global ~ normal(0, 1);
  aux2_global ~ inv_gamma(0.5 * nu_global, 0.5 * nu_global);

  caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);
  beta0 ~ normal(0, scale_icept);
  logsigma ~ normal(log(y_sd), 1.0);
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[n] log_lik;
  vector[n] y_rep;

  for (i in 1:n) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
    y_rep[i] = normal_rng(mu[i], sigma);
  }
}
