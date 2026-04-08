#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 10) {
  stop("Usage: r_bglss_driver.R X.csv y.csv group_sizes.csv coef_out.csv beta_samples_out.csv niter burnin seed save_samples")
}

x_path <- args[[1]]
y_path <- args[[2]]
g_path <- args[[3]]
coef_out <- args[[4]]
samples_out <- args[[5]]
niter <- as.integer(args[[6]])
burnin <- as.integer(args[[7]])
seed <- as.integer(args[[8]])
save_samples <- as.integer(args[[9]]) == 1L

if (!requireNamespace("MBSGS", quietly = TRUE)) {
  stop("R package 'MBSGS' is required. Install it with install.packages('MBSGS').")
}

set.seed(seed)

X <- as.matrix(read.csv(x_path, header = FALSE, check.names = FALSE))
y <- as.numeric(read.csv(y_path, header = FALSE, check.names = FALSE)[, 1])
group_size <- as.integer(scan(g_path, what = integer(), sep = ",", quiet = TRUE))
if (length(group_size) == 0) {
  stop("group_size is empty.")
}

fml <- names(formals(MBSGS::BGLSS))
add_arg_if_supported <- function(lst, candidates, value) {
  for (nm in candidates) {
    if (nm %in% fml) {
      lst[[nm]] <- value
      return(lst)
    }
  }
  return(lst)
}

call_args <- list()
call_args <- add_arg_if_supported(call_args, c("Y", "y"), matrix(y, ncol = 1))
call_args <- add_arg_if_supported(call_args, c("X", "x"), X)
call_args <- add_arg_if_supported(call_args, c("group_size", "group.size", "grp_size", "groupsize", "gsize"), group_size)
call_args <- add_arg_if_supported(call_args, c("niter", "n.iter", "iter", "n_sample"), niter)
call_args <- add_arg_if_supported(call_args, c("burnin", "burn.in", "n_burnin", "n.burnin", "nburn"), burnin)
call_args <- add_arg_if_supported(call_args, c("seed", "random.seed", "rng_seed"), seed)

fit <- do.call(MBSGS::BGLSS, call_args)

extract_first <- function(obj, candidates) {
  for (nm in candidates) {
    if (!is.null(obj[[nm]])) return(obj[[nm]])
  }
  NULL
}

p <- ncol(X)
beta_draws <- extract_first(fit, c("beta", "Beta", "beta_samples", "beta.draws", "beta_sample", "BetaSamples"))
coef_vec <- extract_first(fit, c("beta_hat", "beta.hat", "coef", "coefficients", "postmean", "beta_mean"))

if (!is.null(beta_draws)) {
  if (is.vector(beta_draws)) {
    beta_draws <- matrix(as.numeric(beta_draws), nrow = 1)
  } else {
    beta_draws <- as.matrix(beta_draws)
  }
  if (ncol(beta_draws) == p) {
    coef <- colMeans(beta_draws)
  } else if (nrow(beta_draws) == p) {
    beta_draws <- t(beta_draws)
    coef <- colMeans(beta_draws)
  } else {
    stop("Unable to interpret beta draws from MBSGS::BGLSS output.")
  }
} else if (!is.null(coef_vec)) {
  coef <- as.numeric(coef_vec)
  if (length(coef) != p) {
    stop("Coefficient vector length mismatch in MBSGS::BGLSS output.")
  }
  beta_draws <- NULL
} else {
  stop(paste0(
    "Could not extract coefficients from MBSGS::BGLSS output. Available names: ",
    paste(names(fit), collapse = ", ")
  ))
}

write.table(matrix(coef, nrow = 1), file = coef_out, sep = ",", row.names = FALSE, col.names = FALSE)
if (save_samples && !is.null(beta_draws)) {
  write.table(beta_draws, file = samples_out, sep = ",", row.names = FALSE, col.names = FALSE)
}

