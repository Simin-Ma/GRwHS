
.post_summary = function(draws, dimension = 1) {
  smry = function(x) {c(mean = mean(x), lcl.95 = quantile(x, probs=0.025),
                           ucl.95 = quantile(x, probs=0.975))}
  return(apply(draws, dimension, FUN = smry))
}
