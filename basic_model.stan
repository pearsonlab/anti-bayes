//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//
functions {
  real x_crit(real st, real sJ, real snJ, real logodds) {
    real numerator;
    real denominator;
    numerator = 2 * logodds + log(st^2 + sJ^2) - log(st^2 + snJ^2);
    denominator = 1/(st^2 + snJ^2) - 1/(st^2 + sJ^2);
    return sqrt(numerator/denominator);
  }
}

data {
  int<lower=1> Nrows;
  int<lower=0, upper=1> response[Nrows];
  real<lower=0> jump[Nrows];
  int<lower=0, upper=1> isSaccade[Nrows];
  int<lower=0, upper=1> midwidth[Nrows];
  int<lower=0, upper=1> maxwidth[Nrows];
  int<lower=0, upper=1> minprior[Nrows];
  int<lower=0, upper=1> maxprior[Nrows];
}

parameters {
  real<lower=0> sJ;
  real<lower=0> snJ;
  real<lower=0> dsnJ;
  real<lower=0> st;
  real<lower=0> dst1;
  real<lower=0> dst2;
  real pr_odds;
  real<lower=0> dlpl;
  real<lower=0> dlpu;
  simplex[3] theta; // lapse probabilities
}

transformed parameters {
  real<lower=0> p_resp[Nrows];
  real xc[Nrows];
  
  for (j in 1:Nrows) {
    // lambda[j] = R[j] * alpha[protein[j], section[j]] * beta[protein[j], layer[j]];
    xc[j] = x_crit(st + midwidth[j] * dst1 + maxwidth[j] * dst2, 
                  sJ, snJ + isSaccade[j] * dsnJ, 
                  pr_odds - maxprior[j] * dlpl + minprior[j] * dlpu); 
    p_resp[j] = theta[2] + 
      theta[1] * (Phi_approx(-(jump[j] + xc[j])/(st + midwidth[j] * dst1 + maxwidth[j] * dst2)) 
      + 1 - Phi_approx((xc[j] - jump[j])/(st + midwidth[j] * dst1 + maxwidth[j] * dst2)));
  }
}

model {
  sJ ~ gamma(5, 5/2.5);
  snJ ~ gamma(1, 1/0.2);
  dsnJ ~ gamma(1, 1);
  dst1 ~ gamma(1, 5);
  dst2 ~ gamma(1, 5);
  pr_odds ~ normal(0, .1);
  dlpu ~ normal(0, 1);
  dlpl ~ normal(0, 1);
  theta ~ dirichlet([20, 1, 1]');
  response ~ bernoulli(p_resp);
}

