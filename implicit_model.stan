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
  real<lower=0> xtilde[Nrows];
  simplex[3] theta; // lapse probabilities
}

transformed parameters {
  real<lower=0> p_resp[Nrows];
  real xc[Nrows];
  real<lower=0> this_st;
  real<lower=0> this_snJ;
  real this_pr_odds;
  real log_odds;
  
  for (j in 1:Nrows) {
    this_st =  st + midwidth[j] * dst1 + maxwidth[j] * dst2;
    this_snJ = snJ + isSaccade[j] * dsnJ;
    this_pr_odds = pr_odds - maxprior[j] * dlpl + minprior[j] * dlpu;
    log_odds = normal_lpdf(xtilde[j]|jump[j], sqrt(this_st^2 + sJ^2)) -
                normal_lpdf(xtilde[j]|jump[j], sqrt(this_st^2 + this_snJ^2));
    
    p_resp[j] = theta[2] + theta[1] * (inv_logit(10 * (log_odds - this_pr_odds)));
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
  for (j in 1:Nrows) {
    xtilde[j] ~ normal(jump[j], st + midwidth[j] * dst1 + maxwidth[j] * dst2);
  }
}

