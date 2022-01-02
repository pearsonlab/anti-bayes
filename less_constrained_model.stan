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
  real<lower=0> snJ[2];
  real<lower=0> st[3];
  real pr_odds[3];
  simplex[3] theta; // lapse probabilities
}

transformed parameters {
  real<lower=0> p_resp[Nrows];
  real xc[Nrows];
  real<lower=0> this_snJ;
  real<lower=0> this_st;
  real this_pr_odds;
  
  for (j in 1:Nrows) {
    this_snJ = snJ[isSaccade[j] + 1];
    
    if (maxprior[j]) 
      this_pr_odds = pr_odds[1];
    else if (minprior[j]) 
      this_pr_odds = pr_odds[3]; 
    else 
      this_pr_odds = pr_odds[2];
      
    if (midwidth[j])
      this_st = st[2];
    else if (maxwidth[j])
      this_st = st[3];
    else
      this_st = st[1];
      
    xc[j] = x_crit(this_st, sJ, this_snJ, this_pr_odds);
    p_resp[j] = theta[2] + 
      theta[1] * (Phi_approx(-(jump[j] + xc[j])/this_st) + 1 
      - Phi_approx((xc[j] - jump[j])/this_st));
  }
}

model {
  sJ ~ gamma(1, 1/2.5);
  snJ ~ gamma(1, 1/0.2);
  st ~ gamma(1, 1/0.2);
  pr_odds ~ normal(0, 1);
  theta ~ dirichlet([20, 1, 1]');
  response ~ bernoulli(p_resp);
}

