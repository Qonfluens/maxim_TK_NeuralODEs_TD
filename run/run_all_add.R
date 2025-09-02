library("coda")
library("rjags")

# additive
ID_SET = "additive"
d = read.delim("data/data_artificial_additive.csv", header = TRUE, sep = ",") 
colnames(d)
MIXTURE = c(d$A, d$B)
N_EXPOSURE = 2
source("Bayes_TKTD_n.R")
source("Bayes_TKTD_n_exp.R")
source("Bayes_TKTD_nn_ReLU_n_exp.R")
source("Bayes_TKTD_nn_ReLU_nn_ReLU_n_exp.R")
source("Bayes_TKTD_nn_ReLU_nn_ReLU_nn_ReLU_n_exp.R")