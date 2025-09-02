library("coda")
library("rjags")

# antagonism
ID_SET = "antagonism"
d = read.delim("data/data_artificial_antagonism.csv", header = TRUE, sep = ",") 
colnames(d)
MIXTURE = c(d$A, d$B)
N_EXPOSURE = 2
source("Bayes_TKTD_n.R")
source("Bayes_TKTD_n_exp.R")
source("Bayes_TKTD_nn_ReLU_n_exp.R")
source("Bayes_TKTD_nn_ReLU_nn_ReLU_n_exp.R")
source("Bayes_TKTD_nn_ReLU_nn_ReLU_nn_ReLU_n_exp.R")