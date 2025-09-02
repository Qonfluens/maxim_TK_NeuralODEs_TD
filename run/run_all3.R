library("coda")
library("rjags")

# 3
ID_SET = "3"
d = read.delim("data/MaXim__raw_datasets__set3_CLEAN.csv", header = TRUE, sep = ",") 
colnames(d)
MIXTURE = c(d$thiacloprid, d$imidacloprid, d$cyfluthrin, d$clothianidin, d$beta.cyfluthrin, d$thiodicarb)
N_EXPOSURE = 6
source("Bayes_TKTD_n.R")
source("Bayes_TKTD_n_exp.R")
source("Bayes_TKTD_nn_ReLU_n_exp.R")
source("Bayes_TKTD_nn_ReLU_nn_ReLU_n_exp.R")
source("Bayes_TKTD_nn_ReLU_nn_ReLU_nn_ReLU_n_exp.R")