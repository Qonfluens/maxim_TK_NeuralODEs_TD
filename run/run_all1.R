library("coda")
library("rjags")

# 1
ID_SET = "1"
d = read.delim("data/MaXim__raw_datasets__set1_CLEAN.csv", header = TRUE, sep = ",") 
colnames(d)
MIXTURE = c(d$spiroxamine, d$prothioconazole, d$tebuconazole, d$trifloxystrobin, d$bixafen, d$fluopyram)
N_EXPOSURE = 6
source("Bayes_TKTD_n.R")
source("Bayes_TKTD_n_exp.R")
source("Bayes_TKTD_nn_ReLU_n_exp.R")
source("Bayes_TKTD_nn_ReLU_nn_ReLU_n_exp.R")
source("Bayes_TKTD_nn_ReLU_nn_ReLU_nn_ReLU_n_exp.R")