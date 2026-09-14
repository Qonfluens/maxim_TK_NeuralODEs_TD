# ==============================================================================
# Script 1 : Lancement automatique TKTD-IT Générique (Architecture Flexible)
# ==============================================================================

library(rjags)
library(dplyr)

# 1. Chargement et préparation des données
data_clean <- read.csv("data/MaXim__raw_datasets__set1_CLEAN.csv")

# Nettoyage / structuration basique
data_clean <- data_clean %>%
  arrange(replicate_ID, time) %>%
  group_by(replicate_ID) %>%
  mutate(
    time_ID = row_number(),
    i_prec = ifelse(time_ID == 1, row_number(), lag(row_number())),
    Nprec = ifelse(time_ID == 1, Nsurv, lag(Nsurv))
  ) %>%
  ungroup()

# Identification des colonnes de substances (commençant par 'X')
x_cols <- grep("^X_", colnames(data_clean), value = TRUE)
n_X <- length(x_cols)

# 2. Configuration du réseau de neurones / Pont TK-TD
# Exemple : 2 couches cachées de largeur 4, activation ReLU, sortie exp
hidden_layers <- c(4, 4)
n_layer <- length(hidden_layers) + 1
layer_sizes <- c(n_X, hidden_layers, 1)
M <- max(layer_sizes)

# Construction des masques de poids et de biais
W_mask <- array(0, dim = c(n_layer, M, M))
b_mask <- array(0, dim = c(n_layer, M))

for (l in 1:n_layer) {
  n_in <- layer_sizes[l]
  n_out <- layer_sizes[l + 1]
  W_mask[l, 1:n_out, 1:n_in] <- 1
  b_mask[l, 1:n_out] <- 1
}

# 3. Construction de la liste 'data' pour JAGS
X_mat <- matrix(0, nrow = nrow(data_clean), ncol = M)
X_mat[, 1:n_X] <- as.matrix(data_clean[, x_cols])

jags_data <- list(
  n_data = nrow(data_clean),
  n_X = n_X,
  M = M,
  n_layer = n_layer,
  W_mask = W_mask,
  b_mask = b_mask,
  prec_w = 1.0,                           # Précision des priors des poids (1/sd^2)
  neg_slope = rep(0, n_layer),             # 0 = ReLU
  out_exp = 1,                             # 1 = exp(sortie)
  alpha_split = 0,                         # 0 = alpha global, 1 = alpha distribué
  alpha_out_exp = 0,
  in_mask = c(rep(1, n_X), rep(0, M - n_X)),
  alpha_idx = c(1:n_X, rep(1, M - n_X)),
  
  # Matrice d'exposition
  X = X_mat,
  time = data_clean$time,
  replicate_ID = as.numeric(as.factor(data_clean$replicate_ID)),
  time_ID = data_clean$time_ID,
  i_prec = data_clean$i_prec,
  Nprec = data_clean$Nprec,
  Nsurv = data_clean$Nsurv,
  
  # Indexations
  kd_idx = c(1:n_X, rep(1, M - n_X)),
  n_alpha = 1,
  
  # Priors TKTD
  kd_meanlog10 = rep(-1, n_X),
  kd_sdlog10 = rep(1, n_X),
  hb_meanlog10 = -2,
  hb_sdlog10 = 1,
  hb_value = 1,
  hb_valueFIXED = 0,
  beta_minlog10 = -1,
  beta_maxlog10 = 2,
  alpha_meanlog10 = rep(1, 1),
  alpha_sdlog10 = rep(1, 1)
)

# 4. Ininitialisation et Exécution JAGS
model_generic <- jags.model(
  file = "src/JAGS_TKTD_IT_generic.txt",
  data = jags_data,
  n.chains = 3,
  n.adapt = 1000
)

update(model_generic, 1000) # Burn-in

samples_generic <- coda.samples(
  model_generic,
  variable.names = c("kd", "hb", "beta", "alpha", "D"),
  n.iter = 5000
)

# Résumé des résultats
summary(samples_generic)