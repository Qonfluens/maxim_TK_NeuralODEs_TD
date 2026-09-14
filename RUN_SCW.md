```
MYIP=151.115.166.176
```

1. Créer la machine 

Console Scaleway → Compute → Instances → Create instance → image Ubuntu.

2. Se connecter

```bash
ssh root@$MYIP
```

3. Installer Docker + les outils (directement sur la machine, pas dans un container)


```
curl -fsSL https://get.docker.com | sh
apt-get update && apt-get install -y pipx
pipx install mlflow
```

4. Récupérer le code du dépôt

```
git clone https://github.com/Qonfluens/maxim_TK_NeuralODEs_TD.git maxim_TK_NeuralODEs_TD
cd maxim_TK_NeuralODEs_TD
```

5. Construire les deux images Docker, sur place

```
docker build -t tktd-neuralodes:latest .
docker build -f Dockerfile.mlflow -t tktd-neuralodes:mlflow .
```

6. Démarrer le serveur MLflow

```
mkdir -p ~/mlflow-data
/root/.local/bin/mlflow server \
  --backend-store-uri sqlite:///~/mlflow-data/mlflow.db \
  --default-artifact-root ~/mlflow-data/artifacts \
  --host 0.0.0.0 --port 5000 &
```

Le `&` le lance en arrière-plan pour que vous récupériez la main dans le terminal.

7. Test rapide

```
mkdir -p output
docker run --rm -v "$PWD/output:/repo/output" tktd-neuralodes:latest \
  Rscript run/run_TKTD_bayes.R --data data/data_artificial_additive.csv \
  --bridge generic --out-exp 1 --n-update 200 --n-iter 200 --n-iter-waic 100
```

8. Lancer les simulations en les connectant à MLflow

Adapter le nombre de "JOBS" en fonction du nombre de coeur de la machine.


FAST RUN:
```
export MLFLOW_TRACKING_URI="http://host.docker.internal:5000"
N_UPDATE=200 N_ITER=200 N_ITER_WAIC=100 IMAGE=tktd-neuralodes:mlflow JOBS=4 bash run/reproduce_paper.sh
```

LONG RUN:
```
export MLFLOW_TRACKING_URI="http://host.docker.internal:5000"
N_UPDATE=2500 N_ITER=2500 N_ITER_WAIC=1500 IMAGE=tktd-neuralodes:mlflow JOBS=4 bash run/reproduce_paper.sh
```

`reproduce_paper.sh` transmet automatiquement `MLFLOW_TRACKING_URI` à 
chaque container. Le serveur MLflow tourne sur la machine hôte, les containers
s'y connectent via le réseau interne de Docker.

Note: un Ctrl+C sur `reproduce_paper.sh` n'arrête pas forcément les containers
déjà lancés. Vérifiez toujours avec `docker ps` après coup, et faites:

```
docker kill $(docker ps -q)
```

9. Voir le tableau de bord MLflow — sans ouvrir de port public (plus sûr), depuis votre PC local :

```
ssh -N -L 5000:localhost:5000 root@$MYIP
```

