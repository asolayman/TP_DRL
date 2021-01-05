# TP_DRL
**Projet Apprentissage profond par renforcement**

Par Alexandre DEVILLERS p1608591 & Solayman AYOUBI p1608583

## Installation

```sh
sudo apt install cmake libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libbz2-dev libfluidsynth-dev libgme-dev libopenal-dev zlib1g-dev timidity tar nasm wget ffmpeg
```
**Attention**: On utilise une version custom de vizdoomgym pour régler le problème du wrapper d'enregistrement  
par conséquent il faut bien installer la version de vizdoomgym indiqué dans le fichier requirements.txt.
Le code reste éxecutable avec la version de base de vizdoomgym mais produira un warning

```sh
pip3 install -r requirements.txt
```

## Utilisation

```sh
python3 main.py
```

Pour changer d'environnment il suffit de modifier la chaine de caractères
USED_SET (`main.py`, L.72) avec un des PARAM_SET existant.

Valeurs possibles pour USED_SET :
- 'CartPole-v1_Train'
- 'CartPole-v1_Test'
- 'VizdoomBasic-v0_Train'
- 'VizdoomBasic-v0_Test'
- 'VizdoomTakeCover-v0_Train'
- 'VizdoomTakeCover-v0_Test'
- 'VizdoomCorridor-v0_Kill'
- 'VizdoomCorridor-v0_Train'
- 'VizdoomCorridor-v0_Test'

Les valeurs finissants par "\_Test" ou "\_Kill" sont celles à utiliser pour reproduire ce qui est dans les vidéos (les vidéos sont les meilleurs / plus interessants episodes sur 100).

Les tous les jeux d'hyper-paramètres ayant permis de faire les vidéos (que ça soit le train, ou le test) sont dans PARAM_SET (`main.py`, L.73).


## Architecture

- `best\_networks` contient tout les réseaux de neurones pré entraînés.
- `best\_replays` contient tout les enregistrements _démo_ des différents réseaux de neurones.
- `network` est le dossier où le réseau courant est sauvegardé suite à une exécution.
- `replay` est le dossier où les replays courants sont sauvegardés suite à une exécution.
- `agent.py` contient la classe avec les algorithmes liés au DQN.
- `main.py` **contient en haut du fichier tous les hyper-paramètres des entraînements et phases de tests (Si vous voulez le détail des valeurs des hyper-paramètres, et pour la reproductibilité)**, ainsi que les boucles d'itération sur les environnements.
