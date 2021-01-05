# TP_DRL
Projet Apprentissage profond par renforcement

Alexandre DEVILLERS p1608591  
Solayman AYOUBI p1608583  

## Installation

```sh
sudo apt install cmake libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libbz2-dev libfluidsynth-dev libgme-dev libopenal-dev zlib1g-dev timidity tar nasm wget ffmpeg
```
Attention: On utilise une version custom de vizdoomgym pour régler le problème du wrapper d'enregistrement  
par conséquent il faut bien installer la version de vizdoomgym indiquer dans le fichier requirements.txt.
Le code reste éxecutable avec la version de base de vizdoomgym mais produira un warning

```sh
pip3 install -r requirements.txt
```

## Utilisation

```sh
python3 main.py
```

Pour changer d'environnment il suffit de modifier la chaine de caractères
USED_SET avec un des PARAM_SET existant.
