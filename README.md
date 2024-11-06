# TP2 Vision artificielle et traitement des images

## Étudiants

-   **Goudal** Victor GOUV07120100
-   **Erard** Julien ERAJ09050200
-   **Flaceliere** Matthieu FLAM30090200

## Introduction

Ce rapport présente notre travail sur le TP2 du cours 8INF804 - Vision artificielle et traitement des images. L'objectif est d'utiliser un algorithme de segmentation sur des images d'échantillons de minéraux afin de les isoler et d'extraire la moyenne sur chaque canal BGR de l'image d'origine.

## Utilisation

### Prérequis

-   Python 3
-   Libs: numpy, matplotlib, cv2, pandas, scikit-image
-   Dossier contenant les images des échantillons sous forme :

```
images/
  Echantillion1Mod2_301.png
  Echantillion1Mod2_302.png
  Echantillion1Mod2_303.png
  Echantillion1Mod2_304.png
  Echantillion1Mod2_305.png
  Echantillion1Mod2_306.png
  Echantillion1Mod2_316.png
  Echantillion1Mod2_422.png
  Echantillion1Mod2_471.png
```

## Utilisation des méthodes de segmentation

## Description de l'algorithme

## Résultats

## Ouverture

Le fichier `segment_anything.py` provient du repository suivant [Segment Anything](https://github.com/facebookresearch/segment-anything) avec certaines modifications. Il nous à permit de tester la segmentation avec le model SAM de Facebook. Le modèle à été relativement précis avec un paramètres basic il est surement possible d'améliorer les resultats obtenue en modifiant les paramètres. Nous avons été limité par notre puissance de calcul (environ 2min pour l'image de test).
![SAM](./images_docs/SAM.png)
