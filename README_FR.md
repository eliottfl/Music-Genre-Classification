# Classification Automatique de Genres Musicaux (Projet de TIPE)

**Candidat :** Eliott FLAMENT  
**Formation :** CPGE MPI (Maths, Physique, Informatique)  
**Note obtenue :** 15.8 / 20  
**Problématique :** Peut-on entraîner un modèle capable de reconnaître automatiquement et de manière fiable le genre musical d'un morceau ?

Ce projet étudie la classification de morceaux de musique avec plusieurs algorithmes d'apprentissage supervisé.

## 1. Jeu de Données et Extraction

Le projet utilise le dataset **FMA Small**:
* 8 000 morceaux de 30 secondes.
* 8 genres principaux (Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock) répartis de manière équilibrée.

### Optimisation par Multiprocessing
L'extraction des caractéristiques audio est lourd en calcul. Pour accélérer le processus, j'ai implémenté du **multiprocessing** dans `extraction.py` :
J'ai utilisé la bibliothèque `multiprocessing.Pool` pour paralléliser l'analyse des fichiers mp3.
* Extraction de 33 features par morceau.

## 2. Prétraitement des Données

Pour optimiser l'apprentissage, les données sont nettoyées avec deux scripts :
* **Corrélation (`correlation.py`) :** Suppression des variables corrélées à plus de 95%.
* **Variance (`variance.py`) :** Suppression des variables ayant une variance inférieure à 0.005.

## 3. Comparaison des Modèles

Quatre modèles ont été entraînés et optimisés pour comparer leur précision (Accuracy):

| Modèle | Accuracy |
| :--- | :--- |
| **k-NN** | **39%** |
| **Arbre de Décision** | **35%** |
| **Random Forest** | **47%** |
| **XGBoost** | **48%** |

## 4. Conclusion et Analyse

L'analyse de l'importance des variables montre que les features spectral_centroid, percussive_rms et les mfcc de 1 à 3 sont à chaque fois décisives sur la classification. 
Tous les modèles ont du mal à deviner qu'un morceau est de l'expérimental ou de la pop.

Ce projet m'a permis de découvrir le Machine Learning et la Data Science, un domaine qui m'intéresse beaucoup depuis quelques années. Cela a été l'occasion de mettre en pratique mon cours d'apprentissage automatique avec le k-NN et l'arbre de décision, puis d'aller plus loin avec le Random Forest et XGBoost. Je suis content de ce projet, des résultats et de ce que j'ai appris. J'ai hâte de continuer plus loin.

## 5. Installation et Usage

1. **Dépendances :** `pip install pandas numpy librosa matplotlib seaborn scikit-learn xgboost tqdm`
2. **Pipeline :**
   - Extraire les données : `python extraction.py`
   - Filtrer : `python correlation.py` puis `python variance.py`
   - Entraîner : `python model_xgboost.py`