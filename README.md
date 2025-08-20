# Détection de la Tuberculose à partir de radiographies pulmonaires

## Objectif
Développer un modèle de Deep Learning (CNN) capable de détecter la tuberculose à partir d’images radiographiques de la poitrine.  
Le projet inclut également l’explicabilité via **Grad-CAM** pour visualiser les zones des poumons qui influencent la prédiction.

## Dataset
[TB Chest Radiography Database](https://www.kaggle.com/datasets/nih-chest-xrays/data)  

- Images de poumons normaux et tuberculeux
- Prétraitement : redimensionnement à 224x224, normalisation des pixels

Structure du dataset utilisée :
TB_Chest_Radiography_Database/
├── Tuberculosis/
│ ├── img_0001.png
│ └── ...
├── Normal/
│ ├── img_0001.png
│ └── ...
├── Tuberculosis.metadata.xlsx
├── Normal.metadata.xlsx
└── README.md.txt
## Résultats
| Métrique                 | Valeur |
|---------------------------|--------|
| Accuracy                  | ~92%   |
| Précision (TB)            | 93%    |
| Rappel (TB)               | 88%    |
| F1-Score (TB)             | 90%    |

## Modèle
- Réseau CNN avec plusieurs couches Conv2D + MaxPooling  
- Couche Dense finale pour classification binaire (TB / Normal)  
- Fonction d’activation : Sigmoid  
- Optimiseur : Adam  
- Loss : Binary Crossentropy  
- Explicabilité : Grad-CAM appliqué sur la dernière couche convolutionnelle  