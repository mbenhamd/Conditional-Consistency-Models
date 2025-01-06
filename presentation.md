
## Amélioration du Contraste et de la Luminosité des Images Radiographiques en Niveaux de Gris à l'aide de Modèles de Cohérence Conditionnelle (CCMs)

**Présentation basée sur l'article "Conditional Consistency Guided Image Translation and Enhancement" - Bhagat et al.**

<br>

\[Mohamed BEN HAMDOUNE]

\[06/01/2025]

<br>

---

  

## Introduction - Les Défis de l'Imagerie Radiographique

**Contexte:** Chez Smiths Detection, nous nous efforçons constamment d'améliorer la qualité de nos images radiographiques. Une meilleure qualité d'image se traduit par une meilleure détection des menaces et une sécurité accrue.

**Problématiques:**

- Les images radiographiques en niveaux de gris peuvent parfois manquer de contraste, rendant difficile la distinction entre les objets et les matériaux, en particulier dans les bagages ou les cargaisons denses.

- Des variations de luminosité peuvent également masquer des détails importants.

- Le bruit, inhérent à la capture d'images à faible dose de rayons X, dégrade encore la qualité de l'image.
**Objectif:** L'objectif est donc d'explorer des techniques avancées pour améliorer le contraste et la luminosité de nos images, tout en réduisant le bruit, afin d'optimiser la détection.

---
## Solution: Modèles de Cohérence Conditionnelle (CCMs)

- **Introduction aux CCMs:**
    - Les CCMs sont une nouvelle classe de modèles génératifs qui excellent dans l'amélioration et la transformation d'images.
    - Ils s'inspirent des modèles de diffusion, mais offrent une génération d'images en une seule étape, ce qui les rend beaucoup plus rapides.
- **Principe de Fonctionnement:**
    - Les CCMs apprennent à mapper un échantillon bruité vers l'échantillon de données original en se basant sur une condition. Cette condition guide le processus de débruitage et de génération.
- **Avantages Clés:**
    - Génération d'images de haute qualité.
    - Processus de génération en une seule étape, donc très rapide (important pour l'analyse en temps réel).
    - Capacité à intégrer des informations conditionnelles pour guider l'amélioration.

---

## Adaptation et Entraînement

- **Architecture du Réseau:**
    - L'article utilise une architecture U-Net qui a fait ses preuves dans le traitement d'images.
    - Le code, disponible dans `improved_consistency_model_conditional.py`, fournit une implémentation de cette architecture que nous pouvons adapter.

- **Données d'Entraînement:**
    - Nous devrons constituer un ensemble de données d'images radiographiques en niveaux de gris pour l'entraînement.
    - Idéalement, nous aurions des paires d'images: une image de faible qualité (faible contraste/bruit) et sa version améliorée correspondante.
    - Nous pourrions utiliser des techniques d'augmentation de données pour enrichir notre ensemble de données.

- **Entraînement:**
    - Le code dans `bci/script.py`, `llvip/script.py`, etc., fournit des exemples d'entraînement pour différents datasets. Nous pouvons nous en inspirer pour entraîner un CCM sur nos données.

---

## Intégration dans nos Systèmes

- **Vitesse d'Exécution:** L'avantage majeur des CCMs est leur capacité de génération en une seule étape. Cela signifie qu'une fois le modèle entraîné, l'amélioration des images sera extrêmement rapide, compatible avec un traitement en temps réel.
- **Implémentation:** Nous pourrions intégrer le CCM entraîné comme un module de prétraitement dans nos logiciels d'analyse d'images existants.
- **Workflow:**
    1. Acquisition de l'image radiographique.
    2. Passage de l'image dans le CCM pour améliorer le contraste et la luminosité.
    3. Analyse de l'image améliorée par nos algorithmes de détection.

---

## Adaptation des CCMs à l’Imagerie Radiographique

Dans le cadre des images radiographiques 16 bits, il est crucial de préserver la dynamique étendue tout en améliorant le contraste et la luminosité. Voici comment nous pouvons adapter les CCMs à nos besoins spécifiques :

1. **Création de Jeux de Données Appariés (Pairs):**
   - Pour chaque image radiographique (source), on génère une version améliorée (cible) en appliquant un filtrage BM3D et une adaptation logarithmique.
   - Les deux images (source et cible) sont ensuite normalisées et recadrées (par exemple en 512×512) afin de constituer des paires prêtes pour l’entraînement.

2. **Préprocessing 16 Bits:**
   - L’imagerie radiographique peut dépasser la plage classique [0,1].
   - Il est donc recommandé de normaliser les pixels sur la plage [0, 65535] (ou [0,1] après conversion) et de conserver la précision 16 bits tout au long du pipeline.

3. **Configuration du Modèle CCM:**
   - Passer en mode "grayscale" (1 canal) pour tenir compte de la nature en niveaux de gris des images radiographiques.
   - Maintenir des couches de convolution plus larges (par ex. 256–512 canaux) pour gérer des détails fins et la résolution élevée.

4. **Stratégie d’Entraînement:**
   - Effectuer un recadrage aléatoire (random crop) d’environ 512x512 pour chaque itération, suivant les recommandations de l’article.
   - Entraîner le modèle durant ~1000 époques (ou davantage selon la taille du dataset).
   - Utiliser des pertes axées sur la cohérence (consistency loss) permettant la reconstruction d’images nettes et fidèles.

5. **Évaluation et Validation:**
   - Comparer l’image reconstituée par le CCM (sortie) avec son équivalent filtré (cible).
   - Utiliser des métriques de contraste (PSNR, SSIM) et vérifier la capacité à révéler les détails cruciaux pour la détection d’objets dangereux.

En suivant ce protocole, nous pouvons incorporer efficacement les CCMs dans nos workflows d’imagerie radiographique 16 bits, en garantissant une amélioration sensible du contraste et de la luminosité sans perdre d’informations essentielles.

---

## Fonctionnement Interne des CCMs : Détails Techniques

Pour mieux comprendre ce qui différencie concrètement les CCMs (Conditional Consistency Models) des modèles de diffusion classiques, il est utile de se pencher sur la manière dont ces modèles gèrent le bruit et la génération d’images :

1. **Modèle à Génération en Une Seule Étape :**
   - Contrairement aux modèles de diffusion qui nécessitent plusieurs itérations de débruitage (multi-step), les CCMs effectuent la génération d’images en une seule passe.
   - La clé de ce fonctionnement réside dans la fonction de cohérence (consistency function) qui apprend à projeter un échantillon bruité vers une image propre, et ce, pour différents niveaux de bruit.

2. **Propriété de Self-Consistency (Auto-Cohérence) :**
   - La fonction de cohérence `g_φ` doit respecter une contrainte dite d’auto-cohérence : si on injecte différents niveaux de bruit dans la même image, le modèle doit toujours aboutir à la même image propre en sortie.
   - Cette propriété est imposée par un terme de perte (consistency loss) qui pénalise l'écart entre les sorties du modèle à différents temps (niveaux de bruit).

3. **Boundary Condition (Condition Frontière) :**
   - Le modèle est entraîné de manière à devenir l’identité (`g_φ(r, ε) = r`) quand le bruit `t` tend vers `ε` (le plus bas niveau de bruit).
   - Ainsi, dès que le bruit est très faible (proche de zéro), le modèle laisse l’image telle quelle, et n’effectue aucune transformation.

4. **Intégration des Informations Conditionnelles :**
   - Pour des tâches de traduction d’images (par exemple, image radiographique d’entrée → image radiographique améliorée), on concatène l’image brute et l’image bruitée avant de les passer au réseau (U-Net).
   - Le réseau apprend ainsi à s’appuyer sur cette information “source” pour guider la génération et obtenir une image cohérente avec le contenu d’origine, tout en améliorant la qualité (plus de contraste, moins de bruit).

5. **Avantages Clés en R&D :**
   - **Rapidité d’Exécution :** L’inférence se fait en une seule étape, contrairement aux méthodes itératives de diffusion. C’est un atout majeur pour des applications temps réel ou quasi temps réel.
   - **Stabilité de l’Entraînement:** Les CCMs n’exigent pas d’optimisation adversariale, évitant ainsi les problèmes de collapse de mode et de tuning complexe liés aux GANs.
   - **Flexibilité:** La même architecture peut être adaptée à différents types d’images (y compris 16 bits) du moment qu’on dispose d’un dataset apparié (entrée → sortie).

Au final, les CCMs se positionnent comme une solution hybride reprenant le meilleur des modèles de diffusion (qualité visuelle élevée) et des GANs (vitesse de génération), tout en offrant une robustesse en formation et une relative simplicité d’implémentation.

---

## Résultats et Métriques d'Évaluation

### Métriques Utilisées

- **PSNR (Peak Signal-to-Noise Ratio):**
    - Mesure la qualité de reconstruction entre l'image originale et l'image traitée
    - Plus la valeur est élevée, meilleure est la qualité
    - Particulièrement pertinent pour évaluer la fidélité de la reconstruction

- **SSIM (Structural Similarity Index Measure):**
    - Évalue la similarité structurelle entre deux images
    - Varie entre 0 et 1 (1 = identique)
    - Plus sensible à la perception humaine que le PSNR

- **NIQE (Naturalness Image Quality Evaluator):**
    - Évalue la qualité naturelle de l'image sans référence
    - Plus le score est bas, plus l'image est naturelle
    - Particulièrement utile pour les datasets non appariés

---

## Résultats par Dataset

### LLVIP (Visible vers Infrarouge)

- **Contexte:** Dataset de 15,488 paires d'images visible/infrarouge
- **Résolution:** Tests sur images 512x512 et 256x256
- **Résultats:**
    - PSNR: 13.11 dB (512x512) / 12.59 dB (256x256)
    - SSIM: Performances supérieures aux méthodes existantes
    - Amélioration notable par rapport à CycleGAN et pix2pixGAN

### BCI (Images Médicales HE vers IHC)

- **Contexte:** 9,746 images médicales appariées
- **Résolution:** Évaluation sur 1024x1024
- **Résultats:**
    - PSNR: 18.29 dB
    - SSIM: Performance exceptionnelle, surpassant les méthodes traditionnelles
    - Particulièrement efficace pour la préservation des détails tissulaires

---

## Résultats sur les Datasets d'Amélioration de Luminosité

### LOL-v1 et LOL-v2

- **Contexte:** Datasets de référence pour l'amélioration de la luminosité
- **Résultats LOL-v1:**
    - PSNR: 21.10 dB
    - SSIM: 0.78
- **Résultats LOL-v2:**
    - Real: PSNR 22.72 dB, SSIM 0.79
    - Synthetic: PSNR 22.00 dB, SSIM 0.87

### Datasets Non Appariés (DICM, LIME, MEF, NPE, VV)

- **Évaluation via NIQE:**
    - DICM: 3.09 (2ème meilleur score)
    - LIME: 3.67 (meilleur score)
    - MEF: 2.96 (meilleur score)
    - NPE: 3.65 (performance compétitive)
    - VV: 4.20 (performance modérée due à la haute résolution)

---

## Analyse des Résultats

### Points Forts

- **Polyvalence:** Performances solides sur différents types de données
- **Robustesse:** Bons résultats même sur des datasets non appariés
- **Efficacité:** Génération en une seule étape, crucial pour applications temps réel

### Limitations

- Légère baisse de performance sur les images haute résolution (VV dataset)
- Compromis entre vitesse et qualité selon la résolution d'image

### Implications pour Smiths Detection

- **Avantages Clés:**
    - Traitement rapide adapté aux contraintes temps réel
    - Qualité suffisante pour la détection de menaces
    - Flexibilité pour différentes conditions d'imagerie

---

```
<>
## Démonstration (si possible)

<!-- .slide: data-transition="fade" -->

- Si possible, une démo rapide avec des images de l'article (par exemple, en utilisant `bci/sampling_and_metrics.py` or `llvip/sampling_and_metrics.py`) peut illustrer l'efficacité des CCMs.
- Montrez une image avant et après traitement par le CCM pour visualiser l'amélioration.

```python
# Code to display images in Obsidian
# Assuming you have preprocessed images
from IPython.display import display, Image

display(Image(filename='path/to/before_image.png'))
display(Image(filename='path/to/after_image.png'))

```
