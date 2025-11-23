<img src="https://img.icons8.com/fluency/96/health-graph.png" width="120" align="right" />

# Smart Health Guardian  
### Système Intelligent de Surveillance Sanitaire en Temps Réel

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-link.streamlit.app)](https://your-streamlit-link.streamlit.app)  
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)  
[![Stars](https://img.shields.io/github/stars/votre-pseudo/smart-health-guardian?style=social)](https://github.com/votre-pseudo/smart-health-guardian)

> Une application web moderne basée sur **Streamlit + MediaPipe + YOLOv8** pour détecter automatiquement les comportements à risque sanitaire :  
> - Éternuements non couverts  
> - Port incorrect du masque  
> - Non-respect de la distanciation sociale  

Idéal pour les hôpitaux, écoles, entreprises, gares, aéroports, etc.

---

## Demo en vidéo (2 min)

https://github.com/user-attachments/assets/xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx  
*(Remplace par ta propre vidéo YouTube ou GitHub Assets)*

---

## Fonctionnalités principales

| Module                     | Fonctionnalité                                                                 | Technologie utilisée                     |
|----------------------------|----------------------------------------------------------------------------------|------------------------------------------|
| Détection d’Éternuements   | Détecte les éternuements + vérifie si la personne se couvre (main, coude, etc.) | MediaPipe Pose + Hands                   |
| Détection de Masques       | With mask / Without mask / Incorrect mask                                      | YOLOv8 custom entraîné                   |
| Distanciation Sociale      | Alerte en temps réel quand deux personnes sont trop proches                    | YOLOv8n + calcul euclidien des centroïdes |
| Interface moderne          | Design fluide, responsive, dark gradient, animations                           | Streamlit + CSS personnalisé             |
| Export résultats           | Vidéo annotée + rapport JSON détaillé                                          | OpenCV + JSON                            |

---



## Installation rapide (local)

```bash
# 1. Cloner le projet
git clone https://github.com/votre-pseudo/smart-health-guardian.git
cd smart-health-guardian

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
