# 🎬 Movie Chatbot — RNN LSTM

Chatbot entraîné sur des répliques de films (Cornell Movie-Dialogs Corpus) à l'aide d'un réseau de neurones récurrent LSTM, avec une interface interactive dans Jupyter Notebook.

---

## 📁 Structure du projet

```
projet_chatbot/
├── chatbot_rnn.ipynb        # Notebook principal (pipeline complet)
├── chatbot_model.keras      # Modèle sauvegardé (après entraînement)
├── tokenizer.pkl            # Tokenizer sauvegardé
├── responses.json           # Liste des réponses possibles (classes)
├── meta.json                # Hyperparamètres (MAX_LEN, NUM_CLASSES…)
├── training_curves.png      # Courbes loss / accuracy
├── requirements.txt         # Dépendances Python
└── .gitignore
```

> Les fichiers `*.keras`, `*.pkl`, `*.json` (sauf `meta.json`) sont générés lors de l'exécution du notebook.

---

## ⚙️ Installation

### 1. Cloner le repo

```bash
git clone https://github.com/<ton-user>/projet_chatbot.git
cd projet_chatbot
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer Jupyter

```bash
jupyter notebook chatbot_rnn.ipynb
```

---

## 🚀 Utilisation

Le notebook est divisé en **11 étapes** à exécuter dans l'ordre :

| Étape | Description |
|-------|-------------|
| 0 | Installation des packages |
| 1 | Imports |
| 2 | Téléchargement du Movie-Corpus (ConvoKit) |
| 3 | Extraction des paires input → réponse |
| 4 | Préparation des données (tokenisation, padding, labels) |
| 5 | Construction du modèle LSTM |
| 6 | Entraînement (EarlyStopping) |
| 7 | Sauvegarde du modèle et artefacts |
| 8 | Courbes d'apprentissage |
| 9 | Chargement du modèle *(si reprise sans ré-entraîner)* |
| 10 | Fonction de prédiction |
| 11 | **Interface chatbot interactive** (ipywidgets) |

---

## 🧠 Architecture du modèle

```
Embedding(vocab_size, 64, input_length=15)
       ↓
LSTM(128)
       ↓
Dropout(0.3)
       ↓
Dense(128, activation='relu')
       ↓
Dropout(0.2)
       ↓
Dense(num_classes, activation='softmax')
```

**Approche :** classification — chaque réponse unique est une classe. Le LSTM prédit quelle réponse renvoyer à partir de l'input tokenisé et lemmatisé.

---

## 📦 Dépendances

Voir `requirements.txt`. Principales :

- `tensorflow >= 2.12`
- `convokit`
- `nltk`
- `ipywidgets`
- `matplotlib`

---

## 📊 Dataset

**Cornell Movie-Dialogs Corpus** via [ConvoKit](https://convokit.cornell.edu/)  
220 579 échanges issus de 617 films.  
Chargement automatique via :

```python
from convokit import Corpus, download
corpus = Corpus(filename=download("movie-corpus"))
```

