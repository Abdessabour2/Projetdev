# Projetdev
# MedicalPlatforme Application

Bienvenue dans le projet de **MedicalPlatforme** avec prédiction de maladies de la peau. Ce projet comprend un modèle d'apprentissage profond pour la détection des lésions cutanées, un chatbot médical intelligent développé avec **Rasa**, ainsi qu'un backend en **Django** et un frontend en **React**.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les outils suivants :

- Python 3.8+ (pour le backend)
- Node.js et npm (pour le frontend)
- Un compte Kaggle et le modèle pré-entrainé `model.h5` pour la prédiction de lésions cutanées.

---

## Étape 1 : Telecharger les dossier chatbot.zib et backend.zib et frontend.zib
## Étape 2 :Téléchargez le modèle model.h5 
depuis  output de notebook https://www.kaggle.com/code/ahzababdessabour/cnn-skin-cancer et placez le fichier model.h5 téléchargé dans le dossier /backend/ .
##   Étape 3 : Configuration de Rasa (Chatbot)
** Dans le répertoire /chatbot**
Création de l'environnement:
python -m venv rasa-env
Installation de Rasa :
  pip install rasa
Initialisez un nouveau projet Rasa:
  rasa-env\Scripts\activate
  rasa init
  Lancer l'API:
    rasa run actions
    rasa run --enable-api
##   Étape 4 : Configuration du Backend (Django)
  python manage.py migrate
  python manage.py runserver
##   Étape 5 :Lancer le Frontend (React)
  cd frontend
  npm install
  npm start
  
Cela ouvrira l'interface frontend sur http://localhost:3000.

  

