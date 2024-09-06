import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

def create__preproc_pipe() -> Pipeline:
    """
    Crée un pipeline pour le prétraitement des données, avec un traitement parallèle
    pour les caractéristiques numériques et catégorielles.
    return: un objet pipeline
    """
    # Exemple : Suppose que nous avons des colonnes numériques et catégorielles
    colonnes_numeriques = ['col_num1', 'col_num2']
    colonnes_categorielles = ['col_cat1', 'col_cat2']
    
    # Pipeline pour les colonnes numériques
    preproc_num = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Pipeline pour les colonnes catégorielles
    preproc_cat = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Transformer les colonnes en parallèle
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', preproc_num, colonnes_numeriques),
            ('cat', preproc_cat, colonnes_categorielles)
        ]
    )
    
    return preprocessor

def create_model_pipe() -> Pipeline:
    """
    Crée un modèle d'entraînement avec un pipeline
    return: un objet pipeline ou modèle
    """
    # Pipeline avec le modèle de régression forêt aléatoire
    model_pipeline = Pipeline(steps=[
        ('preprocessor', create__preproc_pipe()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    return model_pipeline

def save_model(model: Pipeline, filename: str) -> None:
    """
    Enregistre le modèle dans le dossier models
    """
    chemin_dossier = 'models'
    os.makedirs(chemin_dossier, exist_ok=True)
    chemin_fichier = os.path.join(chemin_dossier, filename)
    joblib.dump(model, chemin_fichier)
    print(f"Modèle sauvegardé dans {chemin_fichier}")

def save_metrics(metrics: dict, filename: str) -> None:
    """
    Enregistre les métriques dans le dossier metrics
    """
    chemin_dossier = 'metrics'
    os.makedirs(chemin_dossier, exist_ok=True)
    chemin_fichier = os.path.join(chemin_dossier, filename)
    with open(chemin_fichier, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Métriques sauvegardées dans {chemin_fichier}")

def load_pipe(filename: str) -> Pipeline:
    """
    Charge le modèle (ou pipeline) depuis le dossier models
    """
    chemin_fichier = os.path.join('models', filename)
    if not os.path.exists(chemin_fichier):
        raise FileNotFoundError(f"Le fichier {chemin_fichier} n'existe pas.")
    model = joblib.load(chemin_fichier)
    print(f"Modèle chargé depuis {chemin_fichier}")
    return model

def predict(data: pd.DataFrame, model: Pipeline) -> np.ndarray:
    """
    Effectue des prédictions à l'aide du modèle entraîné.
    
    # ATTENTION : Les données doivent être prétraitées de la même manière
    que les données d'entraînement
    """
    if not isinstance(model, Pipeline):
        raise ValueError("Le modèle fourni n'est pas un Pipeline")
    
    predictions = model.predict(data)
    return predictions

# Exemple d'utilisation
if __name__ == "__main__":
    # Supposons que nous avons un DataFrame 'df' avec des données à prédire
    model = create_model_pipe()
    save_model(model, 'random_forest_model.pkl')
    
    # Charger le modèle et faire des prédictions (exemple)
    model_charge = load_pipe('random_forest_model.pkl')
    df_test = pd.DataFrame({
        'col_num1': [1.0, 2.0],
        'col_num2': [3.0, 4.0],
        'col_cat1': ['A', 'B'],
        'col_cat2': ['C', 'D']
    })
    predictions = predict(df_test, model_charge)
    print("Prédictions :", predictions)
