from src.model import create__preproc_pipe, create_model_pipe
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import os
import json

# TODO : Convertir les print en logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    """
    Entraîne le modèle en utilisant un pipeline de prétraitement et de modélisation.
    Évalue ensuite le modèle sur un ensemble de test et enregistre les métriques dans un fichier.
    """
    logger.info("Entraînement du modèle")
    
    # Charger les données
    data = load_data()
    
    # Créer les pipelines de prétraitement et de modèle
    preproc = create__preproc_pipe()
    model = create_model_pipe()
    
    # Diviser les données en ensembles d'entraînement et de test
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    
    # Prétraiter et entraîner le modèle
    train_preproc = preproc.fit_transform(train)
    model.fit(train_preproc)
    
    # Évaluer le modèle sur l'ensemble de test
    test_preproc = preproc.transform(test)
    predictions = model.predict(test_preproc)
    
    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    r2 = r2_score(test, predictions)
    
    logger.info(f"Erreur absolue moyenne (MAE): {mae}")
    logger.info(f"Erreur quadratique moyenne (MSE): {mse}")
    logger.info(f"Coefficient de détermination (R2): {r2}")
    
    logger.info("Modèle entraîné avec succès")
    
    # TODO : Sauvegarder les métriques dans un fichier JSON (PAS DANS LES LOGS)
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    }
    
    # Créer le dossier de métriques s'il n'existe pas
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Sauvegarder les métriques dans un fichier JSON
    metrics_file = os.path.join(metrics_dir, "model_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Métriques sauvegardées dans {metrics_file}")
