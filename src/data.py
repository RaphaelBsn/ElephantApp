import pandas as pd
import os
import urllib.request

def charger_donnees() -> pd.DataFrame:
    """
    Essaie de charger les données depuis le dossier de données (localement).
    Si les données n'existent pas, les récupérer sur le web et les enregistrer dans le dossier de données.
    À la fin, retourner les données sous forme de DataFrame pandas.
    """
    chemin_fichier = 'data/train.csv'
    url_donnees = 'https://storage.googleapis.com/schoolofdata-datasets/Data-Engineering.Production-Machine-Learning-Code/train.csv'
    
    # Vérifier si le fichier existe localement
    if not os.path.exists(chemin_fichier):
        # Si le fichier n'existe pas, le télécharger
        print("Téléchargement des données depuis le web...")
        os.makedirs(os.path.dirname(chemin_fichier), exist_ok=True)
        urllib.request.urlretrieve(url_donnees, chemin_fichier)
        print(f"Données téléchargées et enregistrées dans {chemin_fichier}")
    
    # Charger les données dans un DataFrame pandas
    donnees = pd.read_csv(chemin_fichier)
    print("Données chargées avec succès")
    return donnees

def nettoyer_donnees(donnees: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyer les données en supprimant les valeurs manquantes et les doublons.
    """
    print("Nettoyage des données...")
    # Supprimer les valeurs manquantes
    donnees_sans_na = donnees.dropna()
    # Supprimer les doublons
    donnees_nettoyees = donnees_sans_na.drop_duplicates()
    
    print("Données nettoyées avec succès")
    return donnees_nettoyees

# Exemple d'utilisation
if __name__ == "__main__":
    donnees = charger_donnees()
    donnees_nettoyees = nettoyer_donnees(donnees)
    print(donnees_nettoyees.head())
