# app_cloud.py
import streamlit as st
import requests
import pandas as pd
from pathlib import Path

# Fonction utilitaire pour récupérer le chemin de base
def get_base_path():
    return Path(__file__).parent

# Titre de l'app
st.title("MVP Recommandation Streamlit - Cloud")

# URL de l'API (Azure Function déployée en production)
api_url = st.sidebar.text_input(
    "URL de l'Azure Function (prod)", 
    value="https://reco-func-2025-geavgmgkasbha0a3.francecentral-01.azurewebsites.net/api/HttpTrigger"
)

# Charger les user_ids depuis le CSV local pour le menu déroulant
@st.cache_data
def load_user_ids():
    base = get_base_path()
    csv_path = base / "artifacts" / "user_history.csv"
    if not csv_path.exists():
        st.sidebar.error(f"Fichier introuvable: {csv_path}")
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.sidebar.error(f"Erreur lecture CSV: {e}")
        return []
    if "user_id" not in df.columns:
        st.sidebar.error("Colonne 'user_id' manquante dans le CSV.")
        return []
    return sorted(df['user_id'].astype(str).unique().tolist())

user_ids = load_user_ids()

# Afficher dropdown ou input libre
if user_ids:
    st.sidebar.success(f"{len(user_ids)} user_id disponibles")
    user_id = st.sidebar.selectbox("Sélectionner un user_id", user_ids)
else:
    user_id = st.sidebar.text_input("Entrez votre user_id manuellement")

# Bouton pour déclencher l'appel API
def fetch_recommendations(uid: str, url: str):
    """
    Envoie une requête POST à l'API prod et retourne la liste de recommandations.
    """
    try:
        # Tentative d'appel POST JSON
        resp = requests.post(url, json={"user_id": uid}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("recommendations", [])
    except requests.exceptions.Timeout:
        st.error(f"La requête a dépassé le temps limite vers {url} (timeout).")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Impossible de joindre le service à {url}. Vérifie ta connexion réseau et l'URL.")
        return None
    except requests.exceptions.HTTPError as he:
        st.error(f"Erreur HTTP {he.response.status_code} depuis l'API: {he}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur API prod à {url}: {e}")
        return None

if st.sidebar.button("Obtenir recommandations (prod)"):
    if not user_id:
        st.error("Veuillez renseigner ou sélectionner un user_id.")
    else:
        recs = fetch_recommendations(user_id, api_url)
        if recs is None:
            st.warning("Aucune réponse de l'API.")
        elif recs:
            st.subheader(f"Recommandations pour user_id {user_id} :")
            for idx, item in enumerate(recs, start=1):
                st.write(f"{idx}. {item}")
        else:
            st.info("Aucune recommandation trouvée pour ce user_id.")
