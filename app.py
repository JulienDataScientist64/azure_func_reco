# app.py
import streamlit as st
import requests
import pandas as pd
from pathlib import Path

# Fonction utilitaire pour récupérer le chemin de base
def get_base_path():
    return Path(__file__).parent

# Titre de l'app
st.title("MVP Recommandation Streamlit")

# URL de l'API (configurable) - par défaut sur le port où tourne ta Function
api_url = st.sidebar.text_input(
    "URL de l'Azure Function", 
    value="http://localhost:7071/api/HttpTrigger"
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
if st.sidebar.button("Obtenir recommandations"):
    if not user_id:
        st.error("Veuillez renseigner ou sélectionner un user_id.")
    else:
        try:
            response = requests.post(
                api_url,
                json={"user_id": user_id},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            recs = data.get("recommendations", [])
            if recs:
                st.subheader(f"Recommandations pour user_id {user_id} :")
                for idx, item in enumerate(recs, start=1):
                    st.write(f"{idx}. {item}")
            else:
                st.info("Aucune recommandation trouvée pour ce user_id.")
        except requests.exceptions.RequestException as e:
            st.error(
                f"Impossible de contacter l'API à {api_url} : {e}\n"
                "Vérifie que ta Function tourne (port, dossier) "
            )
