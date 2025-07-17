# azure_func/HttpTrigger/__init__.py
import os
import sys
import json
import pickle
import logging
import pandas as pd
import azure.functions as func

# Rendre hybrid.py visible et injecter classes pour pickle
from ..hybrid import TemporalHybrid, Weights
sys.modules['__main__'].TemporalHybrid = TemporalHybrid
sys.modules['__main__'].Weights = Weights

# Chemins
ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL = os.path.join(ROOT, "artifacts", "temporal_hybrid.pkl")
HIST = os.path.join(ROOT, "artifacts", "user_history.csv")

# Log des chemins et existence des fichiers
logging.info(f"MODEL path: {MODEL}, exists: {os.path.exists(MODEL)}")
logging.info(f"HIST path: {HIST}, exists: {os.path.exists(HIST)}")

# Chargement du modèle
try:
    with open(MODEL, "rb") as f:
        hybrid = pickle.load(f)
        hybrid_model = pickle.load(f)
except Exception as e:
    logging.error(f"Erreur chargement pickle modèle: {e}", exc_info=True)
    raise

# Chargement de l'historique utilisateur
try:
    df_hist = pd.read_csv(HIST)
    user_hist = df_hist.groupby("user_id")["click_article_id"].apply(list).to_dict()
    logging.info(f"Loaded history for {len(user_hist)} users")
except Exception as e:
    logging.error(f"Erreur chargement CSV historique: {e}", exc_info=True)
    user_hist = {}

# Fonction principale

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        uid_str = req.params.get("user_id")
        if not uid_str or not uid_str.isdigit():
            raise ValueError("missing or invalid user_id")
        uid = int(uid_str)
    except Exception as e:
        logging.error(f"Bad request: {e}")
        return func.HttpResponse(
            json.dumps({"error": "missing or invalid user_id"}),
            status_code=400,
            mimetype="application/json"
        )

    # Lecture de l'historique
    try:
        hist = json.loads(req.params.get("history", "[]"))
    except json.JSONDecodeError:
        hist = []
        return func.HttpResponse(
            json.dumps({"error": "invalid history parameter"}),
            status_code=400,
            mimetype="application/json"
        )
    if not hist and uid in user_hist:
        hist = user_hist[uid]

    # Recommandation
    try:
        recs = hybrid_model.recommend(uid, hist, k=5)
    except Exception as e:
        logging.error(f"Erreur during recommend(): {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": "internal server error"}),
            status_code=500,
            mimetype="application/json"
        )

    # Réponse
    return func.HttpResponse(
        json.dumps({"user_id": uid, "recommendations": recs}),
        status_code=200,
        mimetype="application/json"
    )
