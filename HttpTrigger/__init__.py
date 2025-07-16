# azure_func/HttpTrigger/__init__.py
import os
import sys
import json
import pickle
import logging

import pandas as pd
import azure.functions as func

# ---------------------------------------------------
# 0. Rendre hybrid.py visible et injecter classes pour pickle
# ---------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))  # chemin vers azure_func/
sys.path.append(ROOT)
from hybrid import TemporalHybrid, Weights
import importlib
_main_mod = importlib.import_module("__main__")
setattr(_main_mod, "TemporalHybrid", TemporalHybrid)
setattr(_main_mod, "Weights", Weights)

# ---------------------------------------------------
# 1. Chemins vers les artefacts
# ---------------------------------------------------
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
MODEL_PATH     = os.path.join(ARTIFACTS_DIR, "temporal_hybrid.pkl")
CSV_PATH       = os.path.join(ARTIFACTS_DIR, "user_history.csv")

# ---------------------------------------------------
# 2. Chargement du modèle et de l'historique
# ---------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    hybrid_model = pickle.load(f)

user_hist = (
    pd.read_csv(CSV_PATH)
      .groupby("user_id")["click_article_id"]
      .apply(list)
      .to_dict()
)

# ---------------------------------------------------
# 3. Handler HTTP principal
# ---------------------------------------------------
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info(f"🔔 Requête reçue: {req.method} {req.url}")

    # Lecture possible du JSON body
    uid = None
    hist = None
    if req.method == "POST":
        try:
            body = req.get_json()
            uid = body.get("user_id")
            hist = body.get("history")
        except ValueError as e:
            return func.HttpResponse(
                json.dumps({"error": "invalid json", "details": str(e)}),
                status_code=400,
                mimetype="application/json"
            )

    # Fallback sur query params si nécessaire
    if uid is None:
        uid = req.params.get("user_id")
        hist = req.params.get("history")

    if uid is None:
        return func.HttpResponse(
            json.dumps({"error": "missing user_id"}),
            status_code=400,
            mimetype="application/json"
        )

    uid = str(uid)
    # Si history est une string, tenter un JSON parse
    if isinstance(hist, str):
        try:
            hist = json.loads(hist)
        except json.JSONDecodeError:
            hist = None

    # Utiliser l'historique stocké si pas de history valide
    if not isinstance(hist, list):
        hist = user_hist.get(uid, [])

    # Appel du modèle
    try:
        recs = hybrid_model.recommend(uid, hist, k=5)
    except Exception as e:
        logging.error(f"Erreur recommend(): {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": "internal server error"}),
            status_code=500,
            mimetype="application/json"
        )

    # Réponse JSON
    return func.HttpResponse(
        json.dumps({"user_id": uid, "recommendations": recs}),
        status_code=200,
        mimetype="application/json"
    )
