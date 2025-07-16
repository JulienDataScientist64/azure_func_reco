# azure_func/HttpTrigger/__init__.py
import os, sys, json, pickle, logging
import azure.functions as func
import pandas as pd

# 1) Import / désérialisation
from ..hybrid import TemporalHybrid, Weights
sys.modules['__main__'].TemporalHybrid = TemporalHybrid
sys.modules['__main__'].Weights        = Weights

# 2) Chemins
ROOT   = os.path.dirname(os.path.dirname(__file__))
MODEL  = os.path.join(ROOT, "artifacts", "temporal_hybrid.pkl")
HIST   = os.path.join(ROOT, "artifacts", "user_history.csv")

with open(MODEL, "rb") as f:
    hybrid_model = pickle.load(f)

user_hist = (
    pd.read_csv(HIST)
      .groupby("user_id")["click_article_id"]
      .apply(list)
      .to_dict()
)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info(f"🔔 Requête reçue: {req.url}")

    # — user_id
    uid_str = req.params.get("user_id")
    if not uid_str or not uid_str.isdigit():
        return func.HttpResponse(
            json.dumps({"error": "missing or invalid user_id"}),
            status_code=400,
            mimetype="application/json"
        )
    uid = int(uid_str)

    # — history
    try:
        hist = json.loads(req.params.get("history", "[]"))
    except json.JSONDecodeError:
        return func.HttpResponse(
            json.dumps({"error": "invalid history parameter"}),
            status_code=400,
            mimetype="application/json"
        )

    # fallback si vide
    if not hist and uid in user_hist:
        hist = user_hist[uid]

    # — appel modèle
    try:
        recs = hybrid_model.recommend(uid, hist, k=5)
    except Exception as e:
        logging.error(f"Erreur recommend(): {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": "internal server error"}),
            status_code=500,
            mimetype="application/json"
        )

    # — réponse
    return func.HttpResponse(
        json.dumps({"user_id": uid, "recommendations": recs}),
        status_code=200,
        mimetype="application/json"
    )
