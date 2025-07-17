# azure_func/HttpTrigger/__init__.py
from pathlib import Path
import sys, json, pickle, logging
import pandas as pd
import azure.functions as func

# ---------------------------------------------------------------------------
# 1) chemin racine du package déployé  =>   /home/site/wwwroot
ROOT = Path(__file__).resolve().parent.parent

MODEL = ROOT / "artifacts" / "temporal_hybrid.pkl"
HIST  = ROOT / "artifacts" / "user_history.csv"

logging.info("MODEL = %s (exists=%s)", MODEL, MODEL.exists())
logging.info("HIST  = %s (exists=%s)", HIST,  HIST.exists())
# ---------------------------------------------------------------------------

# 2) rendre les classes visibles pour pickle
from ..hybrid import TemporalHybrid, Weights
sys.modules["__main__"].TemporalHybrid = TemporalHybrid
sys.modules["__main__"].Weights        = Weights

# 3) chargement du modèle ----------------------------------------------------
try:
    with MODEL.open("rb") as f:
        hybrid_model = pickle.load(f)
except Exception as e:
    logging.exception("Impossible de charger le modèle")
    raise

# 4) chargement de l’historique utilisateur ---------------------------------
try:
    df_hist   = pd.read_csv(HIST)
    user_hist = (
        df_hist.groupby("user_id")["click_article_id"].apply(list).to_dict()
    )
except Exception as e:
    logging.exception("Impossible de charger l’historique CSV")
    user_hist = {}

# 5) fonction déclenchée par HTTP -------------------------------------------
def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        uid = int(req.params.get("user_id", ""))
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "missing or invalid user_id"}),
            status_code=400,
            mimetype="application/json",
        )

    # historique passé dans l’URL ou récupéré du CSV
    try:
        hist = json.loads(req.params.get("history", "[]"))
    except json.JSONDecodeError:
        return func.HttpResponse(
            json.dumps({"error": "invalid history parameter"}),
            status_code=400,
            mimetype="application/json",
        )
    if not hist and uid in user_hist:
        hist = user_hist[uid]

    try:
        recs = hybrid_model.recommend(uid, hist, k=5)
    except Exception as e:
        logging.exception("Erreur pendant recommend()")
        return func.HttpResponse(
            json.dumps({"error": "internal server error"}),
            status_code=500,
            mimetype="application/json",
        )

    return func.HttpResponse(
        json.dumps({"user_id": uid, "recommendations": recs}),
        status_code=200,
        mimetype="application/json",
    )
