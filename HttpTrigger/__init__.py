# azure_func/HttpTrigger/__init__.py
import os, sys, json, pickle, logging, azure.functions as func

from ..hybrid import TemporalHybrid, Weights
sys.modules['__main__'].TemporalHybrid = TemporalHybrid
sys.modules['__main__'].Weights        = Weights

ROOT   = os.path.dirname(os.path.dirname(__file__))
MODEL  = os.path.join(ROOT, "artifacts", "temporal_hybrid.pkl")
HIST   = os.path.join(ROOT, "artifacts", "user_history.csv")

with open(MODEL, "rb") as f:
    hybrid = pickle.load(f)

import pandas as pd
user_hist = pd.read_csv(HIST).groupby("user_id")["click_article_id"].apply(list).to_dict()

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        uid = int(req.params.get("user_id"))
    except (TypeError, ValueError):
        return func.HttpResponse("missing user_id", status_code=400)

    try:
        hist = json.loads(req.params.get("history", "[]"))
    except json.JSONDecodeError:
        hist = []
    if not hist and uid in user_hist:
        hist = user_hist[uid]

    recs = hybrid.recommend(uid, hist, k=5)
    return func.HttpResponse(
        json.dumps({"user_id": uid, "recommendations": recs}),
        mimetype="application/json")
