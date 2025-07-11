# build_artifacts.py
import pandas as pd, pickle, os
from azure_func.hybrid import TemporalHybridRecommenderOpti, Weights

BASE   = os.path.dirname(__file__)
ART    = os.path.join(BASE, "azure_func", "artifacts")
os.makedirs(ART, exist_ok=True)

# 1) jeux de données bruts -------------------------------------------------
clicks_csv = os.path.join(BASE, "data", "clicks.csv")        # <-- CHEMIN VERS VOTRE CSV
emb_parq   = os.path.join(BASE, "data", "embeddings.parquet")# <-- CHEMIN VERS embeddings

df_clicks  = pd.read_csv(clicks_csv)
emb_df     = pd.read_parquet(emb_parq).set_index("article_id")

# 2) entraînement ----------------------------------------------------------
model = TemporalHybridRecommenderOpti(emb_df, w=Weights(), window_h=24, factors=50)
model.fit(df_clicks)

# 3) sauvegarde ------------------------------------------------------------
with open(os.path.join(ART, "temporal_hybrid.pkl"), "wb") as f:
    pickle.dump(model, f)

# (optionnel) : historique fallback
(
 df_clicks[["user_id", "click_article_id", "click_timestamp"]]
   .sort_values("click_timestamp")
   .to_csv(os.path.join(ART, "user_history.csv"), index=False)
)

print("✅  Nouveau pickle + history générés dans", ART)
