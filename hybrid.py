# azure_func/hybrid.py
from __future__ import annotations
from dataclasses import dataclass
import math, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

@dataclass
class Weights:
    pop:     float = .55
    cf:      float = .15
    content: float = .30

class TemporalHybrid:
    def __init__(self, emb: pd.DataFrame, w: Weights | None = None,
                 window_h: int = 24, factors: int = 50):
        self.emb = emb
        self.emb_norm = emb.div(np.linalg.norm(emb.values, axis=1, keepdims=True))
        self.w = w or Weights()
        self.window_h = window_h
        self.factors = factors
        self.pop_scores = None
        self.u_enc, self.i_enc = LabelEncoder(), LabelEncoder()
        self.item2idx = {}

    @staticmethod
    def _mm(x: np.ndarray) -> np.ndarray:
        span = np.ptp(x)
        return np.zeros_like(x) if span == 0 else (x - x.min()) / span

    def fit(self, df: pd.DataFrame):
        dt = pd.to_datetime(df["click_timestamp"], unit="ms")
        recent = df[dt >= dt.max() - pd.Timedelta(hours=self.window_h)]
        self.pop_scores = recent.groupby("click_article_id").size().astype(float)
        self.pop_scores /= self.pop_scores.sum()
        u = self.u_enc.fit_transform(df["user_id"])
        i = self.i_enc.fit_transform(df["click_article_id"])
        self.item2idx = {itm: idx for idx, itm in enumerate(self.i_enc.classes_)}
        mat = csr_matrix((np.ones(len(df)), (u, i)))
        k = min(self.factors, min(mat.shape)-1)
        svd = TruncatedSVD(k, random_state=42)
        self.U = svd.fit_transform(mat)
        self.I = svd.components_.T

    def recommend(self, uid: int, hist: list[int], k: int = 5) -> list[int]:
        cand = [a for a in self.pop_scores.index if a not in hist]
        if not cand:
            return []
        pop_s = self.pop_scores.loc[cand].values
        try:
            uidx = self.u_enc.transform([uid])[0]
            idxs = [self.item2idx[a] for a in cand]
            cf_s = self._mm(self.U[uidx] @ self.I[idxs].T)
        except ValueError:
            cf_s = np.zeros(len(cand))
        vh = [h for h in hist if h in self.emb_norm.index]
        if vh:
            p = self.emb_norm.loc[vh].mean(axis=0).values
            p /= np.linalg.norm(p) + 1e-9
            cont_s = self._mm(self.emb_norm.loc[cand] @ p)
        else:
            cont_s = np.zeros(len(cand))
        score = (self.w.pop * pop_s +
                 self.w.cf  * cf_s  +
                 self.w.content * cont_s)
        return [cand[i] for i in np.argsort(score)[::-1][:k]]
