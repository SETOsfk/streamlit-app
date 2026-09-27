"""Can this data answer "who shops online?" — a provenance and signal audit. Every README number comes from here.

    python python/audit.py        # ~1 min; writes results/ and docs/
Data: Kaggle sahilgod/consumers-shopping-trends-2026 (CC0). Its author says it was synthesized.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Consumer_Shopping_Trends_2026.csv"
OUT, FIG = ROOT / "results", ROOT / "docs"
SEED, N_NULL = 20260927, 200
CAT, TARGET = ["gender", "city_tier"], "shopping_preference"
SPEND = ["avg_store_spend", "avg_online_spend"]


def load() -> pd.DataFrame:
    return pd.read_csv(DATA)


def spearman_matrix(X: np.ndarray) -> np.ndarray:
    R = np.apply_along_axis(stats.rankdata, 0, X)
    return np.corrcoef(R, rowvar=False)


def folds(y: pd.Series, k: int = 5) -> np.ndarray:
    """Stratified folds, written to results/folds.csv so R scores the very same splits."""
    rng = np.random.default_rng(SEED)
    f = np.empty(len(y), int)
    for cls in sorted(y.unique()):
        idx = rng.permutation(np.flatnonzero(y.to_numpy() == cls))
        f[idx] = np.arange(len(idx)) % k
    return f


def cv_logit(d: pd.DataFrame, num: list[str], cat: list[str], fold: np.ndarray) -> dict:
    """Unpenalised multinomial logistic regression on standardised inputs (R: nnet::multinom, same folds)."""
    y, cls = d[TARGET], np.array(sorted(d[TARGET].unique()))
    P = np.zeros((len(d), len(cls)))
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        pre = ColumnTransformer([("n", StandardScaler(), num)] + ([("c", OneHotEncoder(drop="first"), cat)] if cat else []))
        m = make_pipeline(pre, LogisticRegression(C=np.inf, max_iter=5000)).fit(d.loc[tr, num + cat], y[tr])
        P[te] = m.predict_proba(d.loc[te, num + cat])
    pred = cls[P.argmax(1)]
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"), "log_loss": log_loss(y, P, labels=cls)}


def main() -> None:
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    d = load()
    y = d[TARGET]
    num = [c for c in d.columns if c not in CAT + [TARGET]]
    X = d[num].to_numpy(float)

    # 1. are the behaviours related to each other at all?
    C = spearman_matrix(X)
    iu = np.triu_indices(len(num), 1)
    rng = np.random.default_rng(SEED)
    null_max = [np.abs(spearman_matrix(np.column_stack([rng.permutation(c) for c in X.T]))[iu]).max() for _ in range(N_NULL)]
    obs_max = float(np.abs(C[iu]).max())
    k = int(np.abs(C[iu]).argmax())
    # 2. do the 1-10 scores look like independent uniform draws?
    scores = [c for c in num if set(d[c].unique()) <= set(range(0, 11)) and d[c].nunique() <= 11]
    uniform_p = {c: float(stats.chisquare(d[c].value_counts().sort_index().to_numpy()).pvalue) for c in scores}
    # 3. what separates the three shopping preferences?
    eps = []
    for c in num:
        H = stats.kruskal(*[d.loc[y == g, c] for g in ["Store", "Online", "Hybrid"]]).statistic
        eps.append({"feature": c, "epsilon2": round(float(H / (len(d) - 1)), 4),
                    **{g: round(float(d.loc[y == g, c].median()), 2) for g in ["Store", "Online", "Hybrid"]}})
    eps = sorted(eps, key=lambda r: -r["epsilon2"])
    # 4. how well can it be predicted — with and without the spend columns (spend is an outcome of the channel)?
    fold = folds(y)
    pd.DataFrame({"row": np.arange(len(d)), "fold": fold}).to_csv(OUT / "folds.csv", index=False)
    no_spend = [c for c in num if c not in SPEND]
    prior = y.value_counts(normalize=True)
    models = {"majority": {"accuracy": float(prior.max()), "balanced_accuracy": 1 / 3, "macro_f1": float(f1_score(y, np.full(len(y), prior.idxmax()), average="macro")),
                           "log_loss": float(log_loss(y, np.tile(prior.sort_index().to_numpy(), (len(y), 1)), labels=sorted(prior.index)))},
              "behaviour_only": cv_logit(d, no_spend, CAT, fold),
              "store_spend_only": cv_logit(d, ["avg_store_spend"], [], fold),
              "all_columns": cv_logit(d, num, CAT, fold)}
    models = {m: {k: round(v, 4) for k, v in r.items()} for m, r in models.items()}
    # 5. the label rule: an unpenalised fit on all rows separates the classes perfectly; its weights name the formula
    Z = StandardScaler().fit_transform(d[num])
    full = LogisticRegression(C=np.inf, max_iter=5000).fit(Z, y)
    w = pd.Series(full.coef_[list(full.classes_).index("Online")] - full.coef_[list(full.classes_).index("Store")], index=num)
    w = (w / w.abs().max()).round(3)
    rule = {"train_accuracy": round(float((full.predict(Z) == y).mean()), 4),
            "weights_online_vs_store": w[w.abs() >= 0.01].sort_values(key=abs, ascending=False).to_dict(),
            "n_columns_used": int((w.abs() >= 0.01).sum())}
    spend_q = {g: [float(v) for v in d.loc[y == g, "avg_store_spend"].quantile([.05, .25, .5, .75, .95])] for g in ["Store", "Online", "Hybrid"]}

    res = {"n_rows": len(d), "n_numeric": len(num), "missing": int(d.isna().sum().sum()), "duplicates": int(d.duplicated().sum()),
           "class_share": {k: round(float(v), 4) for k, v in prior.items()},
           "categorical_share": {c: {k: round(float(v), 4) for k, v in d[c].value_counts(normalize=True).items()} for c in CAT},
           "independence": {"pairs": len(iu[0]), "max_abs_rho": round(obs_max, 4), "pair": [num[iu[0][k]], num[iu[1][k]]],
                            "pairs_above_0_1": int((np.abs(C[iu]) > 0.1).sum()), "null_max_abs_rho_median": round(float(np.median(null_max)), 4),
                            "null_share_at_least_obs": round(float(np.mean(np.array(null_max) >= obs_max)), 4),
                            "income_vs_store_spend": round(float(C[num.index("monthly_income"), num.index("avg_store_spend")]), 4),
                            "internet_vs_online_orders": round(float(C[num.index("daily_internet_hours"), num.index("monthly_online_orders")]), 4)},
           "uniform_scores": {"n": len(scores), "n_p_above_0_01": int(sum(p > 0.01 for p in uniform_p.values()))},
           "separation": eps, "models": models, "label_rule": rule, "store_spend_quantiles": spend_q}
    (OUT / "audit.json").write_text(json.dumps(res, indent=1))
    pd.DataFrame(C, index=num, columns=num).round(4).to_csv(OUT / "spearman.csv")
    (ROOT / "web" / "data").mkdir(parents=True, exist_ok=True)                  # the static page reads this copy
    (ROOT / "web" / "data" / "audit.json").write_text(json.dumps({**res, "columns": num, "spearman": np.round(C, 3).tolist()}, separators=(",", ":")))
    print(json.dumps({k: res[k] for k in ("independence", "uniform_scores", "models", "label_rule")}, indent=1))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = {"majority": "Always \"Store\"", "behaviour_only": "20 behaviour + 2 demographic columns",
             "store_spend_only": "Store spend alone", "all_columns": "All columns"}
    keys = list(names)
    ax.barh(range(4), [models[k]["balanced_accuracy"] for k in keys], color=["#b9b0a7", "#b9b0a7", "#2d6a8f", "#2d6a8f"])
    for i, k in enumerate(keys):
        ax.text(models[k]["balanced_accuracy"] + 0.01, i, f"{models[k]['balanced_accuracy']:.2f}  (accuracy {models[k]['accuracy']:.3f})", va="center", fontsize=8.5)
    ax.axvline(1 / 3, color="#333", lw=0.8, ls="--")
    ax.set_yticks(range(4), [names[k] for k in keys]); ax.invert_yaxis(); ax.set_xlim(0, 1.25)
    ax.set_xlabel("Balanced accuracy, 5-fold CV (chance = 0.33)")
    ax.set_title("Only the spend column predicts the channel — which is circular", loc="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIG / "models.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
