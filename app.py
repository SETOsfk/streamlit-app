"""Consumer shopping panel — what a synthetic dataset can and cannot tell you.   streamlit run app.py
Numbers come from results/audit.json (python python/audit.py); the charts are computed live from the CSV."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Consumer shopping panel", page_icon="🛒", layout="wide")


@st.cache_data
def load():
    d = pd.read_csv(ROOT / "Consumer_Shopping_Trends_2026.csv")
    return d, json.loads((ROOT / "results" / "audit.json").read_text()), pd.read_csv(ROOT / "results" / "spearman.csv", index_col=0)


d, A, C = load()
tr = st.sidebar.radio("Dil / Language", ["Türkçe", "English"]) == "Türkçe"
T = (lambda a, b: a if tr else b)
M, I = A["models"], A["independence"]
num = [c for c in d.columns if c not in ("gender", "city_tier", "shopping_preference")]

st.title(T("Bu veri \"kim online alışveriş yapar?\" sorusunu cevaplayabilir mi?", "Can this data answer \"who shops online?\""))
st.caption(T(f"{A['n_rows']:,} satır · 25 sütun · Kaggle, CC0 · yazarına göre sentetik (üretilmiş) veri",
             f"{A['n_rows']:,} rows · 25 columns · Kaggle, CC0 · synthetic by its author's own account"))
c1, c2, c3, c4 = st.columns(4)
c1.metric(T("En güçlü ilişki (231 çift)", "Strongest link (231 pairs)"), f"|ρ| = {I['max_abs_rho']:.3f}",
          T(f"bağımsız sütunlarda beklenen {I['null_max_abs_rho_median']:.3f}", f"independent columns give {I['null_max_abs_rho_median']:.3f}"), delta_color="off", delta_arrow="off")
c2.metric(T("Davranış sütunlarıyla tahmin", "Predicting from behaviour"), f"{M['behaviour_only']['balanced_accuracy']:.2f}",
          T("dengeli doğruluk · şans 0,33", "balanced accuracy · chance 0.33"), delta_color="off", delta_arrow="off")
c3.metric(T("Tüm sütunlarla", "With every column"), f"{M['all_columns']['accuracy']:.1%}",
          T(f"etiket {A['label_rule']['n_columns_used']} sütunluk bir formül", f"the label is a formula of {A['label_rule']['n_columns_used']} columns"), delta_color="off", delta_arrow="off")
c4.metric(T("Gelir ↔ mağaza harcaması", "Income ↔ store spend"), f"ρ = {I['income_vs_store_spend']:+.3f}", T("gerçek hayatta pozitif olurdu", "would be positive in real life"), delta_color="off", delta_arrow="off")

tab1, tab2, tab3 = st.tabs([T("Veri denetimi", "Data audit"), T("Tahmin", "Prediction"), T("Keşif", "Explore")])
with tab1:
    st.markdown(T("**Bulgu:** 22 sayısal sütunun hiçbiri bir diğeriyle ilişkili değil; 13 puan sütununun 13'ü de 0–10 arasında düzgün dağılıyor. "
                  "Bu, sütunların birbirinden bağımsız rastgele çekildiği bir üreteç verisi. Bu veriden çıkan kümeler ve korelasyonlar tüketici davranışı değil, üretecin özelliği.",
                  "**Finding:** none of the 22 numeric columns is related to any other, and all 13 score columns are uniform on 0–10. "
                  "These are independent random draws; clusters or correlations found here describe the generator, not shoppers."))
    fig = px.imshow(C, zmin=-1, zmax=1, color_continuous_scale="RdBu", aspect="auto", title=T("Spearman korelasyonları (hepsi ≈ 0)", "Spearman correlations (all ≈ 0)"))
    st.plotly_chart(fig, width="stretch")
    col = st.selectbox(T("Dağılımına bak", "Look at a distribution"), num, index=num.index("tech_savvy_score"))
    st.plotly_chart(px.histogram(d, x=col, nbins=40), width="stretch")
with tab2:
    names = {"majority": T("Hep \"Mağaza\" de", "Always say \"Store\""), "behaviour_only": T("20 davranış + 2 demografi sütunu", "20 behaviour + 2 demographic columns"),
             "store_spend_only": T("Yalnız mağaza harcaması", "Store spend alone"), "all_columns": T("Tüm sütunlar", "All columns")}
    tbl = pd.DataFrame([{T("Model", "Model"): names[k], T("Doğruluk", "Accuracy"): v["accuracy"], T("Dengeli doğruluk", "Balanced accuracy"): v["balanced_accuracy"],
                         "Log loss": v["log_loss"]} for k, v in M.items()])
    st.dataframe(tbl, hide_index=True, width="stretch")
    st.markdown(T("Davranış sütunları tek başına **hiçbir şey** öğretmiyor (dengeli doğruluk şans düzeyinde). Tahmin gücünün tamamı harcama sütunundan geliyor — "
                  "oysa gerçek hayatta mağaza harcaması kanal tercihinin *sonucu*, nedeni değil. Tüm sütunlarla %99,7'ye çıkan model, etiketi üreten formülü geri buluyor:",
                  "Behaviour columns alone teach **nothing** (balanced accuracy at chance). All the predictive power comes from the spend column — "
                  "yet in real life store spend is a *consequence* of the channel, not a cause. The 99.7% model simply recovers the formula that produced the label:"))
    w = pd.Series(A["label_rule"]["weights_online_vs_store"]).rename(T("ağırlık (online − mağaza)", "weight (online − store)"))
    st.bar_chart(w, horizontal=True)
with tab3:
    col = st.selectbox(T("Sütun", "Column"), num, index=num.index("need_touch_feel_score"), key="ex")
    H = stats.kruskal(*[d.loc[d.shopping_preference == g, col] for g in ["Store", "Online", "Hybrid"]]).statistic
    st.metric(T("Etki büyüklüğü (Kruskal ε²)", "Effect size (Kruskal ε²)"), f"{H / (len(d) - 1):.3f}",
              T("0,01 küçük · 0,06 orta · 0,14 büyük", "0.01 small · 0.06 medium · 0.14 large"), delta_color="off", delta_arrow="off")
    st.plotly_chart(px.box(d, x="shopping_preference", y=col, color="shopping_preference", category_orders={"shopping_preference": ["Store", "Online", "Hybrid"]}),
                    width="stretch")
    st.caption(T("11.789 satırda en küçük fark bile \"anlamlı\" çıkar; p değeri yerine etki büyüklüğüne bakın.",
                 "With 11,789 rows even tiny gaps are \"significant\"; read the effect size, not the p-value."))
st.caption(T("Veri: Kaggle sahilgod/consumers-shopping-trends-2026 (CC0). Nisan 2026 ders sürümü git geçmişinde.",
             "Data: Kaggle sahilgod/consumers-shopping-trends-2026 (CC0). The April 2026 course version is in the git history."))
