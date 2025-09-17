import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

CSV = Path("matches.csv")

st.set_page_config(page_title="Twitch Predictor", layout="centered")
st.title("Twitch Match Predictor")

# ---------------- Storage ----------------
if CSV.exists():
    df = pd.read_csv(CSV)
else:
    df = pd.DataFrame(columns=[
        "map","mode","kills","placement",
        "K","P","ts"
    ])

# ---------------- Helpers ----------------
def add_row(d):
    global df
    df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)
    df.to_csv(CSV, index=False)

def features_from_history(hist, K, P):
    """Build simple, robust features from prior matches only."""
    if len(hist) == 0:
        return None
    last_n = hist.tail(min(20, len(hist)))  # small, stable window
    f = {
        "n_hist": len(hist),
        "kills_avg": last_n["kills"].mean(),
        "kills_std": last_n["kills"].std(ddof=0) if len(last_n)>1 else 0.0,
        "place_avg": last_n["placement"].mean(),
        "place_std": last_n["placement"].std(ddof=0) if len(last_n)>1 else 0.0,
        "hit_rate_k": (last_n["kills"] >= K).mean(),
        "hit_rate_p": (last_n["placement"] <= P).mean(),
        "K": K, "P": P
    }
    return pd.DataFrame([f])

def make_training(df_all):
    """Create per-row training sets using rolling-history features to avoid leakage."""
    rows_k, rows_p = [], []
    for i in range(len(df_all)):
        hist = df_all.iloc[:i]  # past only
        K = int(df_all.iloc[i]["K"])
        P = int(df_all.iloc[i]["P"])
        fx = features_from_history(hist, K, P)
        if fx is None: 
            continue
        yk = int(df_all.iloc[i]["kills"] >= K)
        yp = int(df_all.iloc[i]["placement"] <= P)
        rows_k.append((fx, yk))
        rows_p.append((fx, yp))
    if rows_k:
        Xk = pd.concat([x for x,_ in rows_k], ignore_index=True)
        yk = np.array([y for _,y in rows_k])
    else:
        Xk, yk = None, None
    if rows_p:
        Xp = pd.concat([x for x,_ in rows_p], ignore_index=True)
        yp = np.array([y for _,y in rows_p])
    else:
        Xp, yp = None, None
    return Xk, yk, Xp, yp

def fit_model(X, y):
    if X is None or len(X) < 25 or len(np.unique(y)) < 2:
        return None, None  # need data
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    # calibration
    p = model.predict_proba(X)[:,1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    return model, iso

def predict(model, iso, Xrow, fallback):
    if model is None or iso is None:
        return fallback
    p = model.predict_proba(Xrow)[:,1][0]
    return float(iso.transform([p])[0])

def implied_prob(multiplier):
    # Pari-mutuel approx: p_imp ≈ 1/M
    try:
        m = float(multiplier)
        return 1.0/m if m>0 else None
    except:
        return None

def edge_and_kelly(p, m):
    """Return edge and Kelly fraction for a YES bet with payout multiplier m."""
    if m is None or m<=1 or p is None:
        return None, 0.0
    b = m - 1.0
    f_star = (b*p - (1-p)) / b
    return p - 1.0/m, max(0.0, min(1.0, f_star))

# ---------------- Input form ----------------
st.subheader("Add match")
with st.form("match"):
    c1, c2 = st.columns(2)
    map_name = c1.text_input("Map", "")
    mode = c2.text_input("Mode", "")
    c3, c4 = st.columns(2)
    kills = c3.number_input("Final kills", min_value=0, step=1)
    placement = c4.number_input("Final placement (1 = win)", min_value=1, step=1)
    c5, c6 = st.columns(2)
    K = c5.number_input("Kills threshold K", min_value=1, value=5, step=1)
    P = c6.number_input("Placement threshold P (≤P counts as YES)", min_value=1, value=10, step=1)
    submitted = st.form_submit_button("Save match")
    if submitted:
        add_row({
            "map": map_name, "mode": mode, "kills": int(kills),
            "placement": int(placement), "K": int(K), "P": int(P),
            "ts": pd.Timestamp.now().isoformat(timespec="seconds")
        })
        st.success("Saved.")

st.divider()
st.subheader("Current probability and edge")

c1, c2, c3 = st.columns(3)
K = c1.number_input("Evaluate K", min_value=1, value=int(df["K"].iloc[-1]) if len(df) else 5, step=1)
P = c2.number_input("Evaluate P", min_value=1, value=int(df["P"].iloc[-1]) if len(df) else 10, step=1)
hist_n = c3.number_input("History window (display only)", min_value=5, value=20, step=1)

# Train models from df
Xk, yk, Xp, yp = make_training(df)
mk, ck = fit_model(Xk, yk)
mp, cp = fit_model(Xp, yp)

# Build current feature row from full history
Xrow = features_from_history(df, int(K), int(P))
if Xrow is None:
    st.info("Add at least one match.")
else:
    # Fallback = moving hit rates
    last_n = df.tail(min(hist_n, len(df)))
    fb_k = (last_n["kills"] >= K).mean() if len(last_n)>0 else 0.5
    fb_p = (last_n["placement"] <= P).mean() if len(last_n)>0 else 0.5

    p_yes_k = predict(mk, ck, Xrow, fb_k)
    p_yes_p = predict(mp, cp, Xrow, fb_p)

    st.write(f"Pr(kills ≥ {int(K)}) ≈ **{p_yes_k:.2f}**")
    st.write(f"Pr(place ≤ {int(P)}) ≈ **{p_yes_p:.2f}**")

    st.markdown("**Optional:** enter pool multipliers to compute edge and Kelly size.")
    d1, d2 = st.columns(2)
    M_yes_k = d1.text_input("Multiplier for YES (kills≥K)", "2.0")
    M_yes_p = d2.text_input("Multiplier for YES (place≤P)", "2.0")

    imp_k = implied_prob(M_yes_k)
    imp_p = implied_prob(M_yes_p)

    edge_k, kelly_k = edge_and_kelly(p_yes_k, float(M_yes_k) if imp_k else None)
    edge_p, kelly_p = edge_and_kelly(p_yes_p, float(M_yes_p) if imp_p else None)

    st.write(f"Kills≥K: implied p≈{imp_k:.2f} | model p≈{p_yes_k:.2f} | edge≈{(edge_k if edge_k is not None else 0):+.2f} | Kelly≈{kelly_k:.2f}")
    st.write(f"Place≤P: implied p≈{imp_p:.2f} | model p≈{p_yes_p:.2f} | edge≈{(edge_p if edge_p is not None else 0):+.2f} | Kelly≈{kelly_p:.2f}")

st.divider()
st.subheader("Data")

c1, c2 = st.columns(2)
with c1:
    st.dataframe(df.tail(50), use_container_width=True)
with c2:
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="matches.csv")
    up = st.file_uploader("Restore/Append CSV", type=["csv"])
    if up is not None:
        try:
            newdf = pd.read_csv(up)
            df = pd.concat([df, newdf], ignore_index=True).drop_duplicates()
            df.to_csv(CSV, index=False)
            st.success("CSV merged.")
        except Exception as e:
            st.error(f"Upload failed: {e}")
