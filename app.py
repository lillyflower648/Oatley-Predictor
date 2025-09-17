import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression

st.set_page_config(page_title="Twitch Predictor", layout="centered")
CSV = Path("matches.csv")

# -------------------- Storage --------------------
cols = [
    "event_type","team_mode","match_number_in_tourney","total_elims_in_tourney_so_far",
    "kills","placement","K","P","ts"
]
if CSV.exists():
    df = pd.read_csv(CSV)
else:
    df = pd.DataFrame(columns=cols)

# -------------------- Helpers --------------------
EVENT_TYPES = ["scrims","fncs","console","cash","skin_cup"]
TEAM_MODES  = ["solo","duo","trio"]

def add_row(d):
    global df
    df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)
    df.to_csv(CSV, index=False)

def hist_feats(hist, K, P):
    if len(hist) == 0:
        return None
    last = hist.tail(min(20, len(hist)))  # small, stable window
    return pd.DataFrame([{
        "kills_avg": last["kills"].mean(),
        "kills_std": last["kills"].std(ddof=0) if len(last)>1 else 0.0,
        "place_avg": last["placement"].mean(),
        "place_std": last["placement"].std(ddof=0) if len(last)>1 else 0.0,
        "hit_rate_k": (last["kills"] >= K).mean(),
        "hit_rate_p": (last["placement"] <= P).mean(),
        "K": int(K), "P": int(P)
    }])

def make_training(df_all):
    rows_k, rows_p = [], []
    for i in range(len(df_all)):
        hist = df_all.iloc[:i]
        if len(hist) == 0: 
            continue
        K = int(df_all.iloc[i]["K"])
        P = int(df_all.iloc[i]["P"])
        f_basic = hist_feats(hist, K, P)
        if f_basic is None: 
            continue
        # only your requested fields as inputs
        meta = df_all.iloc[i][["event_type","team_mode","match_number_in_tourney","total_elims_in_tourney_so_far"]].to_frame().T
        Xrow = pd.concat([meta.reset_index(drop=True), f_basic.reset_index(drop=True)], axis=1)
        yk = int(df_all.iloc[i]["kills"] >= K)
        yp = int(df_all.iloc[i]["placement"] <= P)
        rows_k.append((Xrow, yk))
        rows_p.append((Xrow, yp))
    if rows_k:
        Xk = pd.concat([x for x,_ in rows_k], ignore_index=True); yk = np.array([y for _,y in rows_k])
    else:
        Xk, yk = None, None
    if rows_p:
        Xp = pd.concat([x for x,_ in rows_p], ignore_index=True); yp = np.array([y for _,y in rows_p])
    else:
        Xp, yp = None, None
    return Xk, yk, Xp, yp

def fit_model(X, y):
    if X is None or y is None or len(X) < 25 or len(np.unique(y)) < 2:
        return None, None
    cat = ["event_type","team_mode"]
    num = ["match_number_in_tourney","total_elims_in_tourney_so_far",
           "kills_avg","kills_std","place_avg","place_std","hit_rate_k","hit_rate_p","K","P"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num)
    ])
    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)
    # calibration
    p_train = pipe.predict_proba(X)[:,1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_train, y)
    return pipe, iso

def predict_proba(model, iso, Xrow, fallback):
    if model is None or iso is None:
        return fallback
    p = model.predict_proba(Xrow)[:,1][0]
    return float(iso.transform([p])[0])

def implied_from_mult(M_yes, M_no):
    # Approx pari-mutuel implied probs; allow blanks
    py = 1.0/float(M_yes) if M_yes and float(M_yes)>0 else None
    pn = 1.0/float(M_no)  if M_no  and float(M_no)>0  else None
    return py, pn

def recommend_side(p_model, M_yes, M_no):
    py, pn = implied_from_mult(M_yes, M_no)
    if py and pn:
        edge_yes = p_model - py
        edge_no  = (1-p_model) - pn
        if edge_yes>0 or edge_no>0:
            return ("YES" if edge_yes>=edge_no else "NO", edge_yes, edge_no, py, pn)
        return ("PASS", edge_yes, edge_no, py, pn)
    # no multipliers: simple threshold at 0.5
    if p_model > 0.5: 
        return ("YES", None, None, None, None)
    if p_model < 0.5:
        return ("NO", None, None, None, None)
    return ("PASS", None, None, None, None)

# ===================== UI =====================

st.title("Twitch Prediction Helper")

# -------- Section 1: Add Match --------
st.header("Add Match")
with st.form("add_match"):
    c1, c2 = st.columns(2)
    event_type = c1.selectbox("Event type", EVENT_TYPES, index=0)
    team_mode  = c2.selectbox("Team mode", TEAM_MODES, index=0)

    c3, c4 = st.columns(2)
    match_num = c3.number_input("Match number in tournament", min_value=1, step=1)
    total_elims_t = c4.number_input("Total elims in tournament so far", min_value=0, step=1)

    c5, c6 = st.columns(2)
    kills = c5.number_input("Final kills (this match)", min_value=0, step=1)
    placement = c6.number_input("Final placement (1 = win)", min_value=1, step=1)

    c7, c8 = st.columns(2)
    K = c7.number_input("Your K threshold (for elim predictions)", min_value=1, value=5, step=1)
    P = c8.number_input("Your P threshold (for placement predictions)", min_value=1, value=10, step=1)

    submitted = st.form_submit_button("Save match")
    if submitted:
        add_row({
            "event_type": event_type,
            "team_mode": team_mode,
            "match_number_in_tourney": int(match_num),
            "total_elims_in_tourney_so_far": int(total_elims_t),
            "kills": int(kills),
            "placement": int(placement),
            "K": int(K),
            "P": int(P),
            "ts": pd.Timestamp.now().isoformat(timespec="seconds")
        })
        st.success("Saved.")

st.divider()

# -------- Train models (on current data) --------
Xk, yk, Xp, yp = make_training(df)
mk, ck = fit_model(Xk, yk)  # kills>=K
mp, cp = fit_model(Xp, yp)  # placement<=P

# Build current feature row from all history
def current_Xrow(evalK, evalP):
    if len(df)==0: 
        return None
    f = hist_feats(df, evalK, evalP)
    if f is None: 
        return None
    # meta inputs must be provided for evaluation; reuse last known meta
    last_meta = df.iloc[[-1]][["event_type","team_mode","match_number_in_tourney","total_elims_in_tourney_so_far"]].reset_index(drop=True)
    return pd.concat([last_meta, f], axis=1)

# -------- Section 2: Current Prediction --------
st.header("Current Prediction")
c1, c2 = st.columns(2)
pred_type = c1.selectbox("Prediction type", ["eliminations","placement"], index=0)
threshold = c2.number_input("Threshold (K for elims, P for placement)", min_value=1, value=5, step=1)

d1, d2 = st.columns(2)
M_yes = d1.text_input("YES multiplier (optional)", "")
M_no  = d2.text_input("NO multiplier (optional)", "")

Xrow = current_Xrow(threshold if pred_type=="eliminations" else df["K"].iloc[-1] if len(df) else threshold,
                    threshold if pred_type=="placement"    else df["P"].iloc[-1] if len(df) else threshold)

if Xrow is None:
    st.info("Add at least one match to evaluate.")
else:
    # Fallbacks from last 20 matches
    last = df.tail(min(20, len(df)))
    fb_k = (last["kills"]    >= threshold).mean() if len(last)>0 else 0.5
    fb_p = (last["placement"]<=  threshold).mean() if len(last)>0 else 0.5

    if pred_type=="eliminations":
        p_model = predict_proba(mk, ck, Xrow, fb_k)
        st.write(f"Model Pr(kills ≥ {int(threshold)}) ≈ **{p_model:.2f}**")
    else:
        p_model = predict_proba(mp, cp, Xrow, fb_p)
        st.write(f"Model Pr(place ≤ {int(threshold)}) ≈ **{p_model:.2f}**")

    side, edge_yes, edge_no, py, pn = recommend_side(p_model, M_yes, M_no)
    st.subheader(f"Recommendation: **{side}**")
    if py is not None and pn is not None:
        st.caption(f"Implied p(YES)≈{py:.2f}, p(NO)≈{pn:.2f}. Edge(YES)≈{edge_yes:+.2f}, Edge(NO)≈{edge_no:+.2f}.")
    else:
        st.caption("No multipliers provided. Using 0.5 threshold heuristic.")

st.divider()

# -------- Section 3: Data --------
st.header("Data")
c1, c2 = st.columns(2)
with c1:
    st.dataframe(df.tail(100), use_container_width=True)
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

st.subheader("Graphs")
if len(df) >= 1:
    df_plot = df.copy()
    df_plot["idx"] = range(1, len(df_plot)+1)
    kc = st.checkbox("Show kills over matches", value=True)
    pc = st.checkbox("Show placement over matches", value=True)
    hr = st.checkbox("Show rolling hit rates (last 20)", value=True)

    if kc:
        st.line_chart(df_plot.set_index("idx")["kills"])
    if pc:
        st.line_chart(df_plot.set_index("idx")["placement"])
    if hr:
        if len(df_plot) >= 1:
            K_current = int(df_plot["K"].iloc[-1]) if len(df_plot) else 5
            P_current = int(df_plot["P"].iloc[-1]) if len(df_plot) else 10
            win_k = (df_plot["kills"] >= K_current).rolling(20, min_periods=1).mean()
            win_p = (df_plot["placement"] <= P_current).rolling(20, min_periods=1).mean()
            st.line_chart(pd.DataFrame({"hit_rate_k": win_k, "hit_rate_p": win_p}))
else:
    st.info("Graphs appear after you add data.")
