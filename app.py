import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression

st.set_page_config(page_title="Twitch Predictor - Fortnite", layout="centered")
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

def save_df(dframe: pd.DataFrame):
    dframe.to_csv(CSV, index=False)

def add_row(d):
    global df
    df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)
    save_df(df)

def hist_feats(hist, K, P):
    if len(hist) == 0:
        return None
    last = hist.tail(min(20, len(hist)))
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
    p_train = pipe.predict_proba(X)[:,1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_train, y)
    return pipe, iso

def predict_proba(model, iso, Xrow, fallback):
    if model is None or iso is None:
        return fallback
    p = model.predict_proba(Xrow)[:,1][0]
    return float(iso.transform([p])[0])

def get_confidence_level(prob):
    confidence = abs(prob - 0.5) * 2  # 0..1
    if confidence >= 0.6:   return "Very High", confidence
    if confidence >= 0.4:   return "High", confidence
    if confidence >= 0.2:   return "Moderate", confidence
    return "Low", confidence

def make_prediction(p_model):
    conf_level, conf_score = get_confidence_level(p_model)
    if p_model > 0.55: return "YES", conf_score, conf_level
    if p_model < 0.45: return "NO",  conf_score, conf_level
    return "UNCERTAIN", conf_score, "Too Close"

# ===================== UI =====================
st.title("🎮 Twitch Predictor - Fortnite Edition")

with st.expander("ℹ️ About This Predictor"):
    st.markdown("""
    Uses **Logistic Regression** with **Isotonic Calibration**. Needs ~25+ matches to start, 50–100 for stability.
    Trains on: event type, team mode, tournament progress, and rolling 20-match stats.
    """)

# -------- Section 1: Add Match --------
st.header("📊 Add Match Data")
with st.form("add_match"):
    c1, c2 = st.columns(2)
    event_type = c1.selectbox("Event type", EVENT_TYPES, index=0)
    team_mode  = c2.selectbox("Team mode", TEAM_MODES, index=0)

    c3, c4 = st.columns(2)
    match_num      = c3.number_input("Match # in tournament", min_value=1, step=1)
    total_elims_t  = c4.number_input("Total tournament elims so far", min_value=0, step=1)

    st.markdown("### Match Results")
    c5, c6 = st.columns(2)
    kills     = c5.number_input("Eliminations this match", min_value=0, step=1)
    placement = c6.number_input("Final placement (1 = win)", min_value=1, step=1)

    st.markdown("### Prediction Thresholds (for training labels)")
    c7, c8 = st.columns(2)
    K = c7.number_input("Elim threshold (K)", min_value=1, value=5, step=1)
    P = c8.number_input("Placement threshold (P)", min_value=1, value=10, step=1)

    submitted = st.form_submit_button("Save Match", type="primary")
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
        st.success("✅ Match saved")
        st.rerun()

# Data status
if len(df) < 25: st.warning(f"Need {25-len(df)} more matches to start model predictions. ({len(df)}/25)")
elif len(df) < 50: st.info(f"Model training. Add {50-len(df)} more for better accuracy. ({len(df)})")
else: st.success(f"Model trained on {len(df)} matches.")

st.divider()

# -------- Train models --------
Xk, yk, Xp, yp = make_training(df)
mk, ck = fit_model(Xk, yk)  # kills ≥ K
mp, cp = fit_model(Xp, yp)  # placement ≤ P

# -------- Section 2: Current Prediction --------
st.header("🎯 Make a Prediction")

pred_type = st.radio("What are you predicting?", ["Eliminations", "Placement"], horizontal=True)

if pred_type == "Eliminations":
    threshold = st.number_input("🎯 Minimum eliminations (YES if ≥ K)", min_value=1, value=5, step=1)
    st.info(f"Predicting: {threshold}+ eliminations?")
else:
    threshold = st.number_input("🏆 Maximum placement (YES if ≤ P)", min_value=1, value=10, step=1)
    st.info(f"Predicting: top {threshold}?")

# --- Meta to evaluate (explicit, not reusing last row) ---
m1, m2 = st.columns(2)
event_type_eval = m1.selectbox("Event type (eval)", EVENT_TYPES, index=0, key="eval_event")
team_mode_eval  = m2.selectbox("Team mode (eval)", TEAM_MODES, index=0, key="eval_team")
m3, m4 = st.columns(2)
match_num_eval  = m3.number_input("Match # in tournament (eval)", min_value=1, step=1, value=1, key="eval_match")
tot_elims_eval  = m4.number_input("Total tourney elims so far (eval)", min_value=0, step=1, value=0, key="eval_tot")

def current_Xrow_with_meta(evalK, evalP):
    hist = df
    f = hist_feats(hist, evalK, evalP)
    if f is None: return None
    meta = pd.DataFrame([{
        "event_type": event_type_eval,
        "team_mode":  team_mode_eval,
        "match_number_in_tourney": int(match_num_eval),
        "total_elims_in_tourney_so_far": int(tot_elims_eval),
    }])
    return pd.concat([meta.reset_index(drop=True), f.reset_index(drop=True)], axis=1)

Xrow = current_Xrow_with_meta(
    threshold if pred_type == "Eliminations" else (int(df["K"].iloc[-1]) if len(df) else threshold),
    threshold if pred_type == "Placement"    else (int(df["P"].iloc[-1]) if len(df) else threshold),
)

if Xrow is None or ((mk is None) and pred_type=="Eliminations") or ((mp is None) and pred_type=="Placement"):
    st.info("Add ≥25 matches for model predictions. Until then, use historical rate.")
else:
    last = df.tail(min(20, len(df)))
    fb_k = (last["kills"] >= threshold).mean() if len(last)>0 else 0.5
    fb_p = (last["placement"] <= threshold).mean() if len(last)>0 else 0.5

    if pred_type == "Eliminations":
        p_model = predict_proba(mk, ck, Xrow, fb_k)
    else:
        p_model = predict_proba(mp, cp, Xrow, fb_p)

    pred_label, conf_score, conf_level = make_prediction(p_model)

    st.markdown("---")
    if pred_label == "YES":   st.success("# ✅ Predict: **YES**")
    elif pred_label == "NO":  st.error("# ❌ Predict: **NO**")
    else:                     st.warning("# ⚠️ **Too Close to Call**")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Probability", f"{p_model:.1%}")
        st.progress(p_model)
    with c2:
        st.metric("Confidence", f"{conf_score*100:.0f}%")
        st.caption(conf_level)
    with c3:
        if pred_type == "Eliminations":
            hist_rate = (last["kills"] >= threshold).mean() if len(last)>0 else np.nan
        else:
            hist_rate = (last["placement"] <= threshold).mean() if len(last)>0 else np.nan
        hr = 0.5 if pd.isna(hist_rate) else hist_rate
        st.metric("Historical Rate", f"{hr:.1%}")
        st.caption(f"Last {len(last)} matches")

    with st.expander("📊 Detailed Analysis"):
        if pred_type == "Eliminations":
            avg_kills = last["kills"].mean() if len(last) > 0 else 0
            std_kills = last["kills"].std() if len(last) > 1 else 0
            st.write(f"- Avg elims (last 20): {avg_kills:.1f}")
            st.write(f"- Std dev: {std_kills:.1f}")
            st.write(f"- Hits of {threshold}+: {(last['kills'] >= threshold).sum()}/{len(last)}")
            st.write(f"- Max elims: {last['kills'].max() if len(last) > 0 else 0}")
        else:
            avg_place = last["placement"].mean() if len(last) > 0 else 0
            st.write(f"- Avg placement (last 20): {avg_place:.1f}")
            st.write(f"- Top {threshold} hits: {(last['placement'] <= threshold).sum()}/{len(last)}")
            st.write(f"- Best placement: {last['placement'].min() if len(last) > 0 else 'N/A'}")
            st.write(f"- Victory Royales: {(last['placement'] == 1).sum()}")

st.divider()

# -------- Section 3: Data & Analytics --------
st.header("📂 Data & Analytics")
tab_stats, tab_data, tab_graphs = st.tabs(["📊 Statistics", "📋 Data Table", "📈 Performance Graphs"])

with tab_stats:
    if len(df) > 0:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Overall")
            st.metric("Total Matches", len(df))
            st.metric("Avg Eliminations", f"{df['kills'].mean():.1f}")
            st.metric("Avg Placement", f"{df['placement'].mean():.1f}")
        with c2:
            st.subheader("Bests")
            st.metric("Max Eliminations", f"{df['kills'].max()}")
            st.metric("Victory Royales", f"{(df['placement'] == 1).sum()}")
            st.metric("Top 10 Rate", f"{(df['placement'] <= 10).mean():.1%}")
        st.subheader("By Event Type")
        st.dataframe(
            df.groupby('event_type').agg({'kills':'mean','placement':'mean'}).round(1),
            use_container_width=True
        )
    else:
        st.info("Add match data to see statistics.")

with tab_data:
    if len(df) == 0:
        st.info("No matches recorded yet.")
    else:
        df_display = df.copy().reset_index().rename(columns={"index": "_row_id"})
        df_display["_delete"] = False
        edited = st.data_editor(
            df_display,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "_row_id": st.column_config.NumberColumn("id", disabled=True),
                "_delete": st.column_config.CheckboxColumn("delete?")
            },
            hide_index=True,
            key="editor"
        )
        c1, c2, c3 = st.columns([1,1,1])
        save_clicked   = c1.button("💾 Save edits")
        delete_clicked = c2.button("🗑️ Delete selected")
        csv_data = df.to_csv(index=False)
        c3.download_button("📥 Download CSV", data=csv_data,
                           file_name=f"fortnite_matches_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv")

        if save_clicked:
            cols_keep = [c for c in edited.columns if c not in ["_row_id","_delete"]]
            new_df = edited[cols_keep].copy()
            # enforce integer cols
            int_cols = ["match_number_in_tourney","total_elims_in_tourney_so_far","kills","placement","K","P"]
            for c in int_cols:
                if c in new_df.columns:
                    new_df[c] = pd.to_numeric(new_df[c], errors="coerce").fillna(0).astype(int)
            save_df(new_df)
            df = new_df
            st.success("Edits saved.")
            st.rerun()

        if delete_clicked:
            to_drop_ids = edited.loc[edited["_delete"]==True, "_row_id"].tolist()
            if to_drop_ids:
                df = df.drop(index=to_drop_ids).reset_index(drop=True)
                save_df(df)
                st.success(f"Deleted {len(to_drop_ids)} rows.")
                st.rerun()
            else:
                st.info("No rows selected for deletion.")

        st.markdown("---")
        up = st.file_uploader("📤 Import CSV", type=["csv"])
        if up is not None:
            try:
                newdf = pd.read_csv(up)
                df = pd.concat([df, newdf], ignore_index=True).drop_duplicates()
                save_df(df)
                st.success("✅ Data imported")
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")

with tab_graphs:
    if len(df) >= 2:
        df_plot = df.copy()
        df_plot["match_num"] = range(1, len(df_plot) + 1)
        st.subheader("🎯 Eliminations Trend")
        st.line_chart(df_plot.set_index("match_num")["kills"], height=300)
        st.subheader("🏆 Placement Trend")
        st.line_chart(df_plot.set_index("match_num")["placement"], height=300)
        if len(df_plot) >= 5:
            st.subheader("📊 Rolling Performance (20-match window)")
            c1, c2 = st.columns(2)
            viz_k = c1.slider("Elim threshold for graph", 1, 15, 5)
            viz_p = c2.slider("Placement threshold for graph", 1, 50, 10)
            win_k = (df_plot["kills"] >= viz_k).rolling(20, min_periods=1).mean()
            win_p = (df_plot["placement"] <= viz_p).rolling(20, min_periods=1).mean()
            perf = pd.DataFrame({f"{viz_k}+ Elim Rate": win_k.values, f"Top {viz_p} Rate": win_p.values},
                                index=df_plot["match_num"])
            st.line_chart(perf, height=300)
    else:
        st.info("Graphs appear after ≥2 matches.")
