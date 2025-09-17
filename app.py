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

CSV = Path("matches.csv")           # training data
TEST_CSV = Path("test_matches.csv") # test/holdout data

# -------------------- Storage --------------------
TRAIN_COLS = ["event_type","team_mode","match_number_in_tourney","kills","placement","ts"]
TEST_COLS  = ["event_type","team_mode","match_number_in_tourney","kills","placement","ts"]

def load_df(path: Path, cols: list):
    if path.exists():
        df = pd.read_csv(path)
        missing = [c for c in cols if c not in df.columns]
        if missing:  # migrate if schema mismatch
            return pd.DataFrame(columns=cols)
        return df
    return pd.DataFrame(columns=cols)

df = load_df(CSV, TRAIN_COLS)
test_df = load_df(TEST_CSV, TEST_COLS)

def save_df(dframe: pd.DataFrame):
    dframe.to_csv(CSV, index=False)

def save_test_df(dframe: pd.DataFrame):
    dframe.to_csv(TEST_CSV, index=False)

# -------------------- Helpers --------------------
EVENT_TYPES = ["scrims","fncs","console","cash","skin_cup"]
TEAM_MODES  = ["solo","duo","trio"]

def add_row(d):
    global df
    df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)
    save_df(df)

def add_test_row(d):
    global test_df
    test_df = pd.concat([test_df, pd.DataFrame([d])], ignore_index=True)
    save_test_df(test_df)

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

def make_training(df_all, evalK, evalP):
    rows_k, rows_p = [], []
    for i in range(len(df_all)):
        hist = df_all.iloc[:i]
        if len(hist) == 0:
            continue
        f_basic = hist_feats(hist, evalK, evalP)
        if f_basic is None:
            continue
        meta = df_all.iloc[i][["event_type","team_mode","match_number_in_tourney"]].to_frame().T
        Xrow = pd.concat([meta.reset_index(drop=True), f_basic.reset_index(drop=True)], axis=1)
        yk = int(df_all.iloc[i]["kills"] >= evalK)
        yp = int(df_all.iloc[i]["placement"] <= evalP)
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
    num = ["match_number_in_tourney","kills_avg","kills_std","place_avg","place_std","hit_rate_k","hit_rate_p","K","P"]
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

with st.expander("ℹ️ Notes"):
    st.markdown(
        "- Model: Logistic Regression + Isotonic calibration.\n"
        "- Needs ~25+ training matches. 50–100 recommended.\n"
        "- Thresholds are chosen at prediction time and used for training labels on the fly."
    )

# -------- Section 1: Add Match (TRAIN) --------
st.header("📊 Add Match Data (Training)")
with st.form("add_match"):
    c1, c2 = st.columns(2)
    event_type = c1.selectbox("Event type", EVENT_TYPES, index=0)
    team_mode  = c2.selectbox("Team mode", TEAM_MODES, index=0)

    c3, c4 = st.columns(2)
    match_num = c3.number_input("Match # in tournament", min_value=1, step=1)
    placement = c4.number_input("Final placement (1 = win)", min_value=1, step=1)
    kills = st.number_input("Eliminations this match", min_value=0, step=1)

    submitted = st.form_submit_button("Save Training Match", type="primary")
    if submitted:
        add_row({
            "event_type": event_type,
            "team_mode": team_mode,
            "match_number_in_tourney": int(match_num),
            "kills": int(kills),
            "placement": int(placement),
            "ts": pd.Timestamp.now().isoformat(timespec="seconds")
        })
        st.success("Saved to training set.")
        st.rerun()

# Data status
if len(df) < 25: st.warning(f"{25-len(df)} more training matches needed. ({len(df)}/25)")
elif len(df) < 50: st.info(f"Training size: {len(df)}. More data improves stability.")
else: st.success(f"Training size: {len(df)}")

st.divider()

# -------- Section 2: Current Prediction --------
st.header("🎯 Make a Prediction")

pred_type = st.radio("What are you predicting?", ["Eliminations", "Placement"], horizontal=True)
if pred_type == "Eliminations":
    threshold = st.number_input("Minimum eliminations (YES if ≥ K)", min_value=1, value=5, step=1)
    st.caption(f"Evaluating probability of {threshold}+ elims.")
else:
    threshold = st.number_input("Maximum placement (YES if ≤ P)", min_value=1, value=10, step=1)
    st.caption(f"Evaluating probability of top {threshold}.")

m1, m2 = st.columns(2)
event_type_eval = m1.selectbox("Event type (eval)", EVENT_TYPES, index=0, key="eval_event")
team_mode_eval  = m2.selectbox("Team mode (eval)", TEAM_MODES, index=0, key="eval_team")
match_num_eval  = st.number_input("Match # in tournament (eval)", min_value=1, step=1, value=1, key="eval_match")

evalK = threshold
evalP = threshold

# train models for this threshold
Xk, yk, Xp, yp = make_training(df, evalK, evalP)
mk, ck = fit_model(Xk, yk)  # kills≥K
mp, cp = fit_model(Xp, yp)  # place≤P

def current_Xrow_with_meta(evalK, evalP):
    f = hist_feats(df, evalK, evalP)
    if f is None: return None
    meta = pd.DataFrame([{
        "event_type": event_type_eval,
        "team_mode":  team_mode_eval,
        "match_number_in_tourney": int(match_num_eval)
    }])
    return pd.concat([meta.reset_index(drop=True), f.reset_index(drop=True)], axis=1)

Xrow = current_Xrow_with_meta(evalK, evalP)

if Xrow is None:
    st.info("Add training matches first.")
else:
    last = df.tail(min(20, len(df)))
    fb_k = (last["kills"] >= threshold).mean() if len(last)>0 else 0.5
    fb_p = (last["placement"] <= threshold).mean() if len(last)>0 else 0.5
    p_model = predict_proba(mk, ck, Xrow, fb_k) if pred_type=="Eliminations" else predict_proba(mp, cp, Xrow, fb_p)
    label, conf_score, conf_level = make_prediction(p_model)

    st.markdown("---")
    if label == "YES":   st.success("# ✅ Predict: **YES**")
    elif label == "NO":  st.error("# ❌ Predict: **NO**")
    else:                st.warning("# ⚠️ **Too Close to Call**")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Probability", f"{p_model:.1%}")
        st.progress(p_model)
    with c2:
        st.metric("Confidence", f"{conf_score*100:.0f}%")
        st.caption(conf_level)
    with c3:
        hist_rate = (last["kills"] >= threshold).mean() if pred_type=="Eliminations" else (last["placement"] <= threshold).mean()
        hr = 0.5 if pd.isna(hist_rate) else hist_rate
        st.metric("Historical Rate", f"{hr:.1%}")
        st.caption(f"Last {len(last)} training matches")

st.divider()

# -------- Section 3: Data & Analytics --------
st.header("📂 Data & Analytics")
tab_stats, tab_data, tab_graphs, tab_test = st.tabs(["📊 Statistics", "📋 Data Table", "📈 Performance Graphs", "🧪 Test"])

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
        st.info("Add training data to see statistics.")

with tab_data:
    if len(df) == 0:
        st.info("No training matches recorded yet.")
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
            key="editor_train"
        )
        c1, c2, c3 = st.columns([1,1,1])
        if c1.button("💾 Save edits"):
            cols_keep = [c for c in edited.columns if c not in ["_row_id","_delete"]]
            new_df = edited[cols_keep].copy()
            for c in ["match_number_in_tourney","kills","placement"]:
                if c in new_df.columns:
                    new_df[c] = pd.to_numeric(new_df[c], errors="coerce").fillna(0).astype(int)
            save_df(new_df); df = new_df
            st.success("Training edits saved."); st.rerun()
        if c2.button("🗑️ Delete selected"):
            to_drop_ids = edited.loc[edited["_delete"]==True, "_row_id"].tolist()
            if to_drop_ids:
                df = df.drop(index=to_drop_ids).reset_index(drop=True)
                save_df(df); st.success(f"Deleted {len(to_drop_ids)} rows."); st.rerun()
            else:
                st.info("No rows selected.")
        c3.download_button("📥 Download training CSV",
                           data=df.to_csv(index=False),
                           file_name=f"fortnite_training_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv")
        st.markdown("---")
        up = st.file_uploader("📤 Import training CSV", type=["csv"], key="up_train")
        if up is not None:
            try:
                newdf = pd.read_csv(up)
                df = pd.concat([df, newdf], ignore_index=True).drop_duplicates()
                save_df(df); st.success("Training data imported."); st.rerun()
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
        st.info("Graphs appear after ≥2 training matches.")

# -------------------- Test Tab --------------------
with tab_test:
    st.subheader("🧪 Holdout / Test Set")
    st.caption("Add known matches here. They are NOT used for training.")

    with st.form("add_test_match"):
        c1, c2 = st.columns(2)
        t_event = c1.selectbox("Event type (test)", EVENT_TYPES, index=0)
        t_team  = c2.selectbox("Team mode (test)", TEAM_MODES, index=0)
        c3, c4 = st.columns(2)
        t_match = c3.number_input("Match # in tournament (test)", min_value=1, step=1)
        t_place = c4.number_input("Final placement (test)", min_value=1, step=1)
        t_kills = st.number_input("Eliminations (test)", min_value=0, step=1)
        t_submit = st.form_submit_button("Add to Test Set")
        if t_submit:
            add_test_row({
                "event_type": t_event,
                "team_mode": t_team,
                "match_number_in_tourney": int(t_match),
                "kills": int(t_kills),
                "placement": int(t_place),
                "ts": pd.Timestamp.now().isoformat(timespec="seconds")
            })
            st.success("Saved to test set."); st.rerun()

    # editable test table
    if len(test_df) == 0:
        st.info("No test matches yet.")
    else:
        t_disp = test_df.copy().reset_index().rename(columns={"index": "_row_id"})
        t_disp["_delete"] = False
        t_ed = st.data_editor(
            t_disp,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "_row_id": st.column_config.NumberColumn("id", disabled=True),
                "_delete": st.column_config.CheckboxColumn("delete?")
            },
            hide_index=True,
            key="editor_test"
        )
        c1, c2, c3 = st.columns([1,1,1])
        if c1.button("💾 Save test edits"):
            cols_keep = [c for c in t_ed.columns if c not in ["_row_id","_delete"]]
            new_tdf = t_ed[cols_keep].copy()
            for c in ["match_number_in_tourney","kills","placement"]:
                if c in new_tdf.columns:
                    new_tdf[c] = pd.to_numeric(new_tdf[c], errors="coerce").fillna(0).astype(int)
            save_test_df(new_tdf); test_df = new_tdf
            st.success("Test edits saved."); st.rerun()
        if c2.button("🗑️ Delete selected tests"):
            to_drop_ids = t_ed.loc[t_ed["_delete"]==True, "_row_id"].tolist()
            if to_drop_ids:
                test_df = test_df.drop(index=to_drop_ids).reset_index(drop=True)
                save_test_df(test_df); st.success(f"Deleted {len(to_drop_ids)} test rows."); st.rerun()
            else:
                st.info("No rows selected.")
        c3.download_button("📥 Download TEST CSV",
                           data=test_df.to_csv(index=False),
                           file_name=f"fortnite_test_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv")

        st.markdown("---")
        up_test = st.file_uploader("📤 Import TEST CSV", type=["csv"], key="up_test")
        if up_test is not None:
            try:
                newtdf = pd.read_csv(up_test)
                test_df = pd.concat([test_df, newtdf], ignore_index=True).drop_duplicates()
                save_test_df(test_df); st.success("Test data imported."); st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")

        st.markdown("### ▶️ Evaluate on Test Set")
        e1, e2 = st.columns(2)
        test_type = e1.selectbox("Prediction type for evaluation", ["Eliminations","Placement"], index=0)
        test_threshold = e2.number_input("Threshold (K for elims, P for placement)", min_value=1, value=5, step=1)

        # Train models for chosen threshold using TRAIN df only
        tK = tP = test_threshold
        Xk_e, yk_e, Xp_e, yp_e = make_training(df, tK, tP)
        mk_e, ck_e = fit_model(Xk_e, yk_e)
        mp_e, cp_e = fit_model(Xp_e, yp_e)

        if (test_type=="Eliminations" and mk_e is None) or (test_type=="Placement" and mp_e is None):
            st.info("Need ≥25 training matches with label variation before evaluation.")
        else:
            # build feature baseline from TRAIN history once
            f_hist = hist_feats(df, tK, tP)
            if f_hist is None:
                st.info("Add training matches first.")
            else:
                # per-test-row meta + same history features
                probs, y_true = [], []
                meta_rows = []
                for _, r in test_df.iterrows():
                    meta = pd.DataFrame([{
                        "event_type": r["event_type"],
                        "team_mode":  r["team_mode"],
                        "match_number_in_tourney": int(r["match_number_in_tourney"])
                    }])
                    Xrow_t = pd.concat([meta.reset_index(drop=True), f_hist.reset_index(drop=True)], axis=1)
                    if test_type=="Eliminations":
                        p = predict_proba(mk_e, ck_e, Xrow_t, fallback=0.5)
                        y = int(r["kills"] >= tK)
                    else:
                        p = predict_proba(mp_e, cp_e, Xrow_t, fallback=0.5)
                        y = int(r["placement"] <= tP)
                    probs.append(p); y_true.append(y)
                    meta_rows.append(meta.iloc[0].to_dict())

                res = test_df.copy()
                res["pred_prob"] = probs
                res["y_true"] = y_true
                res["y_hat"] = (res["pred_prob"] >= 0.5).astype(int)

                acc = (res["y_hat"] == res["y_true"]).mean() if len(res)>0 else np.nan
                brier = np.mean((res["pred_prob"] - res["y_true"])**2) if len(res)>0 else np.nan

                st.metric("Accuracy", f"{(acc*100):.1f}%" if pd.notna(acc) else "N/A")
                st.metric("Brier score", f"{brier:.3f}" if pd.notna(brier) else "N/A")
                st.dataframe(res, use_container_width=True)

                st.download_button("📥 Download TEST RESULTS CSV",
                    data=res.to_csv(index=False),
                    file_name=f"fortnite_test_results_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv")
