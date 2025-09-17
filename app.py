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

# ---------------- Paths ----------------
CSV = Path("matches.csv")            # training data
TEST_CSV = Path("test_matches.csv")  # holdout/test data

# ---------------- Schemas ----------------
TRAIN_COLS = ["event_type","team_mode","match_number_in_tourney","kills","placement","ts"]
TEST_COLS  = [
    "_tid",                          # stable row id for deletes
    "event_type","team_mode","match_number_in_tourney",
    "pred_type","threshold",         # prediction setup
    "kills","placement",             # actual outcomes
    "user_pick",                     # optional: YES/NO you picked
    "ts"
]

EVENT_TYPES = ["scrims","fncs","console","cash","skin_cup"]
TEAM_MODES  = ["solo","duo","trio"]

# ---------------- Storage helpers ----------------
def load_df(path: Path, cols: list, gen_id=False, id_col="_tid"):
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=cols)

    # Backfill missing columns
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")

    # Generate id column if requested
    if gen_id:
        if id_col not in df.columns or df[id_col].isna().any():
            df[id_col] = list(range(1, len(df)+1))
        df[id_col] = df[id_col].astype(int)

    # Keep column order
    return df[cols]

def save_df(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)

df = load_df(CSV, TRAIN_COLS, gen_id=False)
test_df = load_df(TEST_CSV, TEST_COLS, gen_id=True)

def next_tid():
    return 1 if len(test_df)==0 else int(test_df["_tid"].max())+1

# ---------------- Feature + model ----------------
def hist_feats(hist: pd.DataFrame, K: int, P: int):
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

def make_training(df_all: pd.DataFrame, evalK: int, evalP: int):
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

def conf_bucket(prob):
    confidence = abs(prob - 0.5) * 2
    if confidence >= 0.6: return "Very High", confidence
    if confidence >= 0.4: return "High", confidence
    if confidence >= 0.2: return "Moderate", confidence
    return "Low", confidence

# ================= UI =================
st.title("🎮 Twitch Predictor - Fortnite Edition")

with st.expander("ℹ️ Notes"):
    st.markdown(
        "- Model: Logistic Regression + Isotonic calibration.\n"
        "- Needs ~25+ training matches. 50–100 recommended.\n"
        "- Thresholds are chosen at prediction time (no per-row thresholds stored)."
    )

# -------- Section 1: Add Match (TRAIN) --------
st.header("📊 Add Match Data (Training)")
with st.form("add_train"):
    c1, c2 = st.columns(2)
    event_type = c1.selectbox("Event type", EVENT_TYPES, index=0)
    team_mode  = c2.selectbox("Team mode", TEAM_MODES, index=0)
    c3, c4 = st.columns(2)
    match_num  = c3.number_input("Match # in tournament", min_value=1, step=1)
    placement  = c4.number_input("Final placement (1 = win)", min_value=1, step=1)
    kills      = st.number_input("Eliminations this match", min_value=0, step=1)
    if st.form_submit_button("Save Training Match", type="primary"):
        row = {
            "event_type": event_type,
            "team_mode": team_mode,
            "match_number_in_tourney": int(match_num),
            "kills": int(kills),
            "placement": int(placement),
            "ts": pd.Timestamp.now().isoformat(timespec="seconds")
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        save_df(df, CSV)
        st.success("Saved to training set.")
        st.rerun()

# Training size status
if len(df) < 25: st.warning(f"{25-len(df)} more training matches needed. ({len(df)}/25)")
elif len(df) < 50: st.info(f"Training size: {len(df)}. More data improves stability.")
else: st.success(f"Training size: {len(df)}")

st.divider()

# -------- Section 2: Current Prediction --------
st.header("🎯 Make a Prediction")

pred_type = st.radio("What are you predicting?", ["Eliminations", "Placement"], horizontal=True)
threshold = st.number_input("Threshold (K for elims, P for placement)", min_value=1, value=5, step=1)

m1, m2 = st.columns(2)
event_type_eval = m1.selectbox("Event type (eval)", EVENT_TYPES, index=0, key="eval_event")
team_mode_eval  = m2.selectbox("Team mode (eval)", TEAM_MODES, index=0, key="eval_team")
match_num_eval  = st.number_input("Match # in tournament (eval)", min_value=1, step=1, value=1, key="eval_match")

# Train models for this threshold from TRAIN ONLY
evalK = evalP = threshold
Xk, yk, Xp, yp = make_training(df, evalK, evalP)
mk, ck = fit_model(Xk, yk)           # kills ≥ K
mp, cp = fit_model(Xp, yp)           # place ≤ P

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
    label = "YES" if p_model>0.55 else ("NO" if p_model<0.45 else "UNCERTAIN")
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Probability", f"{p_model:.1%}"); st.progress(p_model)
    with c2:
        bucket, score = conf_bucket(p_model)
        st.metric("Confidence", f"{score*100:.0f}%"); st.caption(bucket)
    with c3:
        hist_rate = (last["kills"] >= threshold).mean() if pred_type=="Eliminations" else (last["placement"] <= threshold).mean()
        hr = 0.5 if pd.isna(hist_rate) else hist_rate
        st.metric("Historical Rate", f"{hr:.1%}"); st.caption(f"Last {len(last)}")

st.divider()

# -------- Section 3: Data & Analytics --------
st.header("📂 Data & Analytics")
tab_stats, tab_data, tab_graphs, tab_test = st.tabs(["📊 Statistics", "📋 Data Table", "📈 Performance Graphs", "🧪 Test"])

with tab_stats:
    if len(df) == 0:
        st.info("Add training data to see statistics.")
    else:
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
        st.dataframe(df.groupby('event_type').agg({'kills':'mean','placement':'mean'}).round(1), use_container_width=True)

with tab_data:
    if len(df) == 0:
        st.info("No training matches yet.")
    else:
        disp = df.copy().reset_index().rename(columns={"index":"_row_id"})
        disp["_delete"] = False
        ed = st.data_editor(
            disp, use_container_width=True, num_rows="dynamic",
            column_config={"_row_id": st.column_config.NumberColumn("id", disabled=True),
                           "_delete": st.column_config.CheckboxColumn("delete?")},
            hide_index=True, key="editor_train"
        )
        c1,c2,c3 = st.columns([1,1,1])
        if c1.button("💾 Save edits"):
            keep = [c for c in ed.columns if c not in ["_row_id","_delete"]]
            new_df = ed[keep].copy()
            for c in ["match_number_in_tourney","kills","placement"]:
                if c in new_df.columns:
                    new_df[c] = pd.to_numeric(new_df[c], errors="coerce").fillna(0).astype(int)
            df = new_df; save_df(df, CSV); st.success("Training edits saved."); st.rerun()
        if c2.button("🗑️ Delete selected"):
            to_drop = ed.loc[ed["_delete"]==True, "_row_id"].tolist()
            if to_drop:
                df = df.drop(index=to_drop).reset_index(drop=True)
                save_df(df, CSV); st.success(f"Deleted {len(to_drop)} rows."); st.rerun()
            else:
                st.info("No rows selected.")
        c3.download_button("📥 Download training CSV", data=df.to_csv(index=False),
                           file_name=f"fortnite_training_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        st.markdown("---")
        up = st.file_uploader("📤 Import training CSV", type=["csv"], key="up_train")
        if up is not None:
            try:
                newdf = pd.read_csv(up)
                df = pd.concat([df, newdf], ignore_index=True).drop_duplicates()
                save_df(df, CSV); st.success("Training data imported."); st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")

with tab_graphs:
    if len(df) >= 2:
        plot = df.copy(); plot["match_num"] = range(1, len(plot)+1)
        st.subheader("🎯 Eliminations Trend"); st.line_chart(plot.set_index("match_num")["kills"], height=300)
        st.subheader("🏆 Placement Trend");   st.line_chart(plot.set_index("match_num")["placement"], height=300)
        if len(plot) >= 5:
            st.subheader("📊 Rolling Performance (20-match window)")
            c1,c2 = st.columns(2)
            viz_k = c1.slider("Elim threshold for graph", 1, 15, 5)
            viz_p = c2.slider("Placement threshold for graph", 1, 50, 10)
            win_k = (plot["kills"] >= viz_k).rolling(20, min_periods=1).mean()
            win_p = (plot["placement"] <= viz_p).rolling(20, min_periods=1).mean()
            perf = pd.DataFrame({f"{viz_k}+ Elim Rate": win_k.values, f"Top {viz_p} Rate": win_p.values},
                                index=plot["match_num"])
            st.line_chart(perf, height=300)
    else:
        st.info("Graphs appear after ≥2 training matches.")

# ---------------- Test Tab ----------------
with tab_test:
    st.subheader("🧪 Test Set (holdout)")
    st.caption("Import or add known matches here. They are NOT used for training.")

    # Add single test row
    with st.form("add_test"):
        c1,c2 = st.columns(2)
        t_event = c1.selectbox("Event type (test)", EVENT_TYPES, index=0)
        t_team  = c2.selectbox("Team mode (test)", TEAM_MODES, index=0)
        c3,c4 = st.columns(2)
        t_match = c3.number_input("Match # in tournament (test)", min_value=1, step=1)
        t_ptype = c4.selectbox("Prediction type", ["Eliminations","Placement"])
        c5,c6 = st.columns(2)
        t_thresh = c5.number_input("Threshold (K or P)", min_value=1, value=5, step=1)
        t_place  = c6.number_input("Actual final placement", min_value=1, step=1)
        t_kills  = st.number_input("Actual eliminations", min_value=0, step=1)
        t_pick   = st.selectbox("Your pick (optional)", ["", "YES", "NO"])
        if st.form_submit_button("Add to Test Set"):
            row = {
                "_tid": next_tid(),
                "event_type": t_event, "team_mode": t_team, "match_number_in_tourney": int(t_match),
                "pred_type": t_ptype, "threshold": int(t_thresh),
                "kills": int(t_kills), "placement": int(t_place),
                "user_pick": t_pick, "ts": pd.Timestamp.now().isoformat(timespec="seconds")
            }
            test_df = pd.concat([test_df, pd.DataFrame([row])], ignore_index=True)
            save_df(test_df, TEST_CSV); st.success("Saved test row."); st.rerun()

    # Import/export test CSV
    c1,c2 = st.columns(2)
    c1.download_button(
        "📥 Download TEST CSV",
        data=test_df.to_csv(index=False),
        file_name=f"fortnite_test_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    up_test = c2.file_uploader("📤 Import TEST CSV", type=["csv"], key="up_test")
    if up_test is not None:
        try:
            newt = pd.read_csv(up_test)
            # ensure required cols exist
            for c in TEST_COLS:
                if c not in newt.columns:
                    newt[c] = pd.Series(dtype="object")
            # ensure ids
            if "_tid" not in newt.columns or newt["_tid"].isna().any():
                start = next_tid()
                newt["_tid"] = list(range(start, start+len(newt)))
            newt["_tid"] = newt["_tid"].astype(int)
            # keep canonical order
            newt = newt[TEST_COLS]
            test_df = pd.concat([test_df, newt], ignore_index=True).drop_duplicates(subset=["_tid"])
            save_df(test_df, TEST_CSV); st.success("Imported test CSV."); st.rerun()
        except Exception as e:
            st.error(f"Import failed: {e}")

    # Editable test table with WORKING delete
    if len(test_df)==0:
        st.info("No test rows yet.")
    else:
        t_disp = test_df.copy()
        t_disp["_delete"] = False
        ed = st.data_editor(
            t_disp,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "_tid": st.column_config.NumberColumn("id", disabled=True),
                "_delete": st.column_config.CheckboxColumn("delete?")
            },
            hide_index=True,
            key="editor_test"
        )
        c1,c2 = st.columns(2)
        if c1.button("💾 Save test edits"):
            # Persist edits, keep id
            keep = [c for c in ed.columns if c != "_delete"]
            new_t = ed[keep].copy()
            for c in ["match_number_in_tourney","threshold","kills","placement"]:
                if c in new_t.columns:
                    new_t[c] = pd.to_numeric(new_t[c], errors="coerce").fillna(0).astype(int)
            # Keep column order
            new_t = new_t[TEST_COLS]
            test_df = new_t
            save_df(test_df, TEST_CSV); st.success("Test edits saved."); st.rerun()

        if c2.button("🗑️ Delete selected test rows"):
            ids = ed.loc[ed["_delete"]==True, "_tid"].tolist()
            if ids:
                test_df = test_df[~test_df["_tid"].isin(ids)].reset_index(drop=True)
                save_df(test_df, TEST_CSV); st.success(f"Deleted {len(ids)} test rows."); st.rerun()
            else:
                st.info("No rows selected.")

    st.markdown("---")
    st.subheader("▶️ Evaluate Model on Test Set")

    if len(test_df)==0:
        st.info("Add or import test rows first.")
    else:
        # Train threshold-conditioned models PER DISTINCT (pred_type, threshold) in test set
        results = []
        # Precompute history features from TRAIN for each threshold once
        # Evaluate each row with model trained on TRAIN only
        for (ptype, thr) in sorted(test_df[["pred_type","threshold"]].dropna().drop_duplicates().itertuples(index=False)):
            thr = int(thr)
            Xk_e, yk_e, Xp_e, yp_e = make_training(df, thr, thr)
            mk_e, ck_e = fit_model(Xk_e, yk_e)
            mp_e, cp_e = fit_model(Xp_e, yp_e)
            f_hist = hist_feats(df, thr, thr)
            if f_hist is None:
                continue

            mask = (test_df["pred_type"]==ptype) & (test_df["threshold"].astype(int)==thr)
            subset = test_df.loc[mask].copy()
            probs, y_true, y_hat, you_hat = [], [], [], []
            for _, r in subset.iterrows():
                meta = pd.DataFrame([{
                    "event_type": r["event_type"],
                    "team_mode":  r["team_mode"],
                    "match_number_in_tourney": int(r["match_number_in_tourney"])
                }])
                Xrow_t = pd.concat([meta.reset_index(drop=True), f_hist.reset_index(drop=True)], axis=1)

                if ptype == "Eliminations":
                    p = predict_proba(mk_e, ck_e, Xrow_t, fallback=0.5)
                    y = int(int(r["kills"]) >= thr)
                else:
                    p = predict_proba(mp_e, cp_e, Xrow_t, fallback=0.5)
                    y = int(int(r["placement"]) <= thr)

                probs.append(p)
                y_true.append(y)
                y_hat.append(1 if p>=0.5 else 0)

                if isinstance(r.get("user_pick",""), str) and r["user_pick"] in ["YES","NO"]:
                    you_hat.append(1 if r["user_pick"]=="YES" else 0)
                else:
                    you_hat.append(np.nan)

            subset["pred_prob"] = probs
            subset["y_true"] = y_true
            subset["y_hat"]  = y_hat
            subset["correct"] = (subset["y_hat"] == subset["y_true"]).astype(int)
            subset["user_hat"] = you_hat
            subset["user_correct"] = np.where(pd.notna(subset["user_hat"]),
                                              (subset["user_hat"]==subset["y_true"]).astype(int), np.nan)

            # Metrics per group
            acc  = subset["correct"].mean() if len(subset)>0 else np.nan
            brier= np.mean((subset["pred_prob"] - subset["y_true"])**2) if len(subset)>0 else np.nan
            uacc = subset["user_correct"].mean() if subset["user_correct"].notna().any() else np.nan

            results.append((ptype, thr, acc, brier, uacc, subset))

        if not results:
            st.info("Insufficient training data for evaluation.")
        else:
            # Aggregate easy-to-read summary
            rows = []
            for ptype, thr, acc, brier, uacc, _ in results:
                rows.append({
                    "Prediction": ptype,
                    "Threshold": thr,
                    "Model Accuracy": f"{acc*100:.1f}%" if pd.notna(acc) else "N/A",
                    "Brier Score": f"{brier:.3f}" if pd.notna(brier) else "N/A",
                    "Your Pick Accuracy": f"{uacc*100:.1f}%" if pd.notna(uacc) else "N/A",
                })
            summary = pd.DataFrame(rows)
            st.subheader("Summary")
            st.dataframe(summary, use_container_width=True)

            # Overall accuracy across all test rows (simple mean of correctness)
            all_parts = [sub for *_, sub in results]
            joined = pd.concat(all_parts, ignore_index=True)
            overall_acc = joined["correct"].mean() if len(joined)>0 else np.nan
            overall_brier = np.mean((joined["pred_prob"] - joined["y_true"])**2) if len(joined)>0 else np.nan
            c1,c2 = st.columns(2)
            c1.metric("Overall Model Accuracy", f"{overall_acc*100:.1f}%" if pd.notna(overall_acc) else "N/A")
            c2.metric("Overall Brier Score", f"{overall_brier:.3f}" if pd.notna(overall_brier) else "N/A")

            # Show detailed rows and allow download
            st.subheader("Detailed Results")
            st.dataframe(joined[[
                "_tid","event_type","team_mode","match_number_in_tourney",
                "pred_type","threshold","kills","placement",
                "pred_prob","y_hat","y_true","correct","user_pick","user_correct"
            ]], use_container_width=True)
            st.download_button(
                "📥 Download TEST RESULTS CSV",
                data=joined.to_csv(index=False),
                file_name=f"fortnite_test_results_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
