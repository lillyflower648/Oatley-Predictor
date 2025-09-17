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

def get_confidence_level(prob):
    """Convert probability to confidence level"""
    confidence = abs(prob - 0.5) * 2  # Scale to 0-1
    if confidence >= 0.6:
        return "Very High", confidence
    elif confidence >= 0.4:
        return "High", confidence
    elif confidence >= 0.2:
        return "Moderate", confidence
    else:
        return "Low", confidence

def make_prediction(p_model):
    """Make YES/NO prediction with confidence"""
    conf_level, conf_score = get_confidence_level(p_model)
    
    if p_model > 0.55:  # Slight buffer from 0.5
        return "YES", conf_score, conf_level
    elif p_model < 0.45:
        return "NO", conf_score, conf_level
    else:
        return "UNCERTAIN", conf_score, "Too Close"

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

# ===================== UI =====================

st.title("🎮 Twitch Predictor - Fortnite Edition")

# Info box about the model
with st.expander("ℹ️ About This Predictor - Click to Learn More"):
    st.markdown("""
    ### 🤖 Model Type
    This app uses **Logistic Regression** with **Isotonic Calibration**:
    - **Logistic Regression**: A statistical model that predicts the probability of binary outcomes (YES/NO)
    - **Isotonic Calibration**: Adjusts predictions to be more accurate based on historical performance
    
    ### 📊 Data Requirements
    - **Minimum**: 25 matches to start making predictions
    - **Good**: 50-100 matches for reliable predictions
    - **Best**: 200+ matches for highly accurate predictions
    
    ### 🎯 Why This Model Works Well
    - **Perfect for binary predictions** (will they hit X kills? will they place top Y?)
    - **Learns patterns** from event types, game modes, and recent performance
    - **Adapts over time** as you add more data
    - **Considers momentum** by looking at recent 20-match windows
    
    ### 📈 Features It Analyzes
    - Recent kill average & variance
    - Recent placement average & variance  
    - Historical hit rates for similar predictions
    - Event type (scrims vs tournaments)
    - Match progression in tournament
    """)

# -------- Section 1: Add Match --------
st.header("📊 Add Match Data")
with st.form("add_match"):
    c1, c2 = st.columns(2)
    event_type = c1.selectbox("Event type", EVENT_TYPES, index=0)
    team_mode = c2.selectbox("Team mode", TEAM_MODES, index=0)

    c3, c4 = st.columns(2)
    match_num = c3.number_input("Match # in tournament", min_value=1, step=1, help="Which match is this in the current tournament?")
    total_elims_t = c4.number_input("Total tournament elims so far", min_value=0, step=1, help="Cumulative kills across all matches in this tournament")

    st.markdown("### Match Results")
    c5, c6 = st.columns(2)
    kills = c5.number_input("Eliminations this match", min_value=0, step=1, help="How many elims did they get?")
    placement = c6.number_input("Final placement", min_value=1, step=1, help="1 = Victory Royale, 2 = 2nd place, etc.")

    st.markdown("### Prediction Thresholds (for training)")
    c7, c8 = st.columns(2)
    K = c7.number_input("Elim threshold", min_value=1, value=5, step=1, help="What elim count were you predicting? (e.g., 5 for 5+ elims)")
    P = c8.number_input("Placement threshold", min_value=1, value=10, step=1, help="What placement were you predicting? (e.g., 10 for top 10)")

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
        st.success("✅ Match saved successfully!")
        st.rerun()

# Data status indicator
if len(df) < 25:
    st.warning(f"⚠️ Need {25-len(df)} more matches for predictions to start (currently: {len(df)}/25)")
elif len(df) < 50:
    st.info(f"📊 Model is learning! Add {50-len(df)} more matches for better accuracy (currently: {len(df)})")
else:
    st.success(f"✅ Model trained on {len(df)} matches - Good data coverage!")

st.divider()

# -------- Train models (on current data) --------
Xk, yk, Xp, yp = make_training(df)
mk, ck = fit_model(Xk, yk)  # kills>=K
mp, cp = fit_model(Xp, yp)  # placement<=P

# -------- Section 2: Current Prediction --------
st.header("🎯 Make a Prediction")

pred_type = st.radio(
    "What are you predicting?",
    ["Eliminations", "Placement"],
    horizontal=True,
    help="Choose what type of prediction the streamer is making"
)

if pred_type == "Eliminations":
    threshold = st.number_input(
        "🎯 Minimum eliminations to predict YES", 
        min_value=1, 
        value=5, 
        step=1,
        help="Example: Enter 5 to predict if they'll get 5 or MORE eliminations"
    )
    st.info(f"💡 Predicting: Will they get **{threshold} or more** eliminations?")
else:
    threshold = st.number_input(
        "🏆 Maximum placement to predict YES", 
        min_value=1, 
        value=10, 
        step=1,
        help="Example: Enter 10 to predict if they'll place in the top 10"
    )
    st.info(f"💡 Predicting: Will they place **top {threshold}** or better?")

# Get prediction
Xrow = current_Xrow(
    threshold if pred_type == "Eliminations" else df["K"].iloc[-1] if len(df) else threshold,
    threshold if pred_type == "Placement" else df["P"].iloc[-1] if len(df) else threshold
)

if Xrow is None:
    st.info("📝 Add at least 25 matches to start making predictions.")
else:
    # Fallbacks from last 20 matches
    last = df.tail(min(20, len(df)))
    fb_k = (last["kills"] >= threshold).mean() if len(last) > 0 else 0.5
    fb_p = (last["placement"] <= threshold).mean() if len(last) > 0 else 0.5

    if pred_type == "Eliminations":
        p_model = predict_proba(mk, ck, Xrow, fb_k)
        question = f"Will get {int(threshold)}+ eliminations?"
    else:
        p_model = predict_proba(mp, cp, Xrow, fb_p)
        question = f"Will place top {int(threshold)}?"

    # Get prediction
    prediction, conf_score, conf_level = make_prediction(p_model)
    
    # Display prediction box
    st.markdown("---")
    
    # Main prediction display
    if prediction == "YES":
        st.success(f"# ✅ Predict: **YES**")
    elif prediction == "NO":
        st.error(f"# ❌ Predict: **NO**")
    else:
        st.warning(f"# ⚠️ **Too Close to Call**")
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Probability", f"{p_model:.1%}")
        st.progress(p_model)
    
    with col2:
        confidence_pct = conf_score * 100
        st.metric("Confidence", f"{confidence_pct:.0f}%")
        if conf_level == "Very High":
            st.caption("🔥 Very High Confidence")
        elif conf_level == "High":
            st.caption("👍 High Confidence")
        elif conf_level == "Moderate":
            st.caption("🤔 Moderate Confidence")
        else:
            st.caption("📊 Low Confidence")
    
    with col3:
        # Historical performance
        if pred_type == "Eliminations":
            hist_rate = (last["kills"] >= threshold).mean() if len(last) > 0 else 0.5
            st.metric("Historical Rate", f"{hist_rate:.1%}")
            st.caption(f"Based on last {len(last)} matches")
        else:
            hist_rate = (last["placement"] <= threshold).mean() if len(last) > 0 else 0.5
            st.metric("Historical Rate", f"{hist_rate:.1%}")
            st.caption(f"Based on last {len(last)} matches")
    
    # Additional context
    with st.expander("📊 Detailed Analysis"):
        if pred_type == "Eliminations":
            avg_kills = last["kills"].mean() if len(last) > 0 else 0
            std_kills = last["kills"].std() if len(last) > 1 else 0
            st.write(f"**Recent Performance (last 20 matches):**")
            st.write(f"- Average eliminations: {avg_kills:.1f}")
            st.write(f"- Standard deviation: {std_kills:.1f}")
            st.write(f"- Times hit {threshold}+ elims: {(last['kills'] >= threshold).sum()}/{len(last)}")
            st.write(f"- Max eliminations: {last['kills'].max() if len(last) > 0 else 0}")
        else:
            avg_place = last["placement"].mean() if len(last) > 0 else 0
            st.write(f"**Recent Performance (last 20 matches):**")
            st.write(f"- Average placement: {avg_place:.1f}")
            st.write(f"- Times placed top {threshold}: {(last['placement'] <= threshold).sum()}/{len(last)}")
            st.write(f"- Best placement: {last['placement'].min() if len(last) > 0 else 'N/A'}")
            st.write(f"- Victory Royales: {(last['placement'] == 1).sum()}")

st.divider()

# -------- Section 3: Data Management --------
st.header("📂 Data & Analytics")

tab_stats, tab_data, tab_graphs = st.tabs(["📊 Statistics", "📋 Data Table", "📈 Performance Graphs"])

with tab_stats:
    if len(df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Overall Stats")
            st.metric("Total Matches", len(df))
            st.metric("Average Eliminations", f"{df['kills'].mean():.1f}")
            st.metric("Average Placement", f"{df['placement'].mean():.1f}")
        
        with col2:
            st.subheader("Best Performance")
            st.metric("Max Eliminations", f"{df['kills'].max()}")
            st.metric("Victory Royales", f"{(df['placement'] == 1).sum()}")
            st.metric("Top 10 Rate", f"{(df['placement'] <= 10).mean():.1%}")
        
        # Performance by event type
        st.subheader("Performance by Event Type")
        event_stats = df.groupby('event_type').agg({
            'kills': 'mean',
            'placement': 'mean'
        }).round(1)
        st.dataframe(event_stats)
    else:
        st.info("Statistics will appear after you add match data.")

with tab_data:
    col1, col2 = st.columns([3, 1])
    with col1:
        if len(df) > 0:
            st.dataframe(df.tail(50), use_container_width=True)
        else:
            st.info("No matches recorded yet.")
    
    with col2:
        if len(df) > 0:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv_data,
                file_name=f"fortnite_matches_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        
        up = st.file_uploader("📤 Import CSV", type=["csv"])
        if up is not None:
            try:
                newdf = pd.read_csv(up)
                df = pd.concat([df, newdf], ignore_index=True).drop_duplicates()
                df.to_csv(CSV, index=False)
                st.success("✅ Data imported!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Import failed: {e}")

with tab_graphs:
    if len(df) >= 2:
        df_plot = df.copy()
        df_plot["match_num"] = range(1, len(df_plot) + 1)
        
        # Eliminations over time
        st.subheader("🎯 Eliminations Trend")
        st.line_chart(df_plot.set_index("match_num")["kills"], height=300)
        
        # Placement over time
        st.subheader("🏆 Placement Trend")
        placement_chart = df_plot.set_index("match_num")["placement"]
        st.line_chart(placement_chart, height=300)
        
        # Rolling performance
        if len(df_plot) >= 5:
            st.subheader("📊 Rolling Performance (20-match window)")
            
            # Let user select thresholds for visualization
            col1, col2 = st.columns(2)
            viz_k = col1.slider("Elim threshold for graph", 1, 15, 5)
            viz_p = col2.slider("Placement threshold for graph", 1, 50, 10)
            
            win_k = (df_plot["kills"] >= viz_k).rolling(20, min_periods=1).mean()
            win_p = (df_plot["placement"] <= viz_p).rolling(20, min_periods=1).mean()
            
            performance_df = pd.DataFrame({
                f"{viz_k}+ Elim Rate": win_k.values,
                f"Top {viz_p} Rate": win_p.values
            }, index=df_plot["match_num"])
            
            st.line_chart(performance_df, height=300)
    else:
        st.info("📈 Graphs will appear after you add at least 2 matches.")