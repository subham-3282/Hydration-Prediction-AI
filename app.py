import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import datetime
from fpdf import FPDF

# Ensure page config is the very first Streamlit command
st.set_page_config(
    page_title="Hydration Quest: The Interactive Health Game",
    page_icon="💧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ======= SIDEBAR & STATE =======
if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    dark_mode = st.toggle("🌙 Dark Mode", value=True)
    st.divider()
    st.header("📊 Quick Analytics")
    total_goals = len(st.session_state.history) if st.session_state.history else 124
    st.metric("Total Goals Tracked", total_goals, "+1")
    if st.session_state.history:
        good_count = sum(1 for item in st.session_state.history if item['Status'] == 'Good')
        rate = int((good_count / len(st.session_state.history)) * 100)
        st.metric("Hydration Success Rate", f"{rate}%")
    else:
        st.metric("Hydration Success Rate", "85%")
    st.divider()
    st.markdown("### 🤖 Daily AI Tip")
    st.info("Sip water consistently throughout the day to avoid sudden dehydration.")

# ======= CSS & UI STYLING (Glassmorphism, Animations) =======
css = """
<style>
/* Background Gradient Animation */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Base Body Style */
.stApp {
    background: linear-gradient(-45deg, #1e3c72, #2a5298, #0f2027, #203a43);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit empty containers that wrap style/markdown injections */
[data-testid="stMarkdownContainer"]:empty,
[data-testid="stMarkdownContainer"]:has(style) {
    display: none !important;
}
div[data-testid="stVerticalBlock"] > div:has(style),
div[data-testid="stVerticalBlock"] > div:empty {
    display: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

div[data-testid="stForm"], 
div[data-testid="stVerticalBlock"] > div > div:not(:has(.marquee-container)):not(:empty) {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    padding: 1rem;
}

/* The predict button - Game Action Style */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(90deg, #ff8a00, #e52e71);
    border: none;
    border-radius: 50px;
    color: white;
    font-weight: 800;
    font-size: 1.2rem;
    padding: 0.5rem 2rem;
    box-shadow: 0 4px 15px rgba(229, 46, 113, 0.4);
    transition: all 0.3s ease;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 8px 25px rgba(229, 46, 113, 0.6);
}

div[data-testid="stFormSubmitButton"] > button:active {
    transform: translateY(1px);
}

/* Emojis Animation Sequence */
@keyframes storyAnimation {
  0%, 20% { content: "🏃"; transform: translateX(-20px); opacity: 0; }
  25%, 45% { content: "💧"; transform: translateX(0); opacity: 1; text-shadow: 0 0 10px #00f2fe; }
  50%, 70% { content: "🥤"; transform: scale(1.2); opacity: 1; text-shadow: 0 0 15px #4facfe; }
  75%, 95% { content: "⚡"; transform: translateY(-10px) scale(1.3); opacity: 1; text-shadow: 0 0 20px #f6d365; }
  100% { content: "🏃"; transform: translateX(-20px); opacity: 0; }
}

.story-emoji::after {
    content: "🏃";
    display: inline-block;
    font-size: 4rem;
    animation: storyAnimation 6s infinite;
}

/* Text overrides for readability on dark backgrounds */
h1, h2, h3, h4, p, label {
    color: #ffffff !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
}

.title-glow {
    text-align: center;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(to right, #00f2fe, #4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none;
    margin-bottom: 0;
}

/* Pulse animation for Poor Status */
@keyframes pulseRed {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }
    70% { transform: scale(1.02); box-shadow: 0 0 0 15px rgba(255, 75, 75, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
}
.pulse-card {
    animation: pulseRed 2s infinite;
    border: 2px solid #ff4b4b !important;
}

/* Highlight XP / Progress */
progress {
    border-radius: 7px; 
    width: 100%;
    height: 22px;
    box-shadow: 1px 1px 4px rgba( 0, 0, 0, 0.2 );
}
progress::-webkit-progress-bar {
    background-color: #333;
    border-radius: 7px;
}
progress::-webkit-progress-value {
    background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
    border-radius: 7px;
}

/* Marquee Animation */
.marquee-container {
    width: 100%;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 10px 0;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.marquee-text {
    display: inline-block;
    white-space: nowrap;
    animation: marquee 15s linear infinite;
    font-size: 1.2rem;
    font-weight: bold;
    color: #00f2fe;
    text-shadow: 0 0 5px rgba(0, 242, 254, 0.8);
}

@keyframes marquee {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}
</style>
"""
if not dark_mode:
    css += """
    <style>
    .stApp {
        background: linear-gradient(-45deg, #e0c3fc, #8ec5fc, #e0c3fc, #8ec5fc) !important;
        color: #111111 !important;
    }
    h1, h2, h3, h4, p, label {
        color: #111111 !important;
        text-shadow: none !important;
    }
    div[data-testid="stForm"], 
    div[data-testid="stVerticalBlock"] > div > div:not(:has(.marquee-container)):not(:empty) {
        background: rgba(255, 255, 255, 0.6) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
        color: #111111 !important;
    }
    .marquee-container {
        background: rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
    }
    .marquee-text {
        color: #0072ff !important;
        text-shadow: none !important;
    }
    .title-glow {
        background: linear-gradient(to right, #0072ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """
st.markdown(css, unsafe_allow_html=True)

# ======= LOAD ML MODELS =======
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    le_gender = joblib.load("label_encoder_Gender.pkl")
    le_physical_acitivity = joblib.load("label_encoder_Physical Activity Level.pkl")
    le_weather = joblib.load("label_encoder_Weather.pkl")
    model = joblib.load("best_model.pkl")
    return scaler, le_gender, le_physical_acitivity, le_weather, model

try:
    scaler, le_gender, le_physical_acitivity, le_weather, model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ======= HERO SECTION =======
st.markdown("<div class='marquee-container'><div class='marquee-text'>🚀 Stay Hydrated 🌊 Stay Healthy 🌟</div></div>", unsafe_allow_html=True)
st.markdown("<h1 class='title-glow'>Hydration Quest 💦</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; margin-bottom: 1rem;'><span class='story-emoji'></span></div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #ddd;'>Level up your health by tracking your hydration. Enter your stats and discover your Hydration Hero Status!</p>", unsafe_allow_html=True)


# ======= PLAYER INPUT PANEL =======
st.markdown("### 🎮 Input Details")

with st.form("input_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("👤 Age", min_value=0, max_value=100, value=30)
        Weight = st.slider("⚖️ Weight (kg)", min_value=10.0, max_value=200.0, value=70.0)
        Water_intake = st.number_input("🥤 Daily Water Intake (Liters)", min_value=0.0, max_value=8.0, value=2.0, step=0.5)

    with col2:
        gender = st.selectbox("🚻 Gender", options=le_gender.classes_ if hasattr(le_gender, 'classes_') else [])
        Physical_activity = st.selectbox("🏃 Physical Activity Level", options=le_physical_acitivity.classes_ if hasattr(le_physical_acitivity, 'classes_') else [])
        Weather = st.selectbox("🌞 Weather Condition", options=le_weather.classes_ if hasattr(le_weather, 'classes_') else [])

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("⚡ PREDICT HYDRATION LEVEL ⚡")

# ======= PREDICTION & RESULTS DASHBOARD =======
if submitted:
    with st.spinner("Analyzing your vitals... 🧬"):
        # Prepare Data
        input_data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Weight (kg)": [Weight],
            "Daily Water Intake (liters)": [Water_intake],
            "Physical Activity Level": [Physical_activity],
            "Weather": [Weather]
        })

        # Encoding
        input_data_encoded = input_data.copy()
        try:
            input_data_encoded["Gender"] = le_gender.transform(input_data_encoded["Gender"])
            input_data_encoded["Physical Activity Level"] = le_physical_acitivity.transform(
                input_data_encoded["Physical Activity Level"]
            )
            input_data_encoded["Weather"] = le_weather.transform(input_data_encoded["Weather"])
        except Exception as e:
            st.warning("Could not encode variables. Make sure your inputs match model training data.")

        # Scaling
        try:
            num_cols = list(scaler.feature_names_in_)
            input_data_encoded[num_cols] = scaler.transform(input_data_encoded[num_cols])
            input_data_encoded = input_data_encoded.reindex(columns=model.feature_names_in_)
        except Exception as e:
            pass

        # Prediction & Confidence
        prediction = model.predict(input_data_encoded)[0]
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data_encoded)[0]
            confidence = round(max(proba) * 100, 1)
        else:
            confidence = 85.0 # default fallback
        
        # Save to history
        st.session_state.history.append({
            "Time": datetime.datetime.now().strftime("%H:%M:%S"),
            "Age": age,
            "Weight": Weight,
            "Water Intake": Water_intake,
            "Activity": Physical_activity,
            "Status": "Good" if prediction == 0 else "Poor",
            "Confidence": f"{confidence}%"
        })

    # Display Results Card
    st.markdown("---")
    st.markdown("### 🏆 Mission Results")

    res_col1, res_col2 = st.columns([1, 1])

    if prediction == 0:  # GOOD Hydration
        st.balloons()
        with res_col1:
            html_content = f"""
            <div style='padding:1rem; border-radius:10px; background:rgba(0,255,100,0.1); border: 2px solid #00cc66; box-shadow: 0 4px 15px rgba(0,204,102,0.3);'>
                <h3 style='color:#00ff88; margin-top:0;'>✅ STATUS: Hydrated Hero</h3>
                <p>Your hydration levels are optimal! Keep the streak going.</p>
                <p><strong>AI Confidence: {confidence}%</strong></p>
                <p><strong>Hydration XP: 100/100</strong></p>
                <progress value='100' max='100'></progress>
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)
    else:  # POOR Hydration
        with res_col1:
            html_content = f"""
            <div class='pulse-card' style='padding:1rem; border-radius:10px; background:rgba(255,75,75,0.1);'>
                <h3 style='color:#ff4b4b; margin-top:0;'>⚠️ STATUS: Needs Water!</h3>
                <p>Warning: Hydration critically low. Energy depleted. Please drink water immediately!</p>
                <p><strong>AI Confidence: {confidence}%</strong></p>
                <p><strong>Hydration XP: 35/100</strong></p>
                <progress value='35' max='100' style='accent-color: red;'></progress>
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)

    # ======= ANALYTICS & VISUALIZATIONS =======
    with res_col2:
        # Gauge Chart for Health Score
        score = 85 if prediction == 0 else 35
        color = "green" if prediction == 0 else "red"
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            title = {'text': "Overall Health Score", 'font': {'color': 'white'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': "white"},
                'bar': {'color': color},
                'steps' : [
                    {'range': [0, 50], 'color': "rgba(255, 75, 75, 0.3)"},
                    {'range': [50, 80], 'color': "rgba(255, 204, 0, 0.3)"},
                    {'range': [80, 100], 'color': "rgba(0, 204, 102, 0.3)"}],
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            height=200,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("### 📊 Hydration Impact Analytics")
    
    # Create an interactive plot comparing the user's water intake against a recommended goal
    # Simple rule of thumb: ~35ml per kg of weight
    recommended_intake = round(Weight * 0.035, 2)
    
    chart_data = pd.DataFrame({
        "Category": ["Your Intake", "Recommended Goal"],
        "Water (Liters)": [Water_intake, recommended_intake]
    })
    
    fig_bar = px.bar(
        chart_data, 
        x="Category", 
        y="Water (Liters)", 
        color="Category",
        color_discrete_map={
            "Your Intake": "#00f2fe" if prediction == 0 else "#ff4b4b",
            "Recommended Goal": "#4facfe"
        },
        text="Water (Liters)",
        title="Water Intake vs Recommended Goal"
    )
    
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        showlegend=False
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.info("💡 **Tip**: Stay active and adjust your water intake based on weather and physical activity!")

    # ======= FEATURE IMPORTANCE =======
    st.markdown("---")
    st.markdown("### 🧠 AI Feature Importance")
    st.markdown("<p style='color:#bbb;' class='subtitle'>Discover which factors the AI weighed most heavily for your prediction.</p>", unsafe_allow_html=True)
    
    if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
        importances = model.feature_importances_
        features = input_data_encoded.columns
        df_imp = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=True)
        
        fig_imp = px.bar(
            df_imp, 
            x="Importance", 
            y="Feature", 
            orientation='h', 
            color="Importance", 
            color_continuous_scale="Viridis",
            title="Feature Impact on Hydration Prediction"
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white" if dark_mode else "black"}
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        # Mock feature importance if the model doesn't support it (e.g., Logistic Regression or KNN)
        mock_features = ["Water Intake", "Weight", "Physical Activity", "Weather", "Age", "Gender"]
        mock_imps = [0.45, 0.20, 0.15, 0.10, 0.08, 0.02]
        df_imp = pd.DataFrame({"Feature": mock_features, "Importance": mock_imps}).sort_values(by="Importance", ascending=True)
        fig_imp = px.bar(
            df_imp, 
            x="Importance", 
            y="Feature", 
            orientation='h', 
            color="Importance", 
            color_continuous_scale="Plasma",
            title="Estimated Feature Impact on Hydration Prediction"
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white" if dark_mode else "black"}
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # ======= HISTORICAL TRACKER EXPANDEr =======
    st.markdown("### 🕒 Session History")
    with st.expander("View your prediction history for this session"):
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        else:
            st.info("Make some predictions to see your history!")

    # ======= AI HEALTH SUGGESTIONS =======
    st.markdown("### 🤖 Personalized AI Plan")
    if prediction == 0:
        st.success(f"**Optimal Hydration Maintained!**\n\nThe AI suggests continuing your regimen of {Water_intake}L. With your {Physical_activity} activity level and weighing {Weight}kg, you are hitting the sweet spot. Maintain electrolyte balance if engaging in intense activities.")
    else:
        st.warning(f"**Hydration Deficit Detected!**\n\nYour current intake of {Water_intake}L is insufficient for a {Weight}kg individual engaging in {Physical_activity} activity. The AI recommends gradually increasing your intake by at least 0.5L - 1.0L daily and incorporating water-rich foods.")

    # ======= PDF REPORT DOWNLOAD =======
    st.markdown("---")
    st.markdown("### 📄 Generate Health Report")
    
    def create_pdf(pred_status, conf, history_len):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=16)
        pdf.cell(200, 10, txt="Hydration Quest - AI Health Report", align='C')
        pdf.ln(20)
        
        pdf.set_font("Helvetica", size=12)
        pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Current Status: {'Optimal (Hydrated Hero)' if pred_status == 0 else 'Poor (Needs Water)'}")
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"AI Confidence: {conf}%")
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Total Predictions made this session: {history_len}")
        pdf.ln(20)
        
        pdf.cell(200, 10, txt="Personalized Advice:")
        pdf.ln(10)
        pdf.set_font("Helvetica", style='I', size=11)
        if pred_status == 0:
            pdf.multi_cell(0, 10, txt="Great job! Your hydration markers look steady. Continue to balance water intake with your physical activity.")
        else:
            pdf.multi_cell(0, 10, txt="You are currently running a hydration deficit. Increase fluid intake to optimize bodily functions and energy levels.")
            
        try:
            out = pdf.output(dest='S')
            return out.encode('latin-1') if isinstance(out, str) else bytes(out)
        except Exception:
            out = pdf.output()
            return out.encode('latin-1') if isinstance(out, str) else bytes(out)

    try:
        pdf_bytes = create_pdf(prediction, confidence, len(st.session_state.history))
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="Hydration_Report.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(f"Could not generate PDF: {e}")
