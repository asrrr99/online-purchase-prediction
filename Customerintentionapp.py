import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Load model and encoders
model = joblib.load("online_shopper_model.pkl")
month_encoder = joblib.load("month_encoder.pkl")
visitor_encoder = joblib.load("visitor_encoder.pkl")

# Page configuration
st.set_page_config(
    page_title="Online Shopper Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #dbeafe 50%, #bfdbfe 100%);
}

.main-box {
    background-color: rgba(255,255,255,0.96);
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.1);
}

html, body, p, div, label {
    color: black !important;
}

h1 {
    color:#0f172a !important;
    text-align:center;
    font-weight:700;
}

h2, h3 {
    color:#1e293b !important;
}

section[data-testid="stSidebar"] {
    background-color:white !important;
    border-right:2px solid #dbeafe;
}

section[data-testid="stSidebar"] * {
    color:black !important;
}

div[data-baseweb="select"] > div {
    background-color:white !important;
    color:black !important;
    border:1px solid #cbd5e1 !important;
}

div[data-baseweb="select"] span {
    color:black !important;
}

ul[role="listbox"] {
    background-color:white !important;
}

li[role="option"] {
    background-color:white !important;
    color:black !important;
}

li[role="option"]:hover {
    background-color:#dbeafe !important;
}

div[role="radiogroup"] label {
    color:black !important;
}

input {
    background-color:white !important;
    color:black !important;
}

.stSlider label {
    color:black !important;
}

[data-testid="stDataFrame"] {
    background-color:white !important;
}

[data-testid="stMetricValue"] {
    color:black !important;
    font-weight:bold;
}

.stButton > button {
    background-color:#2563eb;
    color:white !important;
    border:none;
    border-radius:10px;
    font-size:16px;
    font-weight:bold;
    padding:10px 25px;
}

.stButton > button:hover {
    background-color:#1d4ed8;
    color:white !important;
}

table {
    color:black !important;
}

.shopping-card-success {
    text-align:center;
    background: linear-gradient(135deg, #ecfdf5, #dcfce7);
    padding: 25px;
    border-radius: 18px;
    border: 2px solid #22c55e;
    margin-top: 15px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 15px rgba(34,197,94,0.25);
}

.shopping-card-danger {
    text-align:center;
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    padding: 25px;
    border-radius: 18px;
    border: 2px solid #ef4444;
    margin-top: 15px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 15px rgba(239,68,68,0.25);
}

.shopping-icons {
    font-size: 60px;
    margin-bottom: 10px;
}

.success-title {
    color:#16a34a !important;
    font-size:28px;
    font-weight:700;
}

.danger-title {
    color:#dc2626 !important;
    font-size:28px;
    font-weight:700;
}

.shopping-text {
    font-size:18px;
    color:#1e293b !important;
}

.profile-card {
    background:white;
    padding:22px;
    border-radius:18px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.10);
    border-left:6px solid #2563eb;
}

.profile-card h3 {
    color:#0f172a !important;
    margin-bottom:15px;
}

.profile-card p {
    font-size:16px;
    margin:8px 0;
}

.progress-card {
    background:white;
    padding:22px;
    border-radius:18px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.10);
    border-left:6px solid #22c55e;
}
</style>
""", unsafe_allow_html=True)

# Main container
st.markdown('<div class="main-box">', unsafe_allow_html=True)

# Header
st.title("🛒 Online Shopper Purchase Prediction")

st.markdown("""
### Predict Customer Purchase Intention

This machine learning application predicts whether an online visitor is likely to make a purchase based on browsing behavior.
""")

# Sidebar inputs
st.sidebar.header("Customer Browsing Information")

ProductRelated = st.sidebar.slider(
    "Product Pages Visited",
    0, 800, 50,
    step=1
)

ProductRelated_Duration = st.sidebar.slider(
    "Time Spent on Product Pages",
    0.0, 6000.0, 1500.0,
    step=1.0,
    format="%.0f"
)

PageValues = st.sidebar.slider(
    "Page Value",
    0.0, 400.0, 20.0,
    step=1.0,
    format="%.0f"
)

BounceRates = st.sidebar.slider(
    "Bounce Rate",
    0.0, 0.20, 0.02,
    step=0.01,
    format="%.2f"
)

ExitRates = st.sidebar.slider(
    "Exit Rate",
    0.0, 0.20, 0.04,
    step=0.01,
    format="%.2f"
)

VisitorType = st.sidebar.selectbox(
    "Visitor Type",
    visitor_encoder.classes_
)

Month = st.sidebar.selectbox(
    "Month",
    month_encoder.classes_
)

Weekend = st.sidebar.radio(
    "Weekend Visit",
    ["No", "Yes"]
)

# Encoding
VisitorType_encoded = visitor_encoder.transform([VisitorType])[0]
Month_encoded = month_encoder.transform([Month])[0]
Weekend_encoded = 1 if Weekend == "Yes" else 0

# Input data
input_data = pd.DataFrame({
    "ProductRelated": [ProductRelated],
    "ProductRelated_Duration": [ProductRelated_Duration],
    "PageValues": [PageValues],
    "BounceRates": [BounceRates],
    "ExitRates": [ExitRates],
    "VisitorType": [VisitorType_encoded],
    "Weekend": [Weekend_encoded],
    "Month": [Month_encoded]
})

# ==========================
# Visitor Summary + Progress Cards
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Profile")

    st.markdown(f"""
    <div class="profile-card">
        <h3>🛍️ Visitor Summary</h3>
        <p><b>Visitor Type:</b> {VisitorType}</p>
        <p><b>Month:</b> {Month}</p>
        <p><b>Weekend Visit:</b> {Weekend}</p>
        <p><b>Product Pages:</b> {ProductRelated}</p>
        <p><b>Time Spent:</b> {int(ProductRelated_Duration)}</p>
        <p><b>Page Value:</b> {int(PageValues)}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("📊 Customer Behaviour Score")

    st.markdown('<div class="progress-card">', unsafe_allow_html=True)

    st.write("🛒 Product Pages Visited")
    st.progress(ProductRelated / 800)

    st.write("⏱️ Time Spent on Product Pages")
    st.progress(ProductRelated_Duration / 6000)

    st.write("💰 Page Value")
    st.progress(PageValues / 400)

    st.write("📉 Bounce Rate")
    st.progress(BounceRates / 0.20)

    st.write("🚪 Exit Rate")
    st.progress(ExitRates / 0.20)

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# Interactive visuals
st.subheader("📊 Interactive Customer Behaviour Visuals")

chart_data = pd.DataFrame({
    "Feature": [
        "Product Pages",
        "Time Spent",
        "Page Value",
        "Bounce Rate",
        "Exit Rate"
    ],
    "Value": [
        ProductRelated,
        ProductRelated_Duration,
        PageValues,
        BounceRates,
        ExitRates
    ]
})

fig_bar = px.bar(
    chart_data,
    x="Feature",
    y="Value",
    title="Customer Browsing Behaviour"
)

st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# Prediction button
if st.button("🔮 Predict Purchase Intention"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.subheader("Prediction Result")

    if probability >= 70:
        segment = "High Intent"
        segment_message = "High Purchase Potential: Recommend personalized offers and premium products."
    elif probability >= 40:
        segment = "Medium Intent"
        segment_message = "Medium Purchase Potential: Retargeting campaigns may improve conversions."
    else:
        segment = "Low Intent"
        segment_message = "Low Purchase Potential: Increase engagement through discounts and promotions."

    if prediction == 1:
        st.success("🛒 Customer is likely to make a purchase.")

        st.markdown("""
        <div class="shopping-card-success">
            <div class="shopping-icons">🛒 🛍️ 💳 ✅</div>
            <div class="success-title">Purchase Intent Detected!</div>
            <p class="shopping-text">
                Customer shows buying behaviour and is likely to complete a transaction.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("❌ Customer is unlikely to make a purchase.")

        st.markdown("""
        <div class="shopping-card-danger">
            <div class="shopping-icons">🛒 ❌ 🛍️</div>
            <div class="danger-title">Low Purchase Intent</div>
            <p class="shopping-text">
                Customer is browsing but currently shows low probability of purchasing.
            </p>
        </div>
        """, unsafe_allow_html=True)

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.metric("Purchase Probability", f"{probability:.2f}%")

    with kpi2:
        st.metric("Prediction", "Purchase" if prediction == 1 else "No Purchase")

    with kpi3:
        st.metric("Customer Segment", segment)

    st.subheader("📈 Probability Progress")
    st.progress(int(probability))
    st.write(f"Purchase Chance: **{probability:.2f}%**")

    st.subheader("📊 Purchase Probability Gauge")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={"text": "Purchase Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2563eb"},
            "steps": [
                {"range": [0, 40], "color": "#fee2e2"},
                {"range": [40, 70], "color": "#fef3c7"},
                {"range": [70, 100], "color": "#dcfce7"}
            ]
        }
    ))

    st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader("🎯 Customer Analysis")

    if probability >= 70:
        st.success(segment_message)
    elif probability >= 40:
        st.warning(segment_message)
    else:
        st.info(segment_message)

    result_df = pd.DataFrame({
        "Prediction": ["Purchase" if prediction == 1 else "No Purchase"],
        "Probability": [f"{probability:.2f}%"],
        "Customer Segment": [segment]
    })

    st.subheader("📌 Final Prediction")
    st.table(result_df)

# Footer
st.markdown("---")
st.markdown(
    '<p class="footer">Developed using Machine Learning, Random Forest & Streamlit</p>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)