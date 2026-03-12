import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Blinkit AI Decision Platform", layout="wide")

st.title("🚀 AI-Powered Blinkit Business Decision Platform")
st.markdown("---")

# --- DATA PREP (Layer 1, 2 & 4) ---
@st.cache_data
def load_full_data():
    # 1. Marketing Data
    m_data = {
        'Date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07']),
        'Revenue': [15000, 18000, 12000, 25000, 22000, 30000, 28000],
        'Ad_Spend': [5000, 6000, 7000, 5500, 8000, 6000, 9000]
    }
    df_m = pd.DataFrame(m_data)
    df_m['ROAS'] = df_m['Revenue'] / df_m['Ad_Spend']

    # 2. Expanded Feedback Data (For better AI search)
    feedback_data = [
        {"text": "The order was incorrect.", "cat": "App Experience", "keywords": "order incorrect wrong"},
        {"text": "Product was damaged during delivery.", "cat": "Delivery", "keywords": "damaged broken delivery"},
        {"text": "It was okay, nothing special.", "cat": "General", "keywords": "okay special average"},
        {"text": "Highly recommended!", "cat": "Quality", "keywords": "good recommended best"},
        {"text": "Taste was not as expected.", "cat": "Quality", "keywords": "taste food quality expected"},
        {"text": "Delivery was late and I was unhappy.", "cat": "Delivery", "keywords": "late delayed unhappy slow"},
        {"text": "Great prices and fast delivery!", "cat": "Delivery", "keywords": "fast delivery quick price"},
        {"text": "Customer service was not helpful.", "cat": "Customer Service", "keywords": "service support helpful rude"},
        {"text": "Items were missing from my order.", "cat": "Orders", "keywords": "missing items incomplete"},
        {"text": "The packaging was poor.", "cat": "App Experience", "keywords": "packaging box poor bad"}
    ]
    df_f = pd.DataFrame(feedback_data)
    return df_m, df_f

# --- ML MODEL (Layer 3) ---
@st.cache_resource
def train_model():
    X = np.array([[18, 4, 0], [10, 0, 1], [20, 5, 0], [12, 1, 2], [19, 4, 0], [15, 2, 1], [21, 6, 0], [9, 0, 2]])
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0]) 
    model = RandomForestClassifier(n_estimators=10).fit(X, y)
    return model

df_marketing, df_feedback = load_full_data()
ml_model = train_model()

# --- NAVIGATION ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Analytics Dashboard", "Risk Predictor", "AI Business Assistant"])

if app_mode == "Analytics Dashboard":
    st.subheader("📊 Marketing & Revenue Trends")
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Revenue", f"₹{df_marketing['Revenue'].sum():,}")
    col2.metric("📢 Total Ad Spend", f"₹{df_marketing['Ad_Spend'].sum():,}")
    col3.metric("📈 Avg ROAS", f"{df_marketing['ROAS'].mean():.2f}x")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_marketing['Date'], y=df_marketing['Revenue'], name='Revenue', line=dict(color='green', width=4)))
    fig.add_trace(go.Bar(x=df_marketing['Date'], y=df_marketing['Ad_Spend'], name='Ad Spend', marker_color='red'))
    st.plotly_chart(fig)

elif app_mode == "Risk Predictor":
    st.subheader("🔮 Delivery Delay Risk Calculator")
    h = st.slider("Hour", 0, 23, 18)
    d = st.selectbox("Day", options=[0,1,2,3,4,5,6], format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    r = st.selectbox("Region", options=[0, 1, 2], format_func=lambda x: ["Indiranagar", "Koramangala", "Whitefield"][x])
    if st.button("Check Risk"):
        risk = ml_model.predict_proba([[h, d, r]])[0][1] * 100
        st.write(f"Risk Level: {risk}%")

elif app_mode == "AI Business Assistant":
    st.subheader("🤖 AI Business Chatbot (Feedback Analyzer)")
    st.write("I use NLP logic to search through feedback and provide insights.")
    
    query = st.text_input("Ask me about delivery, packaging, quality, etc.")
    
    if query:
        # SMART SEARCH LOGIC
        # Kelviya split panni keywords check pannudhu
        query_words = query.lower().split()
        
        # Finding matches in 'text' OR 'keywords' column
        def find_matches(row):
            for word in query_words:
                if word in row['text'].lower() or word in row['keywords'].lower():
                    return True
            return False

        results = df_feedback[df_feedback.apply(find_matches, axis=1)]
        
        if not results.empty:
            st.write(f"**AI Assistant:** Analyzing {len(results)} relevant feedback rows...")
            
            # Show top 3 matches
            for i, row in results.head(3).iterrows():
                st.info(f"Feedback: '{row['text']}' (Category: {row['cat']})")
            
            # AI Generated Summary (Logic based)
            st.success(f"**AI Summary:** Customers who mentioned '{query}' are primarily facing issues with **{results['cat'].mode()[0]}**. It is recommended to improve the operations in this department.")
        else:
            st.write("**AI Assistant:** I couldn't find a direct match. However, the data shows that **Delivery** is the most complained category this week.")