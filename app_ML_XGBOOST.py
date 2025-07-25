import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os

# Page config
st.set_page_config(
    page_title="Sentiment Analysis - Tokopedia Reviews",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .confidence-score {
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource

def load_vectorizer_and_encoder():
    try:
        vectorizer = joblib.load("Load Model XGBOOST/tfidf_vectorizer.pkl")
        label_encoder = joblib.load("Load Model XGBOOST/label_encoder.pkl")
        return vectorizer, label_encoder
    except Exception as e:
        st.error(f"Error loading vectorizer/encoder: {str(e)}")
        return None, None

@st.cache_resource

def load_xgb_model(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading XGBoost model: {str(e)}")
        return None

def predict_sentiment(text, vectorizer, model, label_encoder):
    if not text.strip():
        return None, None
    try:
        X = vectorizer.transform([text])
        y_proba = model.predict_proba(X)[0]
        y_pred = np.argmax(y_proba)
        label = label_encoder.inverse_transform([y_pred])[0]
        confidence = y_proba[y_pred]
        return label, confidence, y_proba
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None, None

def create_confidence_chart(probabilities, labels):
    import plotly.express as px
    fig = px.bar(
        x=labels,
        y=probabilities,
        color=labels,
        color_discrete_map={
            'negative': '#dc3545',
            'neutral': '#ffc107',
            'positive': '#28a745'
        },
        title="Confidence Scores for Each Sentiment"
    )
    fig.update_layout(
        xaxis_title="Sentiment",
        yaxis_title="Confidence Score",
        showlegend=False,
        height=400
    )
    return fig

def display_sentiment_result(sentiment, confidence):
    if sentiment == 'positive':
        st.markdown(f"""
        <div class="sentiment-positive">
            <h3>😊 Positive Sentiment</h3>
            <div class="confidence-score">Confidence: {confidence:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    elif sentiment == 'negative':
        st.markdown(f"""
        <div class="sentiment-negative">
            <h3>😞 Negative Sentiment</h3>
            <div class="confidence-score">Confidence: {confidence:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="sentiment-neutral">
            <h3>😐 Neutral Sentiment</h3>
            <div class="confidence-score">Confidence: {confidence:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.markdown('<a href="https://sentimen-ulasan-tokopedia.streamlit.app/" target="_blank" style="text-decoration:none;"><button style="background-color:#1f77b4;color:white;padding:10px 20px;border:none;border-radius:5px;font-size:1.1rem;margin-bottom:20px;">Coba versi Deep Learning IndoBERT di sini</button></a>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🛍️ Tokopedia Review Sentiment Analysis</h1>', 
                unsafe_allow_html=True)
    st.markdown("""
    Aplikasi ini menggunakan pipeline **TFIDF + Model Utama** yang telah dilatih untuk menganalisis sentiment 
    dari review produk Tokopedia. Masukkan review Anda dan dapatkan analisis sentiment secara real-time!
    """)

    # Load vectorizer and label encoder
    with st.spinner("Loading vectorizer & label encoder..."):
        vectorizer, label_encoder = load_vectorizer_and_encoder()
    if vectorizer is None or label_encoder is None:
        st.error("Failed to load vectorizer or label encoder.")
        return

    # Sidebar: upload model
    st.sidebar.header("⚙️ Model Utama")
    uploaded_model = st.sidebar.file_uploader(
        "Upload file model utama (.pkl, optional)",
        type=["pkl"],
        help="Jika tidak diupload, akan menggunakan model default."
    )
    if uploaded_model is not None:
        model = pickle.load(uploaded_model)
        model_status = "Model hasil upload"
    else:
        model = load_xgb_model("Load Model XGBOOST/xgb_sentiment_model.pkl")
        model_status = "Model default bawaan"
    if model is None:
        st.error("Failed to load model utama.")
        return
    st.sidebar.success(f"{model_status} digunakan.")

    # Sidebar info
    st.sidebar.header("📋 About the Model")
    st.sidebar.info("""
    **Pipeline:** TFIDF + Model Utama + LabelEncoder
    **Task:** Sentiment Classification
    **Classes:** 
    - 😞 Negative (Rating 1-2)
    - 😐 Neutral (Rating 3)
    - 😊 Positive (Rating 4-5)
    """)
    st.sidebar.header("🧪 Sample Reviews")
    sample_reviews = {
        "Positive": "Produk sangat bagus, kualitas excellent, pengiriman cepat, pokoknya recommended banget!",
        "Negative": "Produk rusak, tidak sesuai deskripsi, pengiriman lama, sangat mengecewakan",
        "Neutral": "Produk biasa saja, sesuai dengan harganya, tidak ada yang istimewa"
    }
    selected_sample = st.sidebar.selectbox("Choose a sample:", [""] + list(sample_reviews.keys()))

    # Main input area
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("💬 Enter Your Review")
        default_text = sample_reviews.get(selected_sample, "")
        user_input = st.text_area(
            "Write your product review here:",
            value=default_text,
            height=150,
            placeholder="Contoh: Produk bagus banget, kualitas oke, pengiriman cepat..."
        )
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    with col2:
        st.subheader("ℹ️ Tips")
        st.info("""
        **Tips untuk review yang baik:**
        - Gunakan bahasa Indonesia
        - Jelaskan pengalaman Anda
        - Sebutkan kualitas produk
        - Berikan detail tentang pengiriman
        """)

    # Analysis results
    if analyze_button and user_input:
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence, probabilities = predict_sentiment(user_input, vectorizer, model, label_encoder)
        if sentiment:
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            col1, col2 = st.columns([1, 1])
            with col1:
                display_sentiment_result(sentiment, confidence)
            with col2:
                labels = label_encoder.classes_.tolist()
                fig = create_confidence_chart(probabilities, labels)
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("📈 Detailed Breakdown")
            breakdown_df = pd.DataFrame({
                'Sentiment': label_encoder.classes_,
                'Confidence': [f"{prob:.2%}" for prob in probabilities],
                'Score': probabilities
            })
            st.dataframe(breakdown_df, use_container_width=True)
            st.subheader("🎯 Interpretation")
            if confidence > 0.8:
                interpretation = "Very confident prediction"
            elif confidence > 0.6:
                interpretation = "Moderately confident prediction"
            else:
                interpretation = "Low confidence prediction - review might be ambiguous"
            st.info(f"**{interpretation}** - The model is {confidence:.2%} confident that this review is {sentiment}.")
    elif analyze_button and not user_input:
        st.warning("Please enter a review to analyze!")

    # Statistics section (optional)
    if st.checkbox("Show Model Performance Stats"):
        st.subheader("📊 Model Performance")
        performance_data = {
            'Metric': ['Accuracy', 'Precision (Positive)', 'Recall (Positive)', 'F1-Score (Positive)'],
            'Score': [0.85, 0.87, 0.83, 0.85]  # Replace with your actual metrics
        }
        perf_df = pd.DataFrame(performance_data)
        import plotly.express as px
        fig_perf = px.bar(perf_df, x='Metric', y='Score', 
                         title="Model Performance Metrics",
                         color='Score',
                         color_continuous_scale='Viridis')
        st.plotly_chart(fig_perf, use_container_width=True)

if __name__ == "__main__":
    main()