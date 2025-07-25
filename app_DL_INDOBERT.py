import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import pickle

st.set_page_config(
    page_title="Sentiment Analysis IndoBERT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_resource

def load_dl_model_and_tokenizer():
    try:
        model_dir = "Load Model INDOBERT"
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading IndoBERT model/tokenizer: {str(e)}")
        return None, None

@st.cache_resource

def load_label_encoder():
    try:
        with open(os.path.join("Load Model INDOBERT", "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        return label_encoder
    except Exception as e:
        st.error(f"Error loading label encoder: {str(e)}")
        return None

def predict_sentiment_dl(text, model, tokenizer, label_encoder):
    if not text.strip():
        return None, None, None
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0].cpu().numpy()
            probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        y_pred = np.argmax(probabilities)
        label = label_encoder.inverse_transform([y_pred])[0]
        confidence = probabilities[y_pred]
        return label, confidence, probabilities
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
    st.markdown('<a href="https://sentimen-ulasan-tokopedia.streamlit.app/" target="_blank" style="text-decoration:none;"><button style="background-color:#1f77b4;color:white;padding:10px 20px;border:none;border-radius:5px;font-size:1.1rem;margin-bottom:20px;">Coba versi Machine Learning XGBoost di sini</button></a>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🤖 Tokopedia Review Sentiment Analysis (IndoBERT)</h1>', 
                unsafe_allow_html=True)
    st.markdown("""
    Halaman ini menggunakan pipeline **IndoBERT Deep Learning** yang telah dilatih untuk menganalisis sentiment 
    dari review produk Tokopedia. Masukkan review Anda dan dapatkan analisis sentiment secara real-time!
    """)

    # Sidebar: upload model
    st.sidebar.header("⚙️ Model IndoBERT")
    uploaded_model = st.sidebar.file_uploader(
        "Upload file IndoBERT (.safetensors, optional)",
        type=["safetensors"],
        help="Jika tidak diupload, akan menggunakan model default."
    )
    model_dir = "Load Model INDOBERT"
    with st.spinner("Loading IndoBERT model & tokenizer..."):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            if uploaded_model is not None:
                # Save uploaded file to a temp path
                temp_path = os.path.join(model_dir, "uploaded_model.safetensors")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_model.read())
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir, state_dict=torch.load(temp_path), from_safetensors=True
                )
                model_status = "Model hasil upload"
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir, from_safetensors=True
                )
                model_status = "Model default bawaan"
        except Exception as e:
            st.error(f"Error loading IndoBERT model/tokenizer: {str(e)}")
            model = None
            tokenizer = None
            model_status = "Gagal load model"
    if model is not None:
        st.sidebar.success(f"{model_status} digunakan.")

    with st.spinner("Loading label encoder..."):
        label_encoder = load_label_encoder()
    if model is None or tokenizer is None or label_encoder is None:
        st.error("Failed to load model/tokenizer/label encoder.")
        return

    # Sidebar info
    st.sidebar.header("📋 About the Model")
    st.sidebar.info("""
    **Pipeline:** IndoBERT + LabelEncoder
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
        analyze_button = st.button("🔍 Analyze Sentiment (IndoBERT)", type="primary", use_container_width=True)
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
        with st.spinner("Analyzing sentiment with IndoBERT..."):
            sentiment, confidence, probabilities = predict_sentiment_dl(user_input, model, tokenizer, label_encoder)
        if sentiment:
            st.markdown("---")
            st.subheader("📊 Analysis Results (IndoBERT)")
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
    if st.checkbox("Show Model Performance Stats (IndoBERT)"):
        st.subheader("📊 Model Performance (IndoBERT)")
        performance_data = {
            'Metric': ['Accuracy', 'F1-score'],
            'Score': [0.87, 0.88]  # Replace with your actual metrics
        }
        perf_df = pd.DataFrame(performance_data)
        import plotly.express as px
        fig_perf = px.bar(perf_df, x='Metric', y='Score', 
                         title="Model Performance Metrics (IndoBERT)",
                         color='Score',
                         color_continuous_scale='Viridis')
        st.plotly_chart(fig_perf, use_container_width=True)

if __name__ == "__main__":
    main() 