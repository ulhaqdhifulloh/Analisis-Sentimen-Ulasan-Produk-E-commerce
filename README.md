# WARM THE HACK
## Analisis Sentimen Ulasan Produk E-commerce

---

### Disusun Oleh:
- Falah Razan Hibrizi
- Siti Nur Khaliza
- Dhifulloh Dhiya Ulhaq

**Data Research:** Central Computer Improvement  
**Institusi:** Telkom University, Bandung  
**Tahun:** 2025

---

## Overview

Proyek ini bertujuan untuk menganalisis sentimen dari ulasan produk pada platform e-commerce Tokopedia menggunakan teknik machine learning dan deep learning. Dengan meningkatnya volume ulasan online, analisis sentimen dapat membantu penjual dan pembeli dalam memahami persepsi terhadap produk secara otomatis dan efisien.

Aplikasi dashboard interaktif ini dibangun menggunakan Streamlit, memanfaatkan pipeline TFIDF, model XGBoost, dan LabelEncoder untuk klasifikasi sentimen (positif, netral, negatif) berdasarkan teks ulasan. Model dan pipeline dikembangkan dan dievaluasi menggunakan dataset ulasan Tokopedia yang telah dibersihkan dan diproses.

---

## Fitur Dashboard
- **Input Ulasan**: Pengguna dapat memasukkan ulasan produk secara manual atau memilih contoh ulasan.
- **Analisis Sentimen**: Hasil analisis sentimen (positif, netral, negatif) ditampilkan beserta tingkat kepercayaan model.
- **Visualisasi Confidence**: Terdapat grafik confidence score untuk setiap kelas sentimen.
- **Statistik Model**: Opsi untuk menampilkan metrik performa model (akurasi, presisi, recall, F1-score).
- **Upload Model**: Mendukung upload model utama (.pkl) secara custom.

---

## Cara Menjalankan Dashboard
1. **Instalasi dependensi** (pastikan Python 3.8+):
   ```bash
   pip install -r requirements.txt
   ```
2. **Pastikan file berikut tersedia di direktori utama:**
   - `app.py` (dashboard)
   - `tfidf_vectorizer.pkl` (vectorizer)
   - `label_encoder.pkl` (label encoder)
   - `xgb_sentiment_model.pkl` (model utama)
3. **Jalankan aplikasi Streamlit:**
   ```bash
   streamlit run app.py
   ```
4. **Akses dashboard** melalui browser di alamat yang tertera (default: http://localhost:8501)

---

## Metodologi Singkat
- **Data**: Dataset ulasan produk Tokopedia (sekitar 40.000+ data), diambil dari sumber publik.
- **Preprocessing**: Pembersihan teks, pemetaan rating ke label sentimen (1-2: negatif, 3: netral, 4-5: positif).
- **Feature Extraction**: TFIDF Vectorizer.
- **Model**: XGBoost untuk deployment (dashboard), IndoBERT untuk eksperimen di notebook.
- **Evaluasi**: Akurasi, presisi, recall, F1-score (contoh: akurasi model XGBoost ~85%).

---

## Struktur File
- `app.py` : Kode utama dashboard Streamlit
- `Okee_bismillah.ipynb` : Notebook eksperimen dan training model
- `tfidf_vectorizer.pkl`, `label_encoder.pkl`, `xgb_sentiment_model.pkl` : Model dan pipeline siap pakai
- `requirements.txt` : Daftar dependensi

---

## Credits & Acknowledgement
Proyek ini disusun untuk kompetisi WARM THE HACK oleh tim dari Central Computer Improvement, Telkom University. Data dan inspirasi model diambil dari berbagai sumber publik dan penelitian terkini di bidang NLP dan analisis sentimen.

---

Untuk detail lebih lanjut, silakan lihat notebook `Okee_bismillah.ipynb` atau hubungi tim penyusun.