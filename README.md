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

Proyek ini bertujuan untuk menganalisis sentimen dari ulasan produk pada platform e-commerce Tokopedia menggunakan teknik machine learning ataupun deep learning. Dengan meningkatnya volume ulasan online, analisis sentimen dapat membantu penjual dan pembeli dalam memahami persepsi terhadap produk secara otomatis dan efisien.

Aplikasi dashboard interaktif ini dibangun menggunakan Streamlit, memanfaatkan pipeline TFIDF, model XGBoost, dan LabelEncoder untuk klasifikasi sentimen (positif, netral, negatif) berdasarkan teks ulasan. Model dan pipeline dikembangkan dan dievaluasi menggunakan dataset ulasan Tokopedia yang telah dibersihkan dan diproses.

---

## Fitur Dashboard
- **Input Ulasan**: Pengguna dapat memasukkan ulasan produk secara manual atau memilih contoh ulasan.
- **Analisis Sentimen**: Hasil analisis sentimen (positif, netral, negatif) ditampilkan beserta tingkat kepercayaan model.
- **Visualisasi Confidence**: Terdapat grafik confidence score untuk setiap kelas sentimen.
- **Statistik Model**: Opsi untuk menampilkan metrik performa model (akurasi, presisi, recall, F1-score).
- **Upload Model**: Mendukung upload model utama (.pkl untuk ML, .safetensors untuk DL) secara custom.
- **Tautan Antar Dashboard**: Terdapat tombol/link di bagian atas untuk berpindah antara dashboard ML dan DL.

---

## Alur Penggunaan Dashboard
1. **Pemilihan Model**
   - Pengguna dapat memilih untuk menggunakan model default (bawaan) yang sudah tersedia di sistem, atau meng-upload model mereka sendiri (misal, model hasil training lanjutan).
   - Untuk model Machine Learning, file yang di-upload berupa `.pkl`, sedangkan untuk model Deep Learning IndoBERT berupa `.safetensors`.
2. **Input Review**
   - Pada kolom “Enter Your Review”, pengguna dapat menuliskan review atau komentar mereka terhadap suatu produk.
   - Alternatifnya, pengguna dapat memilih salah satu sample review (positif, netral, atau negatif) yang telah disediakan di sidebar untuk melihat contoh hasil prediksi.
3. **Analisis Sentimen**
   - Setelah review diinput, pengguna cukup menekan tombol **Analyze Sentiment**.
   - Dashboard akan memproses input tersebut menggunakan model yang dipilih, lalu menampilkan hasil prediksi sentimen beserta tingkat kepercayaan (confidence score) dari model.
4. **Visualisasi & Interpretasi**
   - Hasil prediksi ditampilkan secara visual, baik dalam bentuk label sentimen, confidence score, maupun grafik distribusi probabilitas untuk masing-masing kelas sentimen.
   - Terdapat juga interpretasi tingkat keyakinan model terhadap prediksi yang dihasilkan.
5. **Fitur Tambahan**
   - Pengguna dapat melihat statistik performa model (seperti akurasi, precision, recall, F1-score) melalui opsi tambahan di dashboard.
   - Terdapat tautan untuk mencoba dashboard versi lain (ML atau DL) secara terpisah.

---

## Cara Menjalankan Dashboard
1. **Instalasi dependensi** (direkomendasikan Python 3.9):
    streamlit==1.28.0
    torch==2.0.1
    transformers==4.33.0
    pandas==2.2.2
    numpy==2.0.2
    scikit-learn==1.3.0
    plotly==5.15.0
    datasets==2.14.0
   ```bash
   pip install -r requirements.txt
   ```
2. **Pastikan file berikut tersedia di direktori utama:**
   - `app_ML_XGBOOST.py` (dashboard Machine Learning/XGBoost)
   - `app_DL_INDOBERT.py` (dashboard Deep Learning IndoBERT)
   - `Load Model XGBOOST/tfidf_vectorizer.pkl` (vectorizer ML)
   - `Load Model XGBOOST/label_encoder.pkl` (label encoder ML)
   - `Load Model XGBOOST/xgb_sentiment_model.pkl` (model utama ML)
   - `Load Model INDOBERT/model.safetensors` (model IndoBERT)
   - `Load Model INDOBERT/config.json`, `Load Model INDOBERT/vocab.txt`, `Load Model INDOBERT/tokenizer.json`, `Load Model INDOBERT/tokenizer_config.json`, `Load Model INDOBERT/special_tokens_map.json` (tokenizer IndoBERT)
   - `Load Model INDOBERT/label_encoder.pkl` (label encoder DL)
3. **Jalankan aplikasi Streamlit:**
   - Untuk dashboard Machine Learning/XGBoost:
     ```bash
     streamlit run app_ML_XGBOOST.py
     ```
   - Untuk dashboard Deep Learning IndoBERT:
     ```bash
     streamlit run app_DL_INDOBERT.py
     ```
4. **Akses dashboard** melalui browser di alamat yang tertera (http://localhost). Untuk berpindah antar dashboard, gunakan tautan/link yang tersedia di bagian atas masing-masing aplikasi.

---

## Metodologi Singkat
- **Data**: Dataset ulasan produk Tokopedia (sekitar 40.000+ data), diambil dari sumber publik.
- **Preprocessing**: Pembersihan teks, pemetaan rating ke label sentimen (1-2: negatif, 3: netral, 4-5: positif).
- **Feature Extraction**: TFIDF Vectorizer.
- **Model**: XGBoost untuk deployment (dashboard), IndoBERT untuk eksperimen di notebook.
- **Evaluasi**: Akurasi, presisi, recall, F1-score (contoh: akurasi model XGBoost ~85%).

---

## Struktur File
- `app_ML_XGBOOST.py` : Kode utama dashboard Streamlit untuk Machine Learning/XGBoost
- `app_DL_INDOBERT.py` : Kode utama dashboard Streamlit untuk Deep Learning IndoBERT
- `Load Model XGBOOST/` : Folder model dan pipeline ML (XGBoost)
- `Load Model INDOBERT/` : Folder model dan pipeline Deep Learning (IndoBERT)
- `requirements.txt` : Daftar dependensi

---

## Credits & Acknowledgement
Proyek ini disusun sebagai bentuk WARM THE HACK untuk persiapan lomba yang akan diadakan oleh tim Central Computer Improvement, Telkom University. Data dan inspirasi model diambil dari berbagai sumber publik dan penelitian terkini di bidang NLP dan analisis sentimen.

---

Untuk detail lebih lanjut, silakan lihat notebook `Okee_bismillah.ipynb` atau hubungi tim penyusun.