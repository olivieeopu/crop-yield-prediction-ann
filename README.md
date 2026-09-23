# crop-yield-prediction-ann
Eksplorasi data pertanian dan prediksi hasil panen menggunakan ANN, mencakup data cleaning, feature engineering, serta perbandingan baseline dan arsitektur modifikasi dengan TensorFlow/Keras.

# Prediksi Hasil Panen dengan Artificial Neural Network

Proyek akademik untuk mengeksplorasi data pertanian dan membandingkan dua arsitektur Artificial Neural Network (ANN) dalam memprediksi hasil panen (`yield`). Proyek mencakup exploratory data analysis (EDA), data cleaning, feature engineering, preprocessing, pelatihan model, dan evaluasi regresi.

Model modifikasi menghasilkan error validasi sedikit lebih rendah dibandingkan baseline ANN. Namun, nilai R² masih mendekati nol sehingga kemampuan prediksi model pada eksperimen ini masih terbatas.

## Tujuan

- Mengidentifikasi masalah kualitas data dan mengeksplorasi hubungan fitur dengan hasil panen.
- Menyiapkan fitur numerik dan kategorikal untuk pemodelan ANN.
- Membandingkan baseline ANN dengan arsitektur modifikasi.
- Mengevaluasi hasil menggunakan MAE, MSE, RMSE, dan R² serta mengamati indikasi overfitting.

## Dataset

Dataset dibaca dari file `1A.csv` yang disediakan untuk tugas akademik Deep Learning. Sumber publik asli dan lisensi dataset belum tercantum dalam notebook.

- **Ukuran awal:** 500 observasi dan 22 kolom, termasuk target.
- **Target:** `yield`, dengan satuan hg/ha sesuai keterangan tugas.
- **Contoh fitur:** jenis tanaman, wilayah, kelembapan dan pH tanah, suhu, curah hujan, kelembapan udara, durasi penyinaran, metode irigasi, jenis pupuk, penggunaan pestisida, durasi pertumbuhan, lokasi, NDVI, dan status penyakit tanaman.

## Tools

Python, pandas, NumPy, Matplotlib, seaborn, SciPy, scikit-learn, TensorFlow/Keras.

## Alur Analisis

### 1. Exploratory Data Analysis

- Memeriksa struktur data, distribusi, missing values, duplikasi, dan nilai tidak valid.
- Membandingkan distribusi hasil panen antarkategori menggunakan boxplot dan statistik deskriptif.
- Mengeksplorasi hubungan numerik dengan korelasi Pearson dan perbedaan rata-rata antarkategori melalui one-way ANOVA.

### 2. Data Cleaning

- Menghapus `farm_id` dan `sensor_id` dari fitur pemodelan.
- Memperbaiki penulisan desimal pada `soil_pH` dan mengubahnya menjadi numerik.
- Menghapus `sowing_date`, `harvest_date`, dan `timestamp`; mempertahankan `total_days` sebagai fitur durasi.
- Mengisi missing values pada `irrigation_type` dan `crop_disease_status` dengan kategori `Unknown`.
- Mengisi missing values pada `sunlight_hours` dan `latitude` menggunakan mean.
- Mengganti satu nilai negatif pada `total_days` dengan missing value, kemudian melakukan imputasi mean.

### 3. Feature Engineering

Menambahkan tiga fitur interaksi eksploratif:

| Fitur | Perhitungan |
|---|---|
| `temp_humidity` | `temperature_C × humidity_%` |
| `growth_index` | `NDVI_index × sunlight_hours` |
| `rain_sun` | `rainfall_mm / (sunlight_hours + 1)` |

Fitur tersebut merupakan hipotesis pemodelan; manfaat masing-masing belum diuji melalui ablation study.

### 4. Pembagian Data dan Preprocessing

Data dibagi secara acak dengan `random_state=42`:

| Subset | Proporsi | Observasi |
|---|---:|---:|
| Training | 70% | 350 |
| Validation | 10% | 50 |
| Test | 20% | 100 |

Fitur numerik ditransformasi menggunakan `StandardScaler`, sedangkan fitur kategorikal menggunakan `OneHotEncoder`. Target juga distandardisasi untuk pelatihan, kemudian dikembalikan ke skala asli saat evaluasi.

Encoder dan scaler di-fit pada training set. Hasil preprocessing menghasilkan 30 fitur input. Namun, imputasi mean pada tahap cleaning dilakukan sebelum pembagian data; keterbatasan ini dijelaskan di bawah.

## Arsitektur Model

| Konfigurasi | Baseline ANN | ANN Modifikasi (`Tuned`) |
|---|---|---|
| Hidden layers | 64 → 64 | 128 → 64 → 32 |
| Aktivasi hidden layer | ReLU | ReLU |
| Output | Dense(1), linear | Dense(1), linear |
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.001 |
| Loss | MSE | MSE |
| Batch size | 32 | 16 |
| Maksimum epoch | 50 | 100 |

Kedua model menggunakan `EarlyStopping` dengan `monitor='val_loss'`, `patience=10`, dan `restore_best_weights=True`. Pada output yang tersimpan, baseline berhenti setelah 23 epoch dan model modifikasi setelah 13 epoch.

Istilah `Tuned` mengikuti penamaan notebook. Modifikasi yang ditampilkan berupa perubahan arsitektur dan batch size secara manual, bukan hasil pencarian otomatis Keras Tuner.

## Hasil Evaluasi

**Angka berikut berasal dari output notebook yang tersimpan dan dihitung pada validation set (50 observasi), bukan test set.** Prediksi dan target sudah dikembalikan ke skala asli sebelum metrik dihitung.

| Model | MAE (hg/ha) | MSE ((hg/ha)²) | RMSE (hg/ha) | R² |
|---|---:|---:|---:|---:|
| Baseline ANN | 10,016.72 | ≈131,838,000 | 11,482.07 | -0.0385 |
| ANN Modifikasi | 9,804.08 | ≈124,588,200 | 11,161.91 | 0.0186 |

Angka menggunakan titik sebagai desimal dan koma sebagai pemisah ribuan. MSE ditampilkan sesuai presisi output notebook.

### Temuan Utama

- Model modifikasi menurunkan RMSE validasi sekitar **2.79%** dibandingkan baseline ANN.
- R² meningkat dari negatif menjadi sekitar **0.019**, tetapi tetap menunjukkan kemampuan menjelaskan variasi target yang sangat terbatas. R² bukan persentase accuracy.
- Training loss menurun sementara validation loss kemudian meningkat, menunjukkan indikasi overfitting pada kedua model.
- Korelasi Pearson antara masing-masing fitur numerik awal dan target lemah. Seluruh uji ANOVA kategorikal yang ditampilkan memiliki p-value di atas 0.05. Hasil ini tidak membuktikan ketiadaan hubungan nonlinier atau interaksi antarfitur.

## Keterbatasan

- Imputasi mean dilakukan sebelum train-validation-test split, sehingga informasi dari validation/test set ikut memengaruhi nilai imputasi. Evaluasi perlu diulang dengan imputer yang hanya di-fit pada training set.
- Validation set digunakan untuk early stopping sekaligus pelaporan hasil. Test set sudah disiapkan, tetapi belum digunakan dalam tabel evaluasi akhir notebook.
- Dataset berukuran kecil dan hasil hanya berasal dari satu pembagian data. Belum ada cross-validation atau pengulangan eksperimen untuk menilai kestabilan hasil.
- Belum ada pembanding sederhana seperti `DummyRegressor`, regresi linear, atau model berbasis pohon.
- Seed Python dan NumPy sudah ditetapkan, tetapi seed TensorFlow belum ditetapkan dalam notebook. Hasil pelatihan ulang dapat berbeda.
- Ketersediaan fitur pada waktu prediksi, terutama `total_days`, perlu diperiksa sebelum model digunakan untuk prediksi sebelum panen.

Proyek ini merupakan eksperimen pembelajaran dan belum menunjukkan model yang siap digunakan untuk keputusan pertanian operasional.


## Pengembangan Selanjutnya

- Memindahkan imputasi ke pipeline preprocessing setelah data split.
- Menambahkan baseline sederhana dan memeriksa kualitas serta asal dataset.
- Memilih konfigurasi menggunakan training/validation set, lalu mengevaluasi model akhir pada test set yang disisihkan.
- Menguji kontribusi fitur interaksi dan mengulang eksperimen dengan beberapa seed.
