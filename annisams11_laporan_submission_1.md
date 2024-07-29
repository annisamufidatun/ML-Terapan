# Laporan Proyek Machine Learning - Annisa Mufidatun Sholihah

## Domain Proyek

![Diabetes effect](https://assets.mrmed.in/others/file-1632987112542-597612264-Side%20effects%20of%20Diabetes.jpeg?w=1920&q=75)

Diabetes melitus, umumnya dikenal sebagai diabetes, merupakan penyakit kronis yang serius. Kondisi ini muncul ketika kadar glukosa darah meningkat akibat ketidakmampuan tubuh menghasilkan insulin secara cukup, atau ketidakefektifan dalam menggunakan insulin yang diproduksi. Diabetes melitus menjadi ancaman global yang signifikan terhadap kesehatan, tanpa memandang status sosial ekonomi atau batas negara. Saat ini terdapat 463 juta orang dewasa yang hidup dengan diabetes melitus. Jika tidak ada langkah-langkah yang tepat untuk mengatasi ini, diperkirakan jumlah penderita akan mencapai 578 juta pada tahun 2030. Lebih mengkhawatirkan lagi, angka tersebut diprediksi akan melonjak hingga 700 juta pada tahun 2045 [1].

Meskipun prevalensi diabetes terus meningkat, metode konvensional untuk deteksi dini dan intervensi sering kali terbukti tidak efektif. Skrining tradisional, seperti tes glukosa darah puasa atau tes toleransi glukosa oral, memiliki beberapa keterbatasan. Mereka memerlukan kunjungan ke fasilitas kesehatan, persiapan khusus pasien, dan seringkali mahal untuk dilakukan secara massal. Akibatnya, banyak individu yang berisiko tinggi tidak terdeteksi sampai penyakit mereka sudah berkembang, menyebabkan komplikasi yang serius dan meningkatkan beban pada sistem kesehatan [2].

Dalam konteks ini, machine learning menawarkan pendekatan yang menjanjikan untuk meningkatkan deteksi dini dan intervensi diabetes. Model prediktif berbasis machine learning dapat menganalisis berbagai faktor risiko secara simultan, termasuk gejala yang tampaknya tidak terkait, riwayat medis, dan data gaya hidup, untuk mengidentifikasi pola yang mungkin terlewatkan oleh metode konvensional. Studi oleh Zou et al. (2018) menunjukkan bahwa model machine learning dapat memprediksi onset diabetes hingga 5 tahun sebelumnya dengan akurasi yang lebih tinggi dibandingkan metode tradisional [3].

Selain itu, implementasi machine learning dalam skrining diabetes berpotensi meningkatkan aksesibilitas dan efisiensi deteksi dini. Model prediktif dapat diintegrasikan ke dalam aplikasi mobile atau platform online, memungkinkan penilaian risiko awal tanpa perlu kunjungan langsung ke fasilitas kesehatan. Hal ini dapat secara signifikan meningkatkan jangkauan program skrining, terutama di daerah dengan akses terbatas ke layanan kesehatan [4].

Dengan mengembangkan model prediksi diabetes menggunakan data penderita diabetes, diharapkan dapat membantu dalam mendeteksi pola-pola yang mungkin tidak terlihat oleh metode konvensional. Pendekatan ini dapat memberikan kontribusi signifikan dalam mengidentifikasi individu yang berisiko tinggi menderita diabetes, sehingga intervensi dapat dilakukan lebih awal, potensial mengurangi komplikasi jangka panjang dan meningkatkan kualitas hidup pasien.

## Business Understanding


### Problem Statements

Diabetes melitus, terutama Diabetes Tipe 2, menjadi ancaman kesehatan global dengan prevalensi yang terus meningkat, mempengaruhi jutaan orang di seluruh dunia tanpa memandang status sosial ekonomi. Deteksi dini dan intervensi tepat waktu sangat penting untuk mencegah komplikasi serius dan mengurangi beban ekonomi pada sistem kesehatan. Namun, metode konvensional seringkali gagal dalam mengidentifikasi individu yang berisiko tinggi pada tahap awal, menyebabkan keterlambatan dalam penanganan dan perawatan.

### Goals

Mengembangkan model machine learning yang mampu memprediksi risiko diabetes melitus pada tahap awal dengan menganalisis data pasien, sehingga memungkinkan intervensi dini yang efektif dan personalisasi perawatan untuk mengurangi prevalensi dan komplikasi penyakit ini.


### Solution statements
Untuk mencapai tujuan tersebut, dibuat model machine learning dengan dua metode yaitu 

*   Solution 1: Logistic Regression
*   Solution 2: Random Forest

Kedua model akan diukur  dengan metrik evaluasi akurasi, presisi, recall, dan F1-score. Dengan menggunakan kedua solusi ini, kita dapat membandingkan kinerja model logistic regression dan random forest untuk menentukan pendekatan mana yang lebih efektif dalam memprediksi risiko diabetes pada tahap awal.

## Data Understanding

**Dataset Early Stage Diabetes Risk Prediction Dataset**
Dataset ini berisi informasi mengenai tanda dan gejala pasien diabetes yang baru terdiagnosis atau mereka yang berisiko terkena diabetes. Data dikumpulkan melalui kuesioner langsung yang diberikan kepada pasien di Rumah Sakit Diabetes Sylhet di Sylhet, Bangladesh, dan disetujui oleh dokter. Dataset ini berisi 520 baris data dengan 17 fitur.
[Link to dataset](https://www.kaggle.com/datasets/abdelazizsami/early-stage-diabetes-risk-prediction).

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:

- **Age** (integer): Umur dari pasien.
- **Gender** (categorical): Jenis kelamin pasien.
- **Polyuria** (binary): Apakah pasien buang air kecil lebih sering dibanding hari biasanya.
- **Polydipsia** (binary): Apakah pasien merasa haus yang tidak berkesudahan.
- **Sudden Weight Loss** (binary): Apakah pasien mengalami penurunan berat badan yang signifikan.
- **Weakness** (binary): Apakah pasien mengalami kelemahan.
- **Polyphagia** (binary): Apakah pasien merasa lapar ekstrem yang tidak terpuaskan meskipun sudah makan.
- **Genital Thrush** (binary): Apakah pasien mengalami infeksi jamur pada genital/alat kelamin.
- **Visual Blurring** (binary): Apakah pasien penglihatannya kabur.
- **Itching** (binary): Apakah pasien mengalami gatal.
- **Irritability** (binary): Apakah pasien merasakan perasaan gelisah yang mungkin Anda alami akibat stres, kondisi kesehatan mental, atau gangguan fisik.
- **Delayed Healing** (binary): Apakah pasien mengalami penyembuhan luka yang lebih lambat dari biasanya.
- **Partial Paresis** (binary): Apakah pasien mengalami kelemahan atau kelumpuhan sebagian pada area tubuh tertentu.
- **Muscle Stiffness** (binary): Apakah pasien merasakan kekakuan atau ketegangan pada otot-otot tubuh.
- **Alopecia** (binary): Apakah pasien mengalami kerontokan rambut yang tidak normal atau kebotakan.
- **Obesity** (binary): Apakah pasien memiliki berat badan berlebih atau obesitas.
- **Class** (binary): Parameter apakah pasien mengalami diabetes atau tidak.


**Exploratory Data Analysis**

Untuk memahami data, dilakukan beberapa cara yaitu 
- melihat tipe data setiap fitur dengan **data.info()**
- melihat informasi statistik dari fitur numerik yaitu age dengan **data.describe()**
- melihat apakah ada nilai null dalam dataset dengan **data.isnull().sum()**
- Melakukan univariate analysis dengan melakukan visualisasi data pada setiap fitur. Untuk fitur numerik digunakan histogram dan data kategorikal dengan bar chart.
Hasil EDA

![Hasil](https://raw.githubusercontent.com/annisamufidatun/ML-Terapan/main/eda_result.png)


## Data Preparation
Pada dataset dilakukan beberapa proses berikut
1.  Encoding
    Encoding dilakukan karena model machine learning membutuhkan input data dalam bentuk numerik untuk melakukan perhitungan dan prediksi.Encoding dilakukan pada categorical feature yaitu Gender, Polyuria,Polydipsia, sudden weight loss, weakness, Polyphagia, Genital thrush, visual blurring, Itching, Irritability, delayed healing, partial paresis, muscle stiffness, Alopecia, Obesity, dan class
    Encoding dilakukan sebagai berikut
    Yes = 1; No = 0
    Male = 0; Female = 1
    Negative = 0; Positive = 1

2.  Menghapus outlier
    Pada feature age outlier dihapus. Outlier dapat mempengaruhi kinerja model machine learning dengan membuat model menjadi terlalu kompleks atau terlalu sederhana. Menghapus outlier dapat membantu model untuk generalisasi lebih baik pada data yang sebenarnya, meningkatkan akurasi prediksi.
    Ada 4 baris data yang dihapus yaitu data dengan umur 85 dan 90.
    
3.  Standar Scaler
    Pada fitur umur dilakukan standar scaler, mengubah fitur sehingga memiliki rata-rata 0 dan standar deviasi 1, untuk meningkatkan kinerja dan konvergensi model.

4.  Membagi dataset menjadi 80% data training dan 20% data testing


## Modeling

Metode yang digunakan untuk membangun model adalah **logistic regression** dan **random forest**.

### **Kelebihan dan kekurangan model**
Logistic Regression
Kelebihan:
- Efisien Secara Komputasi:
Logistic Regression relatif cepat untuk dilatih dan dieksekusi, bahkan pada dataset yang besar.
- Baik untuk Data Linear
Logistic Regression bekerja sangat baik ketika hubungan antara fitur dan label adalah linear atau mendekati linear.

Kekurangan:
- Asumsi Independen Fitur:
Logistic Regression mengasumsikan bahwa fitur-fitur input independen, yang tidak selalu sesuai dengan kenyataan.
- Sensitif terhadap Outlier
Model ini dapat dipengaruhi oleh outlier, yang dapat mendistorsi hasil prediksi.

Random Forest
Kelebihan:
- Mengurangi Overfitting
Dengan menggabungkan prediksi dari banyak pohon keputusan (trees), Random Forest mengurangi risiko overfitting yang sering terjadi pada pohon keputusan tunggal.
- Robust terhadap Outlier
Karena menggunakan banyak pohon, Random Forest lebih robust terhadap outlier dibandingkan dengan model lain.
- Feature Importance
Random Forest memberikan informasi tentang pentingnya fitur-fitur dalam prediksi, yang berguna untuk interpretasi dan pemahaman model.

Kekurangan:
- Kompleks dan Sulit Diinterpretasi
Random Forest adalah model yang kompleks dan sulit untuk diinterpretasi dibandingkan dengan model yang lebih sederhana seperti Logistic Regression.
- Waktu dan Sumber Daya Komputasi
Random Forest membutuhkan lebih banyak waktu dan sumber daya komputasi untuk dilatih dan dieksekusi, terutama pada dataset yang besar.
- Memori yang Dibutuhkan:
Karena menyimpan banyak pohon keputusan, Random Forest bisa sangat intensif dalam penggunaan memori.

### **Pengerjaan Model**
Dalam proyek ini, saya mengembangkan dua model machine learning untuk memprediksi risiko diabetes: Logistic Regression dan Random Forest. Berikut adalah penjelasan tentang proses dan tahapan pemodelan, konfirgurasi hyperparameter, serta cara kerja algoritma pada data:

**Proses dan Tahapan Pemodelan**
1.  Persiapan Data:
    Sebelum pemodelan, data dibagi menjadi set pelatihan (X_train, y_train) dan set pengujian (X_test, y_test) menggunakan fungsi train_test_split dari scikit-learn.
2.  Definisi Fungsi Evaluasi:
    Saya membuat fungsi evaluate_model yang akan digunakan untuk melatih model, membuat prediksi, dan menghitung metrik evaluasi (akurasi, presisi, recall, dan F1-score).

3.  Pembuatan dan Evaluasi Model:
    Saya membuat instance dari masing-masing model dan mengevaluasinya menggunakan fungsi evaluate_model.

**Cara Kerja Algoritma**
Logistic Regression
Logistic Regression bekerja dengan membangun model linear dari fitur input, kemudian menerapkan fungsi sigmoid untuk menghasilkan probabilitas kelas output. Model ini cocok untuk masalah klasifikasi biner seperti prediksi diabetes.

Proses:
Model membangun kombinasi linear dari fitur input.
Fungsi sigmoid diterapkan pada kombinasi linear ini untuk menghasilkan probabilitas.
Jika probabilitas > 0.5, prediksi adalah kelas positif (diabetes), jika tidak, kelas negatif (non-diabetes).

Random Forest
Random Forest adalah ensemble dari pohon keputusan. Setiap pohon dibangun menggunakan subset acak dari fitur dan data pelatihan.

Proses:
Beberapa pohon keputusan dibangun, masing-masing menggunakan subset acak dari data dan fitur.
Setiap pohon membuat prediksi independen.
Prediksi akhir adalah hasil voting mayoritas dari semua pohon.

### Konfigurasi Hyperparameter

Dalam implementasi ini, saya menggunakan konfigurasi hyperparameter tertentu untuk kedua model. Berikut adalah penjelasan detail tentang konfigurasi yang digunakan:

1. Logistic Regression:
   ```p
   logistic_regression_model = LogisticRegression(random_state=55)
   ```
   - random_state=55: Ini menetapkan seed untuk generator angka acak, memastikan hasil yang dapat direproduksi. Menggunakan nilai yang sama akan menghasilkan inisialisasi yang sama setiap kali model dijalankan.
   
   Konfigurasi lainnya menggunakan nilai default, termasuk:
   - C=1.0 (inverse of regularization strength)
   - solver='lbfgs' (algoritma optimisasi)
   - max_iter=100 (jumlah maksimum iterasi)

2. Random Forest:
   ```
   random_forest_model = RandomForestClassifier(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
   ```
   - n_estimators=50: Jumlah pohon keputusan dalam forest. Nilai ini lebih rendah dari default (100), yang dapat mempercepat waktu pelatihan dengan sedikit pengurangan pada kinerja.
   - max_depth=16: Kedalaman maksimum setiap pohon. Membatasi kedalaman dapat membantu mencegah overfitting, tetapi nilai 16 masih cukup tinggi untuk menangkap pola kompleks dalam data.
   - random_state=55: Seperti pada Logistic Regression, ini memastikan hasil yang dapat direproduksi.
   - n_jobs=-1: Ini menggunakan semua core CPU yang tersedia untuk pelatihan paralel, yang dapat sangat mempercepat proses pelatihan terutama pada dataset besar.

Konfigurasi ini menunjukkan beberapa pertimbangan penting dalam pemodelan:

1. Keseimbangan antara kecepatan dan kinerja: Penggunaan 50 estimator alih-alih nilai default 100 dapat mempercepat pelatihan dengan hanya sedikit pengurangan pada kinerja potensial.
2. Kontrol overfitting: Pembatasan max_depth dapat membantu mencegah model dari overfitting terhadap data pelatihan.
3. Reproduktifitas: Penggunaan random_state yang konsisten memastikan bahwa eksperimen dapat direproduksi.
4. Efisiensi komputasi: Penggunaan n_jobs=-1 memanfaatkan seluruh kapasitas komputasi yang tersedia.

Perlu dicatat bahwa meskipun konfigurasi ini telah dipilih dengan pertimbangan tertentu, selalu ada ruang untuk optimalisasi lebih lanjut. Teknik seperti Grid Search, Random Search, atau bahkan metode optimisasi Bayesian bisa digunakan untuk menemukan kombinasi hyperparameter yang optimal untuk dataset spesifik ini.

## Evaluation
Untuk menganalisis hasil model digunakna metrik **akurasi, precision, recall, dan F1 score**. 

#### Akurasi (Accuracy)
- **Definisi**: Akurasi mengukur proporsi prediksi yang benar terhadap keseluruhan prediksi.
- **Formula**: 
  `Akurasi = (True Positives (TP) + True Negatives (TN)) / Total Predictions`

#### Precision (Presisi)

- **Definisi**: Presisi mengukur proporsi prediksi positif yang benar terhadap semua prediksi positif.
- **Formula**: 
  `Presisi = True Positives (TP) / (True Positives (TP) + False Positives (FP))`


#### Recall (Recall) atau Sensitivitas

- **Definisi**: Recall mengukur proporsi prediksi positif yang benar terhadap semua sampel yang sebenarnya positif.
- **Formula**: 
  `Recall = True Positives (TP) / (True Positives (TP) + False Negatives (FN))`


#### F1 Score

- **Definisi**: F1 Score adalah rata-rata harmonis dari presisi dan recall, yang memberikan keseimbangan antara keduanya.
- **Formula**: 
  `F1 Score = 2 * (Presisi * Recall) / (Presisi + Recall)`

Untuk menganalisis hasil model, digunakan metrik akurasi, presisi, recall, dan F1 score. Berikut adalah hasil evaluasi kedua model:

![Hasil](https://raw.githubusercontent.com/annisamufidatun/ML-Terapan/main/result.png)

Dari hasil evaluasi, dapat dilihat bahwa model Random Forest menunjukkan kinerja yang lebih baik dibandingkan Logistic Regression, dengan akurasi 99% dibandingkan 91.4%.

**Dampak terhadap Business Understanding**:

1. Menjawab Problem Statement:
   Model yang dikembangkan berhasil mengatasi masalah deteksi dini diabetes yang sering kali gagal diidentifikasi oleh metode konvensional. Dengan akurasi di atas 97%, model ini dapat mengidentifikasi individu yang berisiko tinggi pada tahap awal dengan tingkat keakuratan yang tinggi.

2. Pencapaian Goals:
   Tujuan untuk mengembangkan model machine learning yang mampu memprediksi risiko diabetes melitus pada tahap awal telah tercapai. Model Random Forest yang dihasilkan memiliki kemampuan prediksi yang sangat baik, memungkinkan intervensi dini yang lebih efektif.

3. Dampak Solution Statement:
   Kedua solusi yang diajukan (Logistic Regression dan Random Forest) terbukti efektif, dengan Random Forest menunjukkan kinerja yang sedikit lebih unggul. Ini memvalidasi pendekatan machine learning dalam prediksi risiko diabetes.

**Perbandingan dengan Metode Konvensional**:
Model machine learning yang dikembangkan memiliki beberapa keunggulan dibandingkan metode konvensional:

1. Akurasi Tinggi: Dengan akurasi di atas 99%, model ini melebihi akurasi metode skrining konvensional seperti Fasting Plasma Glucose (FPG) yang memiliki sensitivitas sekitar 50-60% [5].
2. Efisiensi: Model ini dapat memberikan hasil prediksi secara instan tanpa memerlukan tes laboratorium yang memakan waktu dan biaya.
3. Aksesibilitas: Model dapat diintegrasikan ke dalam aplikasi mobile atau platform online, meningkatkan jangkauan skrining diabetes ke daerah-daerah yang sulit dijangkau oleh fasilitas kesehatan konvensional.

Potensi Dampak pada Prevalensi Diabetes:
Implementasi model prediksi ini berpotensi mengurangi prevalensi diabetes melalui:

1. Deteksi Dini: Identifikasi individu berisiko tinggi pada tahap awal memungkinkan intervensi yang lebih cepat, potensial mencegah atau menunda onset diabetes.
2. Personalisasi Intervensi: Hasil prediksi dapat membantu penyedia layanan kesehatan dalam merancang intervensi yang lebih tepat sasaran dan personal.
3. Peningkatan Kesadaran: Penggunaan model prediksi dapat meningkatkan kesadaran masyarakat tentang faktor risiko diabetes, mendorong gaya hidup lebih sehat.

Studi simulasi menunjukkan bahwa implementasi skrining berbasis machine learning secara luas dapat mengurangi prevalensi diabetes tipe 2 hingga 10-15% dalam jangka waktu 5 tahun [6].

Berikut merupakan hasil percobaan dengan data testing
![Prediksi](https://raw.githubusercontent.com/annisamufidatun/ML-Terapan/main/prediksi.png)

Kesimpulan:
Model prediksi diabetes berbasis machine learning yang dikembangkan dalam proyek ini menunjukkan potensi signifikan dalam meningkatkan deteksi dini dan manajemen diabetes. Dengan akurasi tinggi, efisiensi, dan aksesibilitas yang ditawarkan, model ini dapat menjadi alat yang berharga dalam upaya global untuk mengurangi beban diabetes. Namun, perlu dicatat bahwa efektivitas sebenarnya dari model ini dalam setting klinis masih perlu dievaluasi melalui uji coba klinis yang komprehensif.


## Referensi

[1] [IDF. 2019. IDF Diabetes Atlas 9th Edition.](https://diabetesatlas.org/atlas/ninth-edition/)
[2] [American Diabetes Association. (2019). 2. Classification and Diagnosis of Diabetes: Standards of Medical Care in Diabetes—2019. Diabetes Care, 42(Supplement 1), S13-S28.](https://diabetesjournals.org/care/article/42/Supplement_1/S13/31150/2-Classification-and-Diagnosis-of-Diabetes)
[3] [Zou, Q., Qu, K., Luo, Y., Yin, D., Ju, Y., & Tang, H. (2018). Predicting diabetes mellitus with machine learning techniques. Frontiers in genetics, 9, 515.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6232260/)
[4] [Kavakiotis, I., Tsave, O., Salifoglou, A., Maglaveras, N., Vlahavas, I., & Chouvarda, I. (2017). Machine learning and data mining methods in diabetes research. Computational and structural biotechnology journal, 15, 104-116.](https://www.sciencedirect.com/science/article/pii/S2001037016300733)
[5] [Sacks, D. B. (2011). A1C versus glucose testing: a comparison. Diabetes care, 34(2), 518-523.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3024379/)
[6] [Zheng, Y., Ley, S. H., & Hu, F. B. (2018). Global aetiology and epidemiology of type 2 diabetes mellitus and its complications. Nature Reviews Endocrinology, 14(2), 88-98.](https://pubmed.ncbi.nlm.nih.gov/29219149/)


**---Ini adalah bagian akhir laporan---**



