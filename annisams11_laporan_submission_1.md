# Laporan Proyek Machine Learning - Annisa Mufidatun Sholihah

## Domain Proyek

![Diabetes effect](https://assets.mrmed.in/others/file-1632987112542-597612264-Side%20effects%20of%20Diabetes.jpeg?w=1920&q=75)

Diabetes melitus, umumnya dikenal sebagai diabetes, merupakan penyakit kronis yang serius. Kondisi ini muncul ketika kadar glukosa darah meningkat akibat ketidakmampuan tubuh menghasilkan insulin secara cukup, atau ketidakefektifan dalam menggunakan insulin yang diproduksi. Diabetes melitus menjadi ancaman global yang signifikan terhadap kesehatan, tanpa memandang status sosial ekonomi atau batas negara. Saat ini terdapat 463 juta orang dewasa yang hidup dengan diabetes melitus. Jika tidak ada langkah-langkah yang tepat untuk mengatasi ini, diperkirakan jumlah penderita akan mencapai 578 juta pada tahun 2030. Lebih mengkhawatirkan lagi, angka tersebut diprediksi akan melonjak hingga 700 juta pada tahun 2045 [1].

Diabetes Melitus adalah kondisi yang mengerikan menurut laporan WHO tahun 2016, langkah-langkah penting diperlukan untuk mencegah dan mengobati penyakit ini. Diabetes Tipe I adalah jenis Diabetes paling umum yang terjadi pada kelompok usia yang lebih muda. Penyakit ini meningkat pesat di seluruh dunia karena perubahan gaya hidup dan pola makan yang tidak sehat. Menurut perkiraan Federasi Diabetes Internasional (IDF), prevalensi Diabetes melitus kemungkinan akan meningkat setiap tahunnya. Kasus Diabetes Tipe 2 menjadi lebih menonjol pada usia yang lebih muda dan negara-negara berkembang, mencakup 85-95% dari pasien Diabetes melitus. Penyakit ini terjadi pada kelompok usia antara 18-75 tahun; sekitar 285 juta orang menderita Diabetes di seluruh dunia, menurut survei yang dilakukan pada tahun 2010. Pada tahun 2025, diperkirakan 438 juta orang di negara-negara berkembang akan meningkat dengan tingkat 60-75% dan meningkatkan angka kematian hingga sekitar 60%, yang bisa berakibat fatal [2].

Prediksi dini diabetes melitus sangat penting untuk mencegah komplikasi serius yang bisa terjadi akibat penyakit ini. Dengan deteksi dini, pasien dapat menerima intervensi yang lebih cepat, seperti perubahan gaya hidup, pengelolaan diet, dan terapi medis yang sesuai, yang semuanya dapat memperlambat atau bahkan menghentikan perkembangan penyakit. Selain itu, prediksi dini dapat membantu mengurangi beban ekonomi pada sistem kesehatan dengan menurunkan angka rawat inap dan perawatan intensif yang mahal. Mengingat angka prevalensi yang terus meningkat, kemampuan untuk memprediksi diabetes melitus pada tahap awal menjadi semakin mendesak untuk mengendalikan epidemi global ini.

Machine learning sangat membantu untuk melakukan prediksi dini diabetes. Dengan mengembangkan model prediksi diabetes menggunakan data penderita diabetes, diharapkan dapat membantu dalam mendeteksi pola-pola yang mungkin tidak terlihat oleh metode konvensional. Pendekatan ini dapat memberikan kontribusi signifikan dalam mengidentifikasi individu yang berisiko tinggi menderita diabetes, sehingga intervensi dapat dilakukan lebih awal.

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
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

Metode yang digunakan untuk membangun model adalah **logistic regression** dan **random forest**.
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


Kedua model dievaluasi dengan metrik evaluasi akurasi, presisi, recall, dan F1-score dan didapatkan hasil sebagai berikut
![Hasil](https://raw.githubusercontent.com/annisamufidatun/ML-Terapan/main/result.png)

Dapat dilihat bahwa model dengan menggunakan metode random forest memiliki hasil yang lebih baik
Berikut merupakan hasil percobaan dengan data testing
![Prediksi](https://raw.githubusercontent.com/annisamufidatun/ML-Terapan/main/prediksi.png)

Dapat dilihat bahwa hasil prediksi model yang dibuat sama dengan data testing yang artinya model efektif untuk melakukan prediksi dini diabetes melitus.

## Referensi

[1] [IDF. 2019. IDF Diabetes Atlas 9th Edition.](https://diabetesatlas.org/atlas/ninth-edition/)
[2] [Ahuja, Ashima & Gupta, Reena & Gupta, Jitendra. (2020). Diabetes Silent Killer: Medical focus on Food Replacement and Dietary Plans. Advances in Bioresearch. 11. 128-135. 10.15515/abr.0976-4585.11.5.128135. ](https://www.researchgate.net/publication/344901853_Diabetes_Silent_Killer_Medical_focus_on_Food_Replacement_and_Dietary_Plans)


**---Ini adalah bagian akhir laporan---**



