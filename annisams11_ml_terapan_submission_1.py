# -*- coding: utf-8 -*-
"""ML Terapan - Submission 1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1k_7HTmRpPdwp5Ih5Br4GIHBgLfGcqaR4

# **Predictive Analysis** - Early Stage Diabetes Risk Prediction

<p align="right">by Annisa Mufidatun Sholihah</p>

## Problem Background

![Diabetes effect](https://assets.mrmed.in/others/file-1632987112542-597612264-Side%20effects%20of%20Diabetes.jpeg?w=1920&q=75)


Prediksi Risiko Diabetes pada tahap awal sangat penting karena memungkinkan intervensi dini yang dapat secara signifikan meningkatkan hasil kesehatan. Berikut adalah beberapa alasan mengapa ini penting:

1. **Pencegahan Perkembangan Penyakit**: Dengan mengidentifikasi individu yang berisiko mengembangkan diabetes sejak dini, langkah-langkah pencegahan dapat diterapkan untuk menunda atau mencegah timbulnya penyakit tersebut.

2. **Modifikasi Gaya Hidup**: Prediksi dini memungkinkan individu untuk melakukan perubahan gaya hidup yang diperlukan, seperti memperbaiki pola makan, meningkatkan aktivitas fisik, dan menjaga berat badan yang sehat, yang sangat penting untuk mengurangi risiko diabetes.

3. **Efisiensi Biaya**: Deteksi dan pencegahan dini umumnya lebih hemat biaya daripada mengobati diabetes dan komplikasinya. Ini dapat mengurangi beban finansial pada individu dan sistem kesehatan.

4. **Peningkatan Kualitas Hidup**: Intervensi dini dapat mencegah atau menunda komplikasi yang terkait dengan diabetes, seperti penyakit kardiovaskular, gagal ginjal, dan kerusakan saraf, sehingga meningkatkan kualitas hidup secara keseluruhan.

5. **Intervensi yang Tepat Sasaran**: Penyedia layanan kesehatan dapat menawarkan rencana perawatan yang dipersonalisasi dan memantau pasien dengan lebih efektif ketika mereka mengetahui faktor risiko sejak awal.

Dengan fokus pada prediksi risiko diabetes pada tahap awal, sistem kesehatan dapat beralih dari pendekatan reaktif ke proaktif yang pada akhirnya mengarah pada hasil kesehatan yang lebih baik dan penurunan morbiditas serta mortalitas terkait diabetes.

**Goals (Tujuan)**
Tujuan dari proyek ini adalah untuk mengembangkan model prediktif yang akurat untuk mengidentifikasi individu dengan risiko tinggi terkena diabetes pada tahap awal. Dengan memprediksi risiko ini, kita dapat:
1. Memberikan informasi yang dapat digunakan untuk intervensi kesehatan yang proaktif.
2. Meningkatkan kualitas hidup individu dengan mencegah perkembangan penyakit.
3. Mengurangi biaya perawatan kesehatan yang terkait dengan pengobatan diabetes dan komplikasinya.

**Solution Statement**

Untuk mencapai tujuan tersebut, kita mengajukan beberapa solusi yang dapat diukur dengan metrik evaluasi seperti akurasi, presisi, recall, dan F1-score.


*   Solution 1: Logistic Regression
*   Solution 2: Random Forest

Dengan menggunakan kedua solusi ini, kita dapat membandingkan kinerja model logistic regression dan random forest untuk menentukan pendekatan mana yang lebih efektif dalam memprediksi risiko diabetes pada tahap awal.

## **Import library dan dataset**

Dibagian ini library yang diperlukan diimport
"""

# import library
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

"""Dataset diambil langsung dari kaggle dengan import"""

#import kaggle dataset
!pip install kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d abdelazizsami/early-stage-diabetes-risk-prediction
!unzip early-stage-diabetes-risk-prediction.zip

"""Link to dataset: [diabetes dataset](https://www.kaggle.com/datasets/abdelazizsami/early-stage-diabetes-risk-prediction)

Melihat isi dataset yang sudah diimport
"""

df = pd.read_csv('/content/diabetes_data_upload.csv')
df

"""ada 520 baris data dan 17 feature/variabel pada dataset

## **Exploratory Data Analysis**

Pertama, cek features dalam dataset beserta tipe datanya
"""

#list feature di dataset
df.info()

"""Kemudian untuk fitur numerik kita lihat statistic summarynya"""

df.describe()

"""Kita juga melihat apakah ada null value di dataset"""

df.isnull().sum()

"""dari hasil tidak ditemukan null value dari dataset

### **Univariate Analysis**

Untuk univariate analysys kita plot masing-masing feature di dataset. Pertama untuk data numerik digunakan histogram untuk melihat persebarannya
"""

# fitur age
df.hist(bins=50, figsize=(20,15))

"""Sedangkan untuk fitur kategorikal digunakan barchart untuk melihat persebaran di setiap kategorinya."""

# categorical features
categorical_features = ['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia',
                        'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing',
                        'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity', 'class']

# Loop melalui setiap fitur kategorikal
for feature in categorical_features:
    count = df[feature].value_counts()
    percent = 100 * df[feature].value_counts(normalize=True)
    data = pd.DataFrame({'jumlah sampel': count, 'persentase': percent.round(1)})

    print(f"Analisis univariate untuk fitur: {feature}")
    print(data)

    count.plot(kind='bar', title=feature)
    plt.xlabel(feature)
    plt.ylabel('Jumlah sampel')
    plt.show()

"""## **Data Preparation**

### **Encoding**

Untuk encoding dilakukan dengan cara mapping integer ke kategori. Untuk fitur yang kategorinya yes or no menggunakan looping untuk mempermudah proses encoding. Sedangkan untuk gender dan class encoding di cell terpisah karena memiliki kategori yang berbeda.
"""

# encoding fitur yang kategorinya yes and no
features_to_encode = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia',
                      'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing',
                      'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']

# Loop melalui setiap fitur dan lakukan encoding
for feature in features_to_encode:
    df[feature] = df[feature].map({'No': 0, 'Yes': 1})

# Menampilkan hasil encoding
print(df.head())

# encoding fitur gender
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# encoding fitur class
df['class'] = df['class'].map({'Negative': 0, 'Positive': 1})

df.head()

"""### **Hapus outlier**

Untuk menghapus outlier dibuat function untuk menghitung IQR. Dari hasil IQR didapatkan batas bawah dan atas. Dari situ baris yang umurnya di bawah batas bawah dan di atas batas atas akan dihapus
"""

def outlier_treatment(datacolumn):
  sorted(datacolumn)  # Memastikan data sudah urut
  Q1, Q3 = np.percentile(datacolumn , [25,75])
  IQR = Q3 - Q1
  lower_range = Q1 - (1.5 * IQR)
  upper_range = Q3 + (1.5 * IQR)
  return lower_range, upper_range

# Menghitung batas bawah dan atas untuk kolom Age
lowerbound, upperbound = outlier_treatment(df.Age)
print("Batas bawah: {0}\nBatas atas: {1}".format(lowerbound, upperbound))

# Menampilkan data yang menjadi outlier
df[(df.Age < lowerbound) | (df.Age > upperbound)]

# Menghapus outlier
data = df.drop(df[(df.Age < lowerbound) | (df.Age > upperbound)].index)
data

"""### **Standar Scaler**

Untuk fitur age, value diskala menjadi mean = 0 dan standar deviasi 1 dengan standard scaler.
"""

scaler = StandardScaler()
data['Age'] = scaler.fit_transform(data[['Age']])
data

"""### **Train-Test Split**

split data menjadi 80% data training dan 20% data test
"""

X = data.drop(["class"],axis =1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""### **Model Building**

Agar evaluasi lebih mudah dibuat function untuk looping ke model yang sudah dibuat dan mengevaluasinya dengan data test yang displit sebelumnya. Untuk evaluasi digunakan metriks akurasi, presisi, recall, dan F1-score
"""

# function untuk evaluasi model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

"""### **Logistic Regression**

Model dengan metode logistic regeression menggunakan function dari library sklearn dan untuk evaluasi model menggunakan function yang dibuat sebelumnya
"""

logistic_regression_model = LogisticRegression(random_state=55)
logistic_metrics = evaluate_model(logistic_regression_model,
                                  X_train, X_test, y_train, y_test)

"""### **Random Forest**

Model dengan metode random forest menggunakan function dari library sklearn dan untuk evaluasi model menggunakan function yang dibuat sebelumnya
"""

random_forest_model = RandomForestClassifier(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
random_forest_metrics = evaluate_model(random_forest_model,
                                       X_train, X_test, y_train, y_test)

"""### **evaluasi model**

Hasil dari function evaluasi model ditampilkan dalam bentuk dataframe agar mudah dibaca.
"""

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [logistic_metrics[0], random_forest_metrics[0]],
    'Precision': [logistic_metrics[1], random_forest_metrics[1]],
    'Recall': [logistic_metrics[2], random_forest_metrics[2]],
    'F1 Score': [logistic_metrics[3], random_forest_metrics[3]]
})

print(results)

"""Dari hasil di atas didapatkan bahwa model dengan metode **random forest** mendapatkan hasil lebih baik yaitu dengan akurasi 99%.

**prediksi dengan model**

Untuk memastikan akurasi dilakukan percobaan dengan menggunakan data test yang sudah ada lalu di test menggunakan model dan melihat akurasinya.
"""

# Membuat dictionary untuk menyimpan model
model_dict = {
    'logistic_regression': logistic_regression_model,
    'random_forest': random_forest_model
}

# Memilih satu sampel dari X_test untuk prediksi
prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true': y_test.iloc[:1].values[0]}

# Melakukan prediksi dengan setiap model dan menyimpan hasilnya
for name, model in model_dict.items():
    pred_dict['prediksi_' + name] = model.predict(prediksi).round(1)[0]

# Menampilkan hasil prediksi dalam bentuk DataFrame
prediksi_df = pd.DataFrame([pred_dict])
print(prediksi_df)