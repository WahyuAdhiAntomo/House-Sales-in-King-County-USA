# House Sales in King County, USA

## Import yang Dibutuhkan

- `import pandas as pd` : Membaca CSV dan mengolah data tabel (DataFrame). Digunakan di semua bagian.
- `import numpy as np` : Mendukung operasi numerik dan array. Dimuat di awal notebook, meskipun pada bagian 1–5 tidak dipanggil langsung.
- `import matplotlib.pyplot as plt` : Membuat box plot, histogram, heatmap, dan diagram batang.
- `import seaborn as sns` : Membuat grafik statistik di atas matplotlib, seperti boxplot, histplot, heatmap, dan barplot.
- `from sklearn.preprocessing import StandardScaler, LabelEncoder` : Memuat StandardScaler untuk standardisasi (bagian 2.4) dan LabelEncoder untuk mengubah kolom teks menjadi angka (bagian 2.6 dan 4).
- `from sklearn.model_selection import train_test_split` : Membagi data menjadi data latih dan data uji (bagian 4 dan 5).
- `from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score` : Memuat tiga metrik evaluasi model, yaitu MAE, MSE, dan R² (bagian 5).
- `from sklearn.ensemble import RandomForestRegressor` : Memuat model Random Forest untuk regresi (bagian 5.2).
- `from lightgbm import LGBMRegressor` : Memuat model LightGBM untuk regresi (bagian 5.4). Library perlu dipasang terpisah.
- `import joblib` : Menyimpan model terlatih ke file `.joblib` (bagian 5.7).
- `import json` : Menulis metadata model ke file JSON (bagian 5.7).
- `import subprocess, sys` : Digunakan pada blok `try/except ImportError` untuk memasang XGBoost dan LightGBM melalui pip (bagian 5.3 dan 5.4). Jika library sudah terpasang, blok pemasangan otomatis tidak diperlukan.

## 1. Data Loading

- `import pandas as pd` : Memuat pandas untuk mengolah DataFrame. Alias `pd` digunakan agar penulisan lebih singkat.
- `import numpy as np` : Memuat NumPy untuk operasi numerik dan array.
- `import matplotlib.pyplot as plt` : Memuat matplotlib untuk membuat grafik.
- `import seaborn as sns` : Memuat seaborn untuk visualisasi statistik.
- `df = pd.read_csv('kc_house_data.csv')` : Membaca file CSV dan menyimpannya sebagai DataFrame bernama `df`.
- `print("Data awal:", df.shape)` : Mencetak ukuran data dalam bentuk `(jumlah baris, jumlah kolom)`, yaitu `(21613, 21)`.
- `print(df.info())` : Menampilkan struktur data, jumlah nilai non-null, dan tipe data setiap kolom. Karena `info()` mengembalikan `None`, tulisan `None` ikut tercetak.
- `print(df.head())` : Mencetak lima baris pertama.
- `print(df.describe())` : Mencetak statistik deskriptif kolom numerik, seperti jumlah, rata-rata, standar deviasi, minimum, kuartil, dan maksimum.
- `df.head()` : Jika menjadi ekspresi terakhir dalam sel notebook, menampilkan kembali lima baris pertama dalam bentuk tabel.

## 2. Data Cleaning dan Transformation

### 2.1 Persiapan

- Empat baris import pertama sama seperti pada bagian 1.
- `df = pd.read_csv('kc_house_data.csv')` : Membaca ulang dataset asli agar bagian ini dimulai dari data awal.
- `print("Data awal:", df.shape)` : Mencetak ukuran data awal.
- `df.describe(include="all")` : Menghitung statistik semua kolom, termasuk kolom teks. Hasil tidak ditampilkan otomatis jika bukan ekspresi terakhir sel.

### 2.2 Missing Value

- `missing_values = df.isnull().sum()` : Menghitung jumlah nilai kosong pada setiap kolom.
- `print(missing_values[missing_values > 0])` : Mencetak kolom yang memiliki nilai kosong. Hasil pada notebook adalah `Series([], dtype: int64)` karena tidak ditemukan nilai kosong.
- `less = missing_values[missing_values < 1000].index` : Memilih kolom dengan jumlah nilai kosong kurang dari 1.000 untuk diimputasi. Pada data ini, semua 21 kolom masuk kelompok tersebut.
- `over = missing_values[missing_values >= 1000].index` : Memilih kolom dengan jumlah nilai kosong minimal 1.000 untuk dibuang. Pada data ini, hasilnya kosong.
- `numeric_features = df[less].select_dtypes(include=['number']).columns` : Memilih kolom numerik dari kelompok `less`.
- `df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())` : Mengisi nilai kosong kolom numerik dengan median masing-masing kolom.
- `kategorical_features = df[less].select_dtypes(include=['object']).columns` : Memilih kolom teks, yaitu `date`.
- `for column in kategorical_features:` : Melakukan perulangan untuk setiap kolom teks.
- `df[column] = df[column].fillna(df[column].mode()[0])` : Mengisi nilai kosong dengan modus. Indeks `[0]` mengambil nilai pertama jika terdapat beberapa modus.
- `df = df.drop(columns=over)` : Menghapus kolom yang jumlah nilai kosongnya mencapai batas yang ditentukan.

### 2.3 Outlier

#### Visualisasi Outlier

- `for feature in numeric_features:` : Melakukan perulangan untuk setiap kolom numerik.
- `plt.figure(figsize=(10, 6))` : Membuat kanvas grafik berukuran 10 × 6 inci.
- `sns.boxplot(x=df[feature])` : Membuat box plot untuk melihat sebaran data dan titik outlier.
- `plt.title(f'Box Plot of {feature}')` : Memberikan judul sesuai nama kolom.
- `plt.show()` : Menampilkan grafik.

#### Penyaringan dengan IQR

- `Q1 = df[numeric_features].quantile(0.25)` : Menghitung kuartil pertama setiap kolom numerik.
- `Q3 = df[numeric_features].quantile(0.75)` : Menghitung kuartil ketiga.
- `IQR = Q3 - Q1` : Menghitung rentang antarkuartil.

Kondisi penyaringan:

```python
condition = ~(
    (df[numeric_features] < (Q1 - 1.5 * IQR))
    | (df[numeric_features] > (Q3 + 1.5 * IQR))
).any(axis=1)
```

- Perbandingan terhadap batas bawah dan batas atas menandai nilai di luar rentang IQR.
- `.any(axis=1)` : Menghasilkan `True` jika sebuah baris memiliki minimal satu nilai di luar batas.
- `~` : Membalik hasil tersebut, sehingga `condition` bernilai `True` untuk baris yang lolos penyaringan.
- `df_filtered_numeric = df.loc[condition, numeric_features]` : Mengambil kolom numerik dari baris yang lolos.
- `categorical_features = df.select_dtypes(include=['object']).columns` : Mengambil nama kolom teks.
- `df = pd.concat([df_filtered_numeric, df.loc[condition, categorical_features]], axis=1)` : Menggabungkan kembali kolom numerik dan teks dari baris yang lolos.
- Jumlah data setelah penyaringan adalah 14.934 baris.

### 2.4 Standardisasi

#### Sebelum Standardisasi

- `plt.figure(figsize=(12, 5))` : Membuat kanvas berukuran 12 × 5 inci.
- `plt.subplot(1, 2, 1)` : Membagi kanvas menjadi satu baris dan dua kolom, lalu memilih posisi pertama.
- `sns.histplot(df[numeric_features[3]], kde=True)` : Membuat histogram kolom numerik keempat, yaitu `bathrooms`, beserta garis kepadatan.
- `plt.title("Histogram Sebelum Standardisasi")` : Memberikan judul.
- `plt.show()` : Menampilkan grafik.

#### Proses Standardisasi

- `from sklearn.preprocessing import StandardScaler` : Memuat kelas StandardScaler.
- `scaler = StandardScaler()` : Membuat objek scaler.
- `df[numeric_features] = scaler.fit_transform(df[numeric_features])` : Menghitung parameter standardisasi dan mentransformasikan kolom numerik.

Transformasi menggunakan rumus:

```text
z = (x - rata-rata) / standar deviasi
```

#### Setelah Standardisasi

- Rangkaian `plt.figure()`, `plt.subplot()`, `sns.histplot()`, `plt.title()`, dan `plt.show()` digunakan kembali untuk menampilkan histogram setelah standardisasi.
- Judul grafik menjadi `"Histogram Setelah Standardisasi"`.
- Bentuk sebaran tetap sama, sedangkan skala nilainya berubah.

### 2.5 Duplikasi

- `duplicates = df.duplicated()` : Menandai baris yang sama persis dengan baris yang telah muncul sebelumnya.
- `print("Baris duplikat:", duplicates.sum())` : Mencetak jumlah baris duplikat, yaitu `0`.
- `print(df[duplicates])` : Mencetak baris duplikat. Hasilnya tabel kosong.
- `df = df.drop_duplicates()` : Menghapus baris duplikat. Pada data ini, tidak ada baris yang dibuang.
- `print("Setelah hapus duplikat:", df.shape)` : Mencetak ukuran data, yaitu `(14934, 21)`.

Penurunan jumlah baris dari 21.613 menjadi 14.934 terjadi akibat penyaringan outlier pada bagian 2.3, bukan penghapusan duplikat.

### 2.6 Encoding

#### Persiapan

- `category_features = df.select_dtypes(include=['object']).columns` : Memilih kolom teks, yaitu `date`.
- `df[category_features]` : Mengakses kolom teks. Hasil tidak ditampilkan otomatis jika bukan ekspresi terakhir sel.

#### One-Hot Encoding

- `df_one_hot = pd.get_dummies(df, columns=category_features)` : Mengubah setiap nilai unik pada kolom teks menjadi kolom indikator.
- `df_one_hot` : Menampilkan hasil jika menjadi ekspresi terakhir sel.

#### Label Encoding

- `from sklearn.preprocessing import LabelEncoder` : Memuat kelas LabelEncoder.
- `label_encoder = LabelEncoder()` : Membuat objek encoder.
- `df_lencoder = pd.DataFrame(df)` : Membentuk DataFrame untuk menampung hasil encoding.
- `for col in category_features:` : Melakukan perulangan untuk setiap kolom teks.
- `df_lencoder[col] = label_encoder.fit_transform(df[col])` : Mengubah nilai teks menjadi kode angka, seperti `0`, `1`, `2`, dan seterusnya.
- `df_lencoder` : Menampilkan tabel hasil jika menjadi ekspresi terakhir sel. Kolom numeriknya telah melalui standardisasi pada langkah sebelumnya.

## 3. EDA dan ExDA

### 3.1 Persiapan dan Missing Value

- Empat import dan `df = pd.read_csv('kc_house_data.csv')` digunakan kembali.
- `df` berisi data asli, sedangkan `df_lencoder` berasal dari proses cleaning yang masih tersimpan di memori notebook.
- `df_lencoder.head()` : Mengakses lima baris pertama hasil cleaning.
- `missing_values = df_lencoder.isnull().sum()` : Menghitung nilai kosong per kolom.
- `missing_percentage = (missing_values / len(df_lencoder)) * 100` : Menghitung persentase nilai kosong.
- `missing_data = pd.DataFrame({...}).sort_values(by='Missing Values', ascending=False)` : Membuat tabel jumlah dan persentase nilai kosong, lalu mengurutkannya.
- `missing_data[missing_data['Missing Values'] > 0]` : Menyaring kolom yang memiliki nilai kosong.
- Variabel `missing_values_ori`, `missing_percentage_ori`, dan `missing_data_ori` digunakan untuk melakukan pemeriksaan yang sama pada data asli.

Ekspresi tabel hanya ditampilkan otomatis jika menjadi ekspresi terakhir sel notebook.

### 3.2 Histogram Seluruh Fitur Numerik

- `num_vars = df.select_dtypes(include=['number']).shape[1]` : Menghitung jumlah kolom numerik.
- `n_cols = 4` : Menentukan empat kolom grafik per baris.
- `n_rows = -(-num_vars // n_cols)` : Menghitung jumlah baris grafik dengan pembulatan ke atas.
- `fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))` : Membuat kisi grafik.
- `axes = axes.flatten()` : Meratakan susunan axes menjadi satu dimensi.
- `for i, column in enumerate(df.select_dtypes(include=['number']).columns):` : Mengulang setiap kolom numerik beserta indeksnya.
- `df[column].hist(ax=axes[i], bins=20, edgecolor='black')` : Membuat histogram dengan 20 bin dan garis tepi hitam.
- `axes[i].set_title(column)` : Memberikan judul.
- `axes[i].set_xlabel('Value')` : Memberikan label sumbu X.
- `axes[i].set_ylabel('Frequency')` : Memberikan label sumbu Y.
- `for j in range(i + 1, len(axes)):` dan `fig.delaxes(axes[j])` : Menghapus kotak grafik yang tidak terpakai.
- `plt.tight_layout()` : Merapikan jarak antargrafik.
- `plt.show()` : Menampilkan grafik.

### 3.3 Distribusi Fitur Utama

- `columns_to_plot = ['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living']` : Menentukan kolom yang akan divisualisasikan.
- `plt.figure(figsize=(15, 10))` : Membuat kanvas berukuran 15 × 10 inci.
- `for i, column in enumerate(columns_to_plot, 1):` : Mengulang kolom dengan indeks mulai dari satu.
- `plt.subplot(2, 3, i)` : Memilih posisi grafik pada kisi dua baris dan tiga kolom.
- `sns.histplot(df[column], kde=True, bins=30)` : Membuat histogram dengan 30 bin dan garis kepadatan.
- `plt.title(f'Distribution of {column}')` : Memberikan judul.
- `plt.tight_layout()` dan `plt.show()` : Merapikan serta menampilkan grafik.

### 3.4 Korelasi

#### Matriks Korelasi

- `plt.figure(figsize=(12, 10))` : Membuat kanvas baru.
- `correlation_matrix = df.select_dtypes(include=['number']).corr()` : Menghitung korelasi antarkolom numerik.
- `sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)` : Membuat heatmap tanpa anotasi angka dengan rentang warna dari −1 hingga 1.
- `plt.title('Correlation Matrix')` dan `plt.show()` : Memberikan judul dan menampilkan grafik.

#### Korelasi terhadap Harga

- `target_corr = df.select_dtypes(include=['number']).corr()['price']` : Mengambil korelasi setiap kolom numerik terhadap `price`.
- `target_corr_sorted = target_corr.abs().sort_values(ascending=False)` : Mengurutkan kekuatan korelasi berdasarkan nilai absolutnya.
- `plt.figure(figsize=(10, 6))` : Membuat kanvas baru.
- `target_corr_sorted.plot(kind='bar')` : Membuat diagram batang.
- `plt.title(...)`, `plt.xlabel('Variables')`, `plt.ylabel('Correlation Coefficient')`, dan `plt.show()` : Mengatur judul, label sumbu, dan menampilkan grafik.

## 4. Data Splitting

### Persiapan Data

- `import pandas as pd` : Memuat pandas.
- `from sklearn.model_selection import train_test_split` : Memuat fungsi pembagian data.
- `from sklearn.preprocessing import LabelEncoder` : Memuat LabelEncoder.
- `df = pd.read_csv('kc_house_data.csv')` : Membaca ulang data asli.
- `print("Data awal:", df.shape)` : Mencetak ukuran data.
- `category_features = df.select_dtypes(include=['object']).columns` : Memilih kolom teks.
- `label_encoder = LabelEncoder()` dan `df_lencoder = pd.DataFrame(df)` : Menyiapkan encoder dan DataFrame.
- Perulangan `for col in category_features:` beserta `fit_transform()` digunakan untuk mengubah kolom teks menjadi angka.

### Pengolahan Tanggal

- `df_lencoder["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")` : Mengisi kolom `date` dengan hasil konversi tanggal.
- Format tanggal mencakup tahun, bulan, hari, huruf `T`, jam, menit, dan detik. Contoh: `20141013T000000`.
- `df_lencoder["sale_year"] = df_lencoder["date"].dt.year` : Membuat fitur tahun penjualan.
- `df_lencoder["sale_month"] = df_lencoder["date"].dt.month` : Membuat fitur bulan penjualan.

### Pemisahan Fitur dan Target

- `X = df_lencoder.drop(columns=["price", "id", "date"])` : Menentukan fitur input dengan menghapus target, ID, dan kolom tanggal.
- `y = df_lencoder["price"]` : Menentukan harga sebagai target.
- `print("Jumlah fitur :", X.shape[1])` : Mencetak jumlah fitur, yaitu `20`.
- `print("Jumlah target:", y.shape[0])` : Mencetak jumlah data target, yaitu `21613`.

### Pembagian Data Latih dan Uji

```python
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

- `test_size=0.2` : Mengalokasikan 20% data untuk pengujian dan 80% untuk pelatihan.
- `random_state=42` : Menetapkan seed agar pembagian dapat diulang.
- `print("Jumlah data latih:", len(x_train))` : Menghasilkan 17.290 baris.
- `print("Jumlah data test :", len(x_test))` : Menghasilkan 4.323 baris.

## 5. Modelling

### 5.1 Persiapan Data

- Import pandas, NumPy, matplotlib, seaborn, `train_test_split`, dan `LabelEncoder` sama seperti bagian sebelumnya.
- `from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score` : Memuat metrik evaluasi:
  - MAE: rata-rata selisih absolut.
  - MSE: rata-rata kuadrat selisih.
  - R²: ukuran kemampuan model menjelaskan variasi target.
- Blok pembacaan data hingga pembagian data sama seperti bagian 4.
- `print("Train:", x_train.shape, " Test:", x_test.shape)` : Mencetak ukuran data latih `(17290, 20)` dan data uji `(4323, 20)`.
- `results = {}` : Membuat kamus kosong untuk menyimpan skor setiap model.

### 5.2 Random Forest

- `from sklearn.ensemble import RandomForestRegressor` : Memuat model Random Forest untuk regresi.
- `rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)` : Membuat model dengan parameter:
  - `n_estimators=200` : Menggunakan 200 pohon.
  - `random_state=42` : Menetapkan seed.
  - `n_jobs=-1` : Menggunakan seluruh inti CPU yang tersedia.
- `rf_model.fit(x_train, y_train)` : Melatih model.
- `pred_rf = rf_model.predict(x_test)` : Membuat prediksi untuk data uji.

Penyimpanan hasil evaluasi:

```python
results['Random Forest'] = [
    mean_absolute_error(y_test, pred_rf),
    mean_squared_error(y_test, pred_rf),
    r2_score(y_test, pred_rf)
]
```

- `print("Random Forest selesai")` : Menandai proses selesai.

### 5.3 XGBoost

#### Import dan Instalasi

- `try:` dan `from xgboost import XGBRegressor` : Mencoba memuat XGBoost.
- `except ImportError:` : Menangani kondisi ketika library belum tersedia.
- `import subprocess, sys` : Memuat modul untuk menjalankan instalasi.
- `subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])` : Memasang XGBoost melalui Python yang digunakan notebook.
- `from xgboost import XGBRegressor` : Memuat library setelah instalasi.

#### Parameter Model

`xgb_model = XGBRegressor(...)` menggunakan parameter berikut:

- `n_estimators=300` : Jumlah pohon atau tahap boosting.
- `learning_rate=0.1` : Besarnya kontribusi setiap tahap.
- `max_depth=6` : Kedalaman maksimal pohon.
- `subsample=0.8` : Proporsi baris yang digunakan pada setiap iterasi.
- `colsample_bytree=0.8` : Proporsi fitur yang digunakan untuk setiap pohon.
- `random_state=42` : Menetapkan seed.
- `n_jobs=-1` : Mengatur penggunaan inti CPU.
- `verbosity=0` : Menonaktifkan pesan log.

#### Pelatihan dan Evaluasi

- `xgb_model.fit(x_train, y_train)` : Melatih model.
- `pred_xgb = xgb_model.predict(x_test)` : Membuat prediksi.
- `results['XGBoost'] = [...]` : Menyimpan MAE, MSE, dan R².
- `print("XGBoost selesai")` : Menandai proses selesai.

### 5.4 LightGBM

- Struktur `try/except ImportError` digunakan untuk memuat `LGBMRegressor` dan memasang LightGBM jika belum tersedia.
- `lgbm_model = LGBMRegressor(...)` : Membuat model LightGBM.
- Parameter tambahan yang dijelaskan pada notebook:
  - `num_leaves=31` : Jumlah daun maksimal per pohon.
  - `verbose=-1` : Menonaktifkan pesan log.
- `lgbm_model.fit(...)` : Melatih model.
- `pred_lgbm = lgbm_model.predict(x_test)` : Membuat prediksi.
- `results['LightGBM'] = [...]` : Menyimpan metrik evaluasi.
- `print("LightGBM selesai")` : Menandai proses selesai.

### 5.5 Perbandingan Model

#### Tabel Evaluasi

- `df_results = pd.DataFrame(results, index=['MAE', 'MSE', 'R2']).T` : Membuat tabel evaluasi. Transpose `.T` menempatkan setiap model sebagai satu baris.
- `df_results = df_results.sort_values(by='R2', ascending=False)` : Mengurutkan model berdasarkan R² tertinggi.
- `print(df_results)` : Menampilkan tabel.

Urutan model pada hasil notebook:

1. XGBoost.
2. LightGBM.
3. Random Forest.

#### Grafik R²

- `plt.figure(figsize=(8, 5))` : Membuat kanvas.
- `sns.barplot(x=df_results.index, y=df_results['R2'], palette='viridis')` : Membuat diagram batang R².
- `plt.title('Perbandingan R² Score antar Model')` : Memberikan judul.
- `plt.ylim(0, 1)` : Membatasi sumbu Y dari 0 hingga 1.
- `for idx, val in enumerate(df_results['R2']):` : Mengulang setiap nilai untuk menambahkan label.
- `plt.text(idx, val + 0.01, f"{val:.4f}", ha='center')` : Menampilkan nilai dengan empat angka desimal di atas batang.
- `plt.show()` : Menampilkan grafik.

#### Grafik MAE

- Menggunakan pola yang sama dengan grafik R².
- Palet warna yang digunakan adalah `magma`.
- Posisi label berada pada `val + 100`.
- `f"{val:,.0f}"` : Menampilkan angka tanpa desimal dengan pemisah ribuan.

Catatan: `FutureWarning` terkait penggunaan `palette` merupakan peringatan kompatibilitas versi seaborn, bukan kegagalan proses pelatihan model.

### 5.6 Feature Importance Model Terbaik

- `best_model_name = df_results.index[0]` : Mengambil nama model dengan R² tertinggi, yaitu XGBoost pada hasil notebook.
- `print(f"Model terbaik: {best_model_name}")` : Mencetak nama model terbaik.

Pemilihan objek model:

```python
best_model = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'LightGBM': lgbm_model
}[best_model_name]
```

Penyusunan tabel feature importance:

```python
fi_df = (
    pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    })
    .sort_values(by='Importance', ascending=False)
    .head(15)
)
```

- `plt.figure(figsize=(10, 7))` : Membuat kanvas.
- `sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis')` : Membuat diagram batang horizontal untuk 15 fitur teratas.
- `plt.title(...)`, `plt.tight_layout()`, dan `plt.show()` : Memberikan judul, merapikan, dan menampilkan grafik.

### 5.7 Menyimpan Model dan Metadata

#### Menyimpan Model

- `import joblib, json` : Memuat library penyimpanan model dan pengolahan JSON.
- `model_filename = f'best_model_{best_model_name.replace(" ", "_").lower()}.joblib'` : Membuat nama file berdasarkan nama model.
- Untuk XGBoost, nama file menjadi `best_model_xgboost.joblib`.
- `joblib.dump(best_model, model_filename)` : Menyimpan model terlatih.

#### Menyimpan Metadata

- `metadata = {...}` : Membuat kamus metadata yang berisi:
  - Nama model.
  - Daftar fitur dari `list(X.columns)`.
  - Jumlah fitur.
  - Nilai MAE, MSE, dan R².
- `float(...)` : Mengonversi nilai metrik menjadi tipe angka yang dapat ditulis ke JSON.

```python
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

- `indent=2` : Mengatur indentasi JSON menjadi dua spasi.
- `with` : Menutup file secara otomatis setelah proses selesai.
- Dua perintah `print(...)` terakhir mengonfirmasi nama file yang disimpan.

## Ringkasan Alur

1. Loading: membaca data dari `kc_house_data.csv`.
2. Cleaning: mengeksplorasi missing value, outlier, standardisasi, duplikasi, dan encoding. Hasil cleaning ini tidak digunakan sebagai input pada tahap modelling yang membaca ulang data asli.
3. EDA: menganalisis distribusi fitur dan korelasi terhadap `price`.
4. Splitting: membagi data dengan 20 fitur menjadi 80% data latih dan 20% data uji.
5. Modelling: membandingkan Random Forest, XGBoost, dan LightGBM. Pada hasil notebook, XGBoost memperoleh R² tertinggi sebesar 0,8672.
