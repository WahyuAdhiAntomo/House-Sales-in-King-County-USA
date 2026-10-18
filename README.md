# Import yang Dibutuhkan
 -	import pandas as pd – library utama untuk membaca CSV dan mengolah data tabel (DataFrame). Dipakai di semua bagian.
 •	import numpy as np – library operasi numerik dan array. Dimuat di awal notebook, meskipun pada bagian 1 sampai 5 tidak dipanggil langsung.
 •	import matplotlib.pyplot as plt – membuat grafik: box plot, histogram, heatmap, dan diagram batang.
 •	import seaborn as sns – grafik statistik yang lebih ringkas di atas matplotlib: boxplot, histplot, heatmap, dan barplot.
 •	from sklearn.preprocessing import StandardScaler, LabelEncoder – StandardScaler untuk standardisasi (bagian 2.4), dan LabelEncoder untuk mengubah kolom teks menjadi angka (bagian 2.6 dan 4).
 •	from sklearn.model_selection import train_test_split – membagi data menjadi data latih dan data uji (bagian 4 dan 5).
 •	from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score – tiga ukuran penilai model: MAE, MSE, dan R² (bagian 5).
 •	from sklearn.ensemble import RandomForestRegressor – model Random Forest untuk regresi (bagian 5.2).
 •	from lightgbm import LGBMRegressor – model LightGBM untuk regresi (bagian 5.4). Library ini juga harus dipasang terpisah.
 •	import joblib – menyimpan model terlatih ke file .joblib (bagian 5.7).
 •	import json – menulis metadata model ke file JSON (bagian 5.7).
 •	import subprocess, sys – dipakai pada blok try / except ImportError di bagian 5.3 dan 5.4 untuk memasang xgboost dan lightgbm secara otomatis lewat pip. Jika kedua library sudah terpasang, baris ini boleh dihilangkan.

# 1. Data Loading
•	import pandas as pd – memuat library pandas untuk mengolah data tabel (DataFrame). Alias pd dipakai agar penulisan lebih singkat.
•	import numpy as np – memuat NumPy untuk operasi numerik dan array.
•	import matplotlib.pyplot as plt – memuat matplotlib untuk membuat grafik.
•	import seaborn as sns – memuat seaborn, library visualisasi statistik yang berjalan di atas matplotlib.
•	df = pd.read_csv('kc_house_data.csv') – membaca file CSV dan menyimpannya sebagai DataFrame bernama df.
•	print("Data awal:", df.shape) – mencetak ukuran data dalam bentuk (jumlah baris, jumlah kolom), yaitu (21613, 21).
•	print(df.info()) – mencetak ringkasan struktur: nama kolom, jumlah nilai non-null, dan tipe data tiap kolom. Karena info() mengembalikan None, tulisan None ikut tercetak.
•	print(df.head()) – mencetak 5 baris pertama.
•	print(df.describe()) – mencetak statistik deskriptif kolom numerik: jumlah, rata-rata, standar deviasi, minimum, kuartil, dan maksimum.
•	df.head() – baris terakhir sel, sehingga 5 baris pertama ditampilkan lagi dalam bentuk tabel notebook.

# 2. Data Cleaning dan Transformation
## 2.1 Persiapan
•	Empat baris import pertama sama seperti pada bagian 1.
•	df = pd.read_csv('kc_house_data.csv') – membaca ulang dataset asli agar sel ini berdiri sendiri.
•	print("Data awal:", df.shape) – mencetak ukuran data awal.
•	df.describe(include="all") – menghitung statistik untuk semua kolom, termasuk yang berupa teks. Hasilnya tidak tampil karena bukan baris terakhir sel.

## 2.2 Missing value
•	missing_values = df.isnull().sum() – menghitung jumlah nilai kosong di setiap kolom.
•	print(missing_values[missing_values > 0]) – mencetak hanya kolom yang punya nilai kosong. Hasilnya Series([], dtype: int64) karena dataset ini tidak punya nilai kosong.
•	less = missing_values[missing_values < 1000].index – mengambil nama kolom yang nilai kosongnya kurang dari 1000. Kolom ini akan diisi (diimputasi). Di sini semua 21 kolom masuk.
•	over = missing_values[missing_values >= 1000].index – mengambil nama kolom yang nilai kosongnya 1000 atau lebih. Kolom ini akan dibuang. Di sini hasilnya kosong.
•	numeric_features = df[less].select_dtypes(include=['number']).columns – memilih kolom bertipe angka dari kelompok less. Urutannya: id, price, bedrooms, bathrooms, dan seterusnya.
•	df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median()) – mengisi nilai kosong kolom numerik dengan median kolomnya masing-masing. Median dipilih karena tidak sensitif terhadap outlier.
•	kategorical_features = df[less].select_dtypes(include=['object']).columns – memilih kolom bertipe teks, yaitu date.
•	for column in kategorical_features: – mengulang untuk setiap kolom teks.
•	df[column] = df[column].fillna(df[column].mode()[0]) – mengisi nilai kosong dengan modus (nilai yang paling sering muncul). [0] dipakai karena mode() bisa mengembalikan beberapa nilai, dan diambil yang pertama.
•	df = df.drop(columns=over) – membuang kolom yang terlalu banyak nilai kosongnya.

## 2.3 Outlier
•	for feature in numeric_features: – mengulang untuk setiap kolom numerik.
•	plt.figure(figsize=(10, 6)) – membuat kanvas grafik baru berukuran 10 × 6 inci.
•	sns.boxplot(x=df[feature]) – menggambar box plot untuk melihat sebaran dan titik outlier.
•	plt.title(f'Box Plot of {feature}') – memberi judul grafik sesuai nama kolom.
•	plt.show() – menampilkan grafik.
•	Q1 = df[numeric_features].quantile(0.25) – kuartil pertama (persentil ke-25) tiap kolom numerik.
•	Q3 = df[numeric_features].quantile(0.75) – kuartil ketiga (persentil ke-75).
•	IQR = Q3 - Q1 – rentang antar kuartil, ukuran sebaran bagian tengah data.
•	condition = ~((df[numeric_features] < (Q1 - 1.5 * IQR)) | (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1) – membuat penanda per baris. Bagian dalam menandai nilai yang di bawah batas bawah atau di atas batas atas (aturan 1,5 × IQR). .any(axis=1) bernilai True jika baris itu punya minimal satu nilai di luar batas, dan ~ membalik hasilnya. Jadi condition True untuk baris yang semua nilai numeriknya wajar.
•	df_filtered_numeric = df.loc[condition, numeric_features] – menyimpan kolom numerik hanya dari baris yang lolos.
•	categorical_features = df.select_dtypes(include=['object']).columns – mengambil nama kolom teks.
•	df = pd.concat([df_filtered_numeric, df.loc[condition, categorical_features]], axis=1) – menggabungkan kolom numerik dan kolom teks dari baris yang lolos. Hasilnya 14.934 baris.

## 2.4 Standardisasi
•	plt.figure(figsize=(12, 5)) – kanvas baru berukuran 12 × 5 inci.
•	plt.subplot(1, 2, 1) – membagi kanvas menjadi 1 baris × 2 kolom dan memakai kotak pertama.
•	sns.histplot(df[numeric_features[3]], kde=True) – histogram kolom numerik ke-4 (indeks 3), yaitu bathrooms, ditambah garis kepadatan (kde).
•	plt.title("Histogram Sebelum Standardisasi") dan plt.show() – memberi judul dan menampilkan grafik.
•	from sklearn.preprocessing import StandardScaler – memuat kelas standardisasi dari scikit-learn.
•	scaler = StandardScaler() – membuat objek scaler.
•	df[numeric_features] = scaler.fit_transform(df[numeric_features]) – menghitung rata-rata dan standar deviasi, lalu mengubah tiap nilai menjadi (x − rata-rata) / standar deviasi. Hasilnya berpusat di 0 dengan standar deviasi 1.
•	Tiga baris berikutnya (plt.figure, plt.subplot, sns.histplot, plt.title, plt.show) menggambar histogram yang sama setelah standardisasi, dengan judul "Histogram Setelah Standardisasi". Bentuk sebarannya sama, hanya skalanya berubah.

## 2.5 Duplikasi
•	duplicates = df.duplicated() – menandai baris yang sama persis dengan baris sebelumnya (True/False).
•	print("Baris duplikat:", duplicates.sum()) – mencetak jumlah baris duplikat, yaitu 0.
•	print(df[duplicates]) – mencetak baris duplikat. Hasilnya tabel kosong.
•	df = df.drop_duplicates() – membuang baris duplikat (tidak ada yang dibuang).
•	print("Setelah hapus duplikat:", df.shape) – mencetak ukuran data, yaitu (14934, 21). Angka ini turun dari 21.613 karena penghapusan outlier di langkah 2.3, bukan karena duplikat.

## 2.6 Encoding
•	category_features = df.select_dtypes(include=['object']).columns – mengambil nama kolom teks (date).
•	df[category_features] – menyebut kolom teks. Tidak menampilkan apa pun karena bukan baris terakhir sel.
•	df_one_hot = pd.get_dummies(df, columns=category_features) – One Hot Encoding: tiap nilai unik date dijadikan kolom biner (0/1) sendiri.
•	df_one_hot – hanya menyebut variabel, tidak tampil.
•	from sklearn.preprocessing import LabelEncoder – memuat kelas Label Encoding.
•	label_encoder = LabelEncoder() – membuat objek encoder yang mengubah teks menjadi angka bulat.
•	df_lencoder = pd.DataFrame(df) – membuat DataFrame baru dari df untuk menampung hasil encoding.
•	for col in category_features: – mengulang untuk tiap kolom teks.
•	df_lencoder[col] = label_encoder.fit_transform(df[col]) – mengganti isi kolom teks dengan kode angka (0, 1, 2, …) sesuai urutan nilai uniknya.
•	df_lencoder – baris terakhir sel, jadi tabel hasil ditampilkan. Nilai-nilai numeriknya sudah terstandardisasi.

# 3. EDA dan ExDA
## 3.1 Persiapan dan missing value
•	Empat import dan df = pd.read_csv('kc_house_data.csv') – sama seperti sebelumnya. df di sini adalah data asli, sedangkan df_lencoder adalah variabel dari sel cleaning yang masih tersimpan di memori notebook.
•	df_lencoder.head() – memanggil 5 baris pertama data hasil cleaning. Tidak tampil karena bukan baris terakhir.
•	missing_values = df_lencoder.isnull().sum() – jumlah nilai kosong per kolom pada data hasil cleaning.
•	missing_percentage = (missing_values / len(df_lencoder)) * 100 – mengubahnya menjadi persentase dari total baris.
•	missing_data = pd.DataFrame({...}).sort_values(by='Missing Values', ascending=False) – menyusun tabel berisi jumlah dan persentase nilai kosong, diurutkan dari yang terbanyak.
•	missing_data[missing_data['Missing Values'] > 0] – menyaring kolom yang punya nilai kosong. Tidak tampil karena bukan baris terakhir.
•	Empat baris berikutnya (missing_values_ori, missing_percentage_ori, missing_data_ori, dan penyaringannya) melakukan hal yang sama untuk data asli df. Karena baris penyaringan ini terakhir di blok tersebut, hasilnya (tabel kosong) ditampilkan jika tidak diikuti kode lain.

## 3.2 Histogram seluruh fitur numerik
•	num_vars = df.select_dtypes(include=['number']).shape[1] – menghitung jumlah kolom numerik.
•	n_cols = 4 – grafik disusun 4 kolom per baris.
•	n_rows = -(-num_vars // n_cols) – menghitung jumlah baris grafik dengan pembulatan ke atas. Tanda minus ganda dipakai karena // membulatkan ke bawah.
•	fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4)) – membuat kisi grafik, tinggi menyesuaikan jumlah baris.
•	axes = axes.flatten() – meratakan kisi 2 dimensi menjadi daftar 1 dimensi agar mudah diulang.
•	for i, column in enumerate(df.select_dtypes(include=['number']).columns): – mengulang tiap kolom numerik dengan nomor urut i.
•	df[column].hist(ax=axes[i], bins=20, edgecolor='black') – menggambar histogram 20 batang di kotak ke-i, dengan garis tepi hitam.
•	axes[i].set_title(column), axes[i].set_xlabel('Value'), axes[i].set_ylabel('Frequency') – memberi judul dan label sumbu.
•	for j in range(i + 1, len(axes)): dan fig.delaxes(axes[j]) – menghapus kotak grafik kosong yang tersisa.
•	plt.tight_layout() – merapikan jarak antar grafik.
•	plt.show() – menampilkan hasil.

## 3.3 Distribusi fitur utama
•	columns_to_plot = ['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living'] – daftar kolom yang akan digambar.
•	plt.figure(figsize=(15, 10)) – kanvas 15 × 10 inci.
•	for i, column in enumerate(columns_to_plot, 1): – mengulang dengan nomor mulai dari 1.
•	plt.subplot(2, 3, i) – memakai kotak ke-i pada kisi 2 × 3.
•	sns.histplot(df[column], kde=True, bins=30) – histogram 30 batang dengan garis kepadatan.
•	plt.title(f'Distribution of {column}') – judul tiap grafik.
•	plt.tight_layout() dan plt.show() – merapikan dan menampilkan.

## 3.4 Korelasi
•	plt.figure(figsize=(12, 10)) – kanvas baru.
•	correlation_matrix = df.select_dtypes(include=['number']).corr() – menghitung korelasi antar semua pasangan kolom numerik (nilai −1 sampai 1).
•	sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1) – menggambar peta panas korelasi tanpa angka (annot=False), skala warna biru–merah dari −1 sampai 1.
•	plt.title('Correlation Matrix') dan plt.show() – judul dan tampilkan.
•	target_corr = df.select_dtypes(include=['number']).corr()['price'] – mengambil korelasi semua kolom terhadap price.
•	target_corr_sorted = target_corr.abs().sort_values(ascending=False) – mengambil nilai mutlaknya (kekuatan hubungan tanpa melihat arah) lalu mengurutkan dari yang terbesar.
•	plt.figure(figsize=(10, 6)) – kanvas baru.
•	target_corr_sorted.plot(kind='bar') – diagram batang kekuatan korelasi tiap variabel.
•	plt.title(...), plt.xlabel('Variables'), plt.ylabel('Correlation Coefficient'), plt.show() – judul, label sumbu, dan tampilkan.

# 4. Data Splitting
•	import pandas as pd – memuat pandas.
•	from sklearn.model_selection import train_test_split – memuat fungsi pembagi data latih dan uji.
•	from sklearn.preprocessing import LabelEncoder – memuat Label Encoder.
•	df = pd.read_csv('kc_house_data.csv') dan print("Data awal:", df.shape) – membaca ulang data asli dan mencetak ukurannya.
•	category_features = df.select_dtypes(include=['object']).columns – kolom teks (date).
•	label_encoder = LabelEncoder() dan df_lencoder = pd.DataFrame(df) – menyiapkan encoder dan DataFrame penampung.
•	for col in category_features: dan df_lencoder[col] = label_encoder.fit_transform(df[col]) – mengubah kolom teks menjadi angka.
•	df_lencoder["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S") – menimpa kolom date dengan tanggal sungguhan. Formatnya tahun-bulan-hari, huruf T, lalu jam-menit-detik (contoh: 20141013T000000).
•	df_lencoder["sale_year"] = df_lencoder["date"].dt.year – membuat fitur tahun penjualan.
•	df_lencoder["sale_month"] = df_lencoder["date"].dt.month – membuat fitur bulan penjualan.
•	X = df_lencoder.drop(columns=["price", "id", "date"]) – fitur (input model): semua kolom kecuali target price, id (hanya penanda), dan date (sudah diwakili sale_year dan sale_month). Tersisa 20 fitur.
•	y = df_lencoder["price"] – target yang akan diprediksi.
•	print("Jumlah fitur :", X.shape[1]) – mencetak jumlah fitur (20).
•	print("Jumlah target:", y.shape[0]) – mencetak jumlah data target (21613).
•	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) – membagi data secara acak: 20% untuk uji dan 80% untuk latih. random_state=42 mengunci keacakan agar hasil bisa diulang.
•	print("Jumlah data latih:", len(x_train)) – 17.290 baris.
•	print("Jumlah data test :", len(x_test)) – 4.323 baris.

# 5. Modelling
## 5.1 Persiapan data
•	Empat import (pandas, numpy, matplotlib, seaborn), train_test_split, dan LabelEncoder – sama seperti bagian sebelumnya.
•	from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score – memuat tiga fungsi penilai model: MAE (rata-rata selisih absolut), MSE (rata-rata kuadrat selisih), dan R² (seberapa besar variasi harga yang bisa dijelaskan model).
•	Blok dari df = pd.read_csv(...) sampai x_train, x_test, y_train, y_test = train_test_split(...) – identik dengan bagian 4: membaca data, meng-encode kolom date, membuat sale_year dan sale_month, memisahkan X dan y, lalu membagi 80:20 dengan random_state=42.
•	print("Train:", x_train.shape, " Test:", x_test.shape) – mencetak ukuran data latih (17290, 20) dan uji (4323, 20).
•	results = {} – kamus kosong untuk menampung skor tiap model.

## 5.2 Random Forest
•	from sklearn.ensemble import RandomForestRegressor – memuat model Random Forest untuk regresi, yaitu gabungan banyak pohon keputusan yang hasilnya dirata-rata.
•	rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1) – membuat model dengan 200 pohon. random_state=42 mengunci keacakan, dan n_jobs=-1 memakai seluruh inti CPU.
•	rf_model.fit(x_train, y_train) – melatih model memakai data latih.
•	pred_rf = rf_model.predict(x_test) – memprediksi harga untuk data uji.
•	results['Random Forest'] = [mean_absolute_error(y_test, pred_rf), mean_squared_error(y_test, pred_rf), r2_score(y_test, pred_rf)] – menghitung MAE, MSE, dan R² dengan membandingkan harga aktual (y_test) dan prediksi, lalu menyimpannya di results.
•	print("Random Forest selesai") – penanda proses selesai.

## 5.3 XGBoost
•	try: / from xgboost import XGBRegressor – mencoba memuat model XGBoost.
•	except ImportError: – dijalankan jika library belum terpasang.
•	import subprocess, sys dan subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"]) – memasang xgboost lewat pip memakai Python yang sedang dipakai notebook.
•	from xgboost import XGBRegressor – memuat ulang setelah terpasang.
•	xgb_model = XGBRegressor(...) – membuat model dengan parameter: n_estimators=300 (300 pohon berurutan), learning_rate=0.1 (seberapa besar tiap pohon mengoreksi kesalahan sebelumnya), max_depth=6 (kedalaman maksimal pohon), subsample=0.8 (tiap pohon memakai 80% baris), colsample_bytree=0.8 (tiap pohon memakai 80% kolom), random_state=42, n_jobs=-1, dan verbosity=0 (tanpa pesan log).
•	xgb_model.fit(x_train, y_train), pred_xgb = xgb_model.predict(x_test), results['XGBoost'] = [...], dan print("XGBoost selesai") – melatih, memprediksi, menghitung skor, dan memberi penanda, sama seperti Random Forest.

## 5.4 LightGBM
•	Struktur try / except ImportError untuk from lightgbm import LGBMRegressor sama dengan XGBoost, termasuk pemasangan otomatis lewat pip.
•	lgbm_model = LGBMRegressor(...) – membuat model LightGBM. Parameternya mirip XGBoost, ditambah num_leaves=31 (jumlah daun maksimal per pohon) dan verbose=-1 (tanpa log).
•	lgbm_model.fit(...), pred_lgbm = ...predict(x_test), results['LightGBM'] = [...], dan print("LightGBM selesai") – melatih, memprediksi, dan menyimpan skor.

## 5.5 Perbandingan model
•	df_results = pd.DataFrame(results, index=['MAE', 'MSE', 'R2']).T – mengubah kamus menjadi tabel. Awalnya model menjadi kolom, lalu .T (transpose) menukar sehingga tiap model menjadi satu baris.
•	df_results = df_results.sort_values(by='R2', ascending=False) – mengurutkan dari R² tertinggi.
•	print(df_results) – mencetak tabel perbandingan. Urutannya XGBoost, LightGBM, Random Forest.
•	plt.figure(figsize=(8, 5)) – kanvas grafik R².
•	sns.barplot(x=df_results.index, y=df_results['R2'], palette='viridis') – diagram batang R² tiap model dengan palet warna viridis. Peringatan FutureWarning soal palette yang muncul di output hanya pemberitahuan versi seaborn, bukan kesalahan.
•	plt.title('Perbandingan R² Score antar Model') – judul.
•	plt.ylim(0, 1) – sumbu Y dibatasi 0 sampai 1.
•	for idx, val in enumerate(df_results['R2']): dan plt.text(idx, val + 0.01, f"{val:.4f}", ha='center') – menulis nilai R² (4 desimal) sedikit di atas tiap batang, rata tengah.
•	plt.show() – menampilkan grafik.
•	Blok grafik kedua memakai pola yang sama untuk MAE: palet magma, teks diletakkan di val + 100, dan angkanya diformat dengan pemisah ribuan lewat f"{val:,.0f}".

## 5.6 Feature importance model terbaik
•	best_model_name = df_results.index[0] – nama model di baris teratas (R² tertinggi), yaitu XGBoost.
•	print(f"Model terbaik: {best_model_name}") – mencetak nama tersebut.
•	best_model = {'Random Forest': rf_model, 'XGBoost': xgb_model, 'LightGBM': lgbm_model}[best_model_name] – kamus yang memetakan nama ke objek model, lalu mengambil objek sesuai nama terbaik.
•	fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': best_model.feature_importances_}) – tabel berisi nama fitur dan tingkat kepentingannya menurut model.
•	.sort_values(by='Importance', ascending=False).head(15) – mengurutkan dari yang terpenting dan mengambil 15 teratas.
•	plt.figure(figsize=(10, 7)) – kanvas baru.
•	sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis') – diagram batang horizontal.
•	plt.title(...), plt.tight_layout(), plt.show() – judul, merapikan tata letak, dan menampilkan.
5.7 Menyimpan model dan metadata
•	import joblib, json – joblib untuk menyimpan objek model ke file, json untuk membaca dan menulis file JSON.
•	model_filename = f'best_model_{best_model_name.replace(" ", "_").lower()}.joblib' – membuat nama file dari nama model: spasi diganti garis bawah dan huruf dikecilkan, sehingga menjadi best_model_xgboost.joblib.
•	joblib.dump(best_model, model_filename) – menyimpan model terlatih ke file.
•	metadata = {...} – kamus berisi nama model, daftar fitur (list(X.columns)), jumlah fitur, dan tiga metrik. float(...) dipakai agar nilai NumPy bisa ditulis ke JSON.
•	with open('model_metadata.json', 'w') as f: dan json.dump(metadata, f, indent=2) – membuka file untuk ditulis dan menyimpan metadata dengan indentasi 2 spasi. with menutup file otomatis.
•	Dua print(...) terakhir – mengonfirmasi nama file yang disimpan.


# Ringkasan Alur
1.	Loading: data dibaca dari kc_house_data.csv.
2.	Cleaning: missing value, outlier, standardisasi, duplikat, dan encoding diuji coba pada salinan data (hanya untuk eksplorasi).
3.	EDA: distribusi dan korelasi fitur terhadap price.
4.	Splitting: 20 fitur, 80% data latih dan 20% data uji.
5.	Modelling: tiga model dibandingkan, XGBoost menang dengan R² 0,8672.
