## Import yang Dibutuhkan

# •	import pandas as pd – library utama untuk membaca CSV dan mengolah data tabel (DataFrame). Dipakai di semua bagian.
•	import numpy as np – library operasi numerik dan array. Dimuat di awal notebook, meskipun pada bagian 1 sampai 5 tidak dipanggil langsung.
•	import matplotlib.pyplot as plt – membuat grafik: box plot, histogram, heatmap, dan diagram batang.
•	import seaborn as sns – grafik statistik yang lebih ringkas di atas matplotlib: boxplot, histplot, heatmap, dan barplot.
•	from sklearn.preprocessing import StandardScaler, LabelEncoder – StandardScaler untuk standardisasi (bagian 2.4), dan LabelEncoder untuk mengubah kolom teks menjadi angka (bagian 2.6 dan 4).
•	from sklearn.model_selection import train_test_split – membagi data menjadi data latih dan data uji (bagian 4 dan 5).
•	from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score – tiga ukuran penilai model: MAE, MSE, dan R² (bagian 5).
•	from sklearn.ensemble import RandomForestRegressor – model Random Forest untuk regresi (bagian 5.2).
•	from xgboost import XGBRegressor – model XGBoost untuk regresi (bagian 5.3). Ini library terpisah yang harus dipasang lebih dulu.
•	from lightgbm import LGBMRegressor – model LightGBM untuk regresi (bagian 5.4). Library ini juga harus dipasang terpisah.
•	import joblib – menyimpan model terlatih ke file .joblib (bagian 5.7).
•	import json – menulis metadata model ke file JSON (bagian 5.7).
•	import subprocess, sys – dipakai pada blok try / except ImportError di bagian 5.3 dan 5.4 untuk memasang xgboost dan lightgbm secara otomatis lewat pip. Jika kedua library sudah terpasang, baris ini boleh dihilangkan.
