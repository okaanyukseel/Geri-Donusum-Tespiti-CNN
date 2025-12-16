♻️ Geri Dönüşüm Görüntü Sınıflandırma (Cam / Metal / Kağıt / Plastik)

Bu proje, geri dönüşüm atıklarını görüntü üzerinden Cam / Metal / Kağıt / Plastik olarak sınıflandırmak için:

veriyi train/val/test olarak ayırır,

MobileNetV2 (transfer learning) ile modeli eğitir ve raporlar üretir,

eğitilmiş modeli Streamlit arayüzünde görsel yükleyerek test etmeni sağlar.

📁 Proje Yapısı
.
├─ 01_split_dataset_opencv.py
├─ 02_train_eval.py
├─ 03_app_streamlit.py
├─ output_dataset/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ outputs/
   ├─ recycle_best.keras
   ├─ class_names.json
   ├─ epoch_accuracy.png
   ├─ epoch_loss.png
   ├─ confusion_matrix.png
   ├─ roc_auc.png
   └─ classification_report.txt

✅ Gereksinimler

Python 3.9+ önerilir

Temel kütüphaneler: tensorflow, opencv-python, numpy, matplotlib, scikit-learn, streamlit
Kurulum:

pip install tensorflow opencv-python numpy matplotlib scikit-learn streamlit


Not: TensorFlow kurulumu işletim sistemi / CUDA durumuna göre değişebilir.

🧩 1) Dataset’i Split Etme (train/val/test)

01_split_dataset_opencv.py dosyasında ham dataset klasörünü belirt:

RAW_DIR: Sınıf klasörlerini içeren ana klasör (ör: cam/metal/kagit/plastik)

OUT_DIR: Çıkış klasörü (varsayılan: output_dataset)

RESIZE_TO: (224, 224) (istersen kapatabilirsin)

Çalıştır:

python 01_split_dataset_opencv.py


Çıktı olarak şu yapı oluşur:

output_dataset/
  train/<class>/
  val/<class>/
  test/<class>/
