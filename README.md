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
🧠 2) Model Eğitimi + Değerlendirme

02_train_eval.py:

output_dataset/ içinden veriyi okur,

MobileNetV2 tabanlı modeli eğitir,

en iyi modeli outputs/recycle_best.keras olarak kaydeder,

sınıf isimlerini outputs/class_names.json içine yazar,

accuracy/loss grafikleri, confusion matrix, ROC eğrileri ve classification report üretir.

Çalıştır:

python 02_train_eval.py


Üretilen dosyalar:

outputs/recycle_best.keras (en iyi model)

outputs/class_names.json (Streamlit app için gerekli)

outputs/*.png (grafikler)

outputs/classification_report.txt

🖥️ 3) Streamlit Uygulaması

03_app_streamlit.py:

outputs/recycle_best.keras ve outputs/class_names.json dosyalarını kullanır,

görsel yükleyince sınıf tahmini + confidence gösterir,

tüm sınıf olasılıklarını listeler,

opsiyonel olarak “gerçek sınıf” seçtirip doğru/yanlış kontrol eder.

Çalıştır:

streamlit run 03_app_streamlit.py


Tarayıcıda açılan ekranda görsel yükleyip test edebilirsin.

🔧 Ayarlar / Özelleştirme

Görüntü boyutu: IMG_SIZE = (224, 224)

Epoch sayısı: EPOCHS = 25

Batch size: BATCH_SIZE = 32

Split oranları: TEST_SIZE = 0.15, VAL_SIZE = 0.15

İstersen sınıf sayısı arttırılabilir: ham dataset’e yeni klasör eklemen yeterli (eğitim kodu sınıfları otomatik okur).

⚠️ Dikkat Edilecekler

01_split_dataset_opencv.py içindeki RAW_DIR Windows path ile yazılmış olabilir; kendi bilgisayarına göre güncelle.

Streamlit uygulamasını çalıştırmadan önce eğitim çalıştırılıp outputs/ içine model ve json üretildiğinden emin ol.

📌 Kullanılan Yöntem

Transfer learning: MobileNetV2 (ImageNet weights)

Veri artırma (augmentation): flip/rotation/zoom/contrast

Kayıp: sparse_categorical_crossentropy

Metrik: accuracy

Callback’ler: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
