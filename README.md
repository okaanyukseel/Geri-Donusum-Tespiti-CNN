# ♻️ Geri Dönüşüm Görüntü Sınıflandırma (Cam / Metal / Kağıt / Plastik)

Bu proje, geri dönüşüm atıklarını görüntü üzerinden **Cam / Metal / Kağıt / Plastik** olarak sınıflandırmak için:

- veriyi **train/val/test** olarak ayırır,
- **MobileNetV2 (transfer learning)** ile modeli eğitir ve raporlar üretir,
- eğitilmiş modeli **Streamlit** arayüzünde görsel yükleyerek test etmeni sağlar.

Sınıf isimleri koda sabit yazılmamıştır; ham veri klasöründeki alt klasör isimlerinden otomatik okunur.

## 📁 Proje Yapısı

Depoda yer alan dosyalar:

```
.
├─ 01_split_dataset_opencv.py   # Ham veriyi train/val/test olarak ayırır
├─ 02_train_eval.py             # MobileNetV2 eğitimi + değerlendirme
├─ 03_app_streamlit.py          # Streamlit tahmin arayüzü
└─ README.md
```

Scriptler çalıştırıldığında oluşturulan klasörler (depoda **yer almaz**):

```
├─ output_dataset/
│  ├─ train/<sınıf>/
│  ├─ val/<sınıf>/
│  └─ test/<sınıf>/
└─ outputs/
   ├─ recycle_best.keras
   ├─ class_names.json
   ├─ epoch_accuracy.png
   ├─ epoch_loss.png
   ├─ confusion_matrix.png
   ├─ roc_auc.png
   └─ classification_report.txt
```

## 📦 Veri Seti

Veri seti depoda bulunmamaktadır. `01_split_dataset_opencv.py` içindeki `RAW_DIR` değişkeni şu yola ayarlıdır:

```
C:\DERSLER\goruntu isleme\trashnet\dataset-resized
```

Bu klasörün, her sınıf için bir alt klasör içermesi beklenir (ör. `cam/metal/kagit/plastik`). Desteklenen uzantılar: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`. En az 2 sınıf klasörü olmalıdır.

## ✅ Gereksinimler

- Python 3.9+ önerilir
- Kütüphaneler: `tensorflow`, `opencv-python`, `numpy`, `matplotlib`, `scikit-learn`, `streamlit`

Kurulum:

```bash
pip install -r requirements.txt
```

> Not: TensorFlow kurulumu işletim sistemi / CUDA durumuna göre değişebilir.

## 🧩 1) Dataset'i Split Etme (train/val/test)

`01_split_dataset_opencv.py` dosyasında ham dataset klasörünü belirt:

- `RAW_DIR`: Sınıf klasörlerini içeren ana klasör
- `OUT_DIR`: Çıkış klasörü (varsayılan: `output_dataset`)
- `RESIZE_TO`: `(224, 224)` (`None` yapılırsa resize kapanır)
- `SAVE_EXT`: `.jpg` (JPEG kalitesi 95)

Bölme işlemi `train_test_split` ile **stratified** yapılır (`SEED = 42`): önce %15 test ayrılır, ardından kalan kısımdan toplamın %15'i kadar validation ayrılır (yaklaşık %70 / %15 / %15). Görseller OpenCV ile okunur; okunamayan dosyalar atlanır ve sayısı yazdırılır.

Çalıştır:

```bash
python 01_split_dataset_opencv.py
```

## 🧠 2) Model Eğitimi + Değerlendirme

`02_train_eval.py`:

- `output_dataset/` içinden veriyi `image_dataset_from_directory` ile okur,
- MobileNetV2 tabanlı modeli eğitir,
- en iyi modeli `outputs/recycle_best.keras` olarak kaydeder,
- sınıf isimlerini `outputs/class_names.json` içine yazar,
- accuracy/loss grafikleri, confusion matrix, ROC eğrileri (one-vs-rest) ve classification report üretir.

Çalıştır:

```bash
python 02_train_eval.py
```

### Model Mimarisi

```
Input (224, 224, 3)
→ MobileNetV2 (ImageNet ağırlıkları, include_top=False, donuk / trainable=False)
→ GlobalAveragePooling2D
→ Dropout(0.3)
→ Dense(num_classes, softmax)
```

- Ön işleme: `mobilenet_v2.preprocess_input`
- Veri artırma (augmentation, sadece eğitimde): `RandomFlip("horizontal")`, `RandomRotation(0.05)`, `RandomZoom(0.1)`, `RandomContrast(0.1)`
- Optimizer: Adam (`learning_rate=1e-3`)
- Kayıp: `sparse_categorical_crossentropy`
- Metrik: `accuracy`
- Callback'ler:
  - `ModelCheckpoint` (`val_accuracy`, en iyi model)
  - `EarlyStopping` (`val_accuracy`, `patience=6`, `restore_best_weights=True`)
  - `ReduceLROnPlateau` (`val_loss`, `factor=0.5`, `patience=3`, `min_lr=1e-6`)

## 🖥️ 3) Streamlit Uygulaması

`03_app_streamlit.py`:

- `outputs/recycle_best.keras` ve `outputs/class_names.json` dosyalarını kullanır,
- görsel yükleyince (jpg/jpeg/png/webp) sınıf tahmini + confidence gösterir,
- en yüksek iki olasılık arasındaki farkı (margin) gösterir,
- tüm sınıf olasılıklarını listeler,
- opsiyonel olarak "gerçek sınıf" seçtirip doğru/yanlış kontrol eder.

Çalıştır:

```bash
streamlit run 03_app_streamlit.py
```

Tarayıcıda açılan ekranda görsel yükleyip test edebilirsin.

## 📊 Sonuçlar

Depoda eğitim çıktıları (`outputs/`) bulunmadığından sayısal sonuçlar burada verilmemiştir. Test accuracy/loss değerleri ve classification report, `02_train_eval.py` çalıştırıldığında konsola yazdırılır ve `outputs/classification_report.txt` dosyasına kaydedilir.

## 🔧 Ayarlar / Özelleştirme

- Görüntü boyutu: `IMG_SIZE = (224, 224)`
- Epoch sayısı: `EPOCHS = 25`
- Batch size: `BATCH_SIZE = 32`
- Split oranları: `TEST_SIZE = 0.15`, `VAL_SIZE = 0.15`

İstersen sınıf sayısı arttırılabilir: ham dataset'e yeni klasör eklemen yeterli (eğitim kodu sınıfları otomatik okur).

## ⚠️ Dikkat Edilecekler

- `01_split_dataset_opencv.py` içindeki `RAW_DIR` Windows path ile yazılmıştır; kendi bilgisayarına göre güncelle.
- Streamlit uygulamasını çalıştırmadan önce eğitim çalıştırılıp `outputs/` içine model ve json üretildiğinden emin ol.
