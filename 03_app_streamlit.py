import json
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Geri Dönüşüm Sınıflandırma", layout="centered")

st.title("♻️ Cam / Metal / Kağıt / Plastik Sınıflandırma")
st.write("Görsel yükle → model sınıfı ve confidence (eminlik) değerini göstersin.")

MODEL_PATH = "outputs/recycle_best.keras"
CLASSNAMES_PATH = "outputs/class_names.json"
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def load_class_names():
    with open(CLASSNAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

st.caption(f"Kullanılan sınıflar: {class_names}")

uploaded = st.file_uploader("📷 Görsel yükle (jpg/png/jpeg/webp)", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    # bytes -> OpenCV image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Görsel okunamadı. Farklı bir dosya dene.")
        st.stop()

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption="Yüklenen Görsel", use_container_width=True)

    # preprocess
    resized = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    prob = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(prob))
    pred_name = class_names[pred_idx]
    conf = float(prob[pred_idx])

    # belirsizlik için margin
    sorted_probs = np.sort(prob)
    top1 = float(sorted_probs[-1])
    top2 = float(sorted_probs[-2]) if len(sorted_probs) >= 2 else 0.0
    margin = top1 - top2

    st.subheader("✅ Tahmin Sonucu")
    st.write(f"**Tahmin edilen sınıf:** `{pred_name}`")
    st.write(f"**Confidence (eminlik):** `{conf:.4f}`")
    st.write(f"**Top1-Top2 farkı (margin):** `{margin:.4f}`  (küçükse daha kararsız)")

    st.subheader("📊 Tüm Sınıf Olasılıkları")
    order = np.argsort(prob)[::-1]
    for idx in order:
        name = class_names[int(idx)]
        p = float(prob[int(idx)])
        st.progress(p)
        st.write(f"`{name}`: {p:.4f}")

    st.divider()
    st.subheader("🧾 İstersen doğruluğu kontrol et")
    st.write("Eğer bu görselin gerçek sınıfını biliyorsan seç. Modelin doğru mu yanlış mı bildiğini gösterelim.")

    true_label = st.selectbox("Gerçek sınıf (opsiyonel)", ["(bilmiyorum)"] + class_names)

    if true_label != "(bilmiyorum)":
        is_correct = (true_label == pred_name)
        if is_correct:
            st.success(f"✅ Doğru bildi! (Gerçek: {true_label}, Tahmin: {pred_name})")
        else:
            st.error(f"❌ Yanlış bildi. (Gerçek: {true_label}, Tahmin: {pred_name})")
