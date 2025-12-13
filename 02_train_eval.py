import os
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# =========================
# AYARLAR
# =========================
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

DATASET_DIR = r"output_dataset"  # split scriptinin çıktısı
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR   = os.path.join(DATASET_DIR, "val")
TEST_DIR  = os.path.join(DATASET_DIR, "test")

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUT_DIR, "recycle_best.keras")
CLASSNAMES_PATH = os.path.join(OUT_DIR, "class_names.json")

tf.random.set_seed(SEED)
np.random.seed(SEED)

# =========================
# DATA LOAD
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Sınıflar:", class_names)

# class_names kaydet (app otomatik okuyacak)
with open(CLASSNAMES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

AUTOTUNE = tf.data.AUTOTUNE

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="augmentation")

def prep(ds, training=False):
    if training:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y), num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(AUTOTUNE)

train_ds_p = prep(train_ds, training=True)
val_ds_p   = prep(val_ds, training=False)
test_ds_p  = prep(test_ds, training=False)

# =========================
# MODEL
# =========================
base = MobileNetV2(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
base.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# CALLBACKS
# =========================
cbs = [
    ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
]

# =========================
# TRAIN
# =========================
history = model.fit(
    train_ds_p,
    validation_data=val_ds_p,
    epochs=EPOCHS,
    callbacks=cbs
)

best_model = tf.keras.models.load_model(BEST_MODEL_PATH)

# =========================
# PLOTS: Accuracy/Loss
# =========================
def plot_history(hist):
    acc = hist.history.get("accuracy", [])
    val_acc = hist.history.get("val_accuracy", [])
    loss = hist.history.get("loss", [])
    val_loss = hist.history.get("val_loss", [])

    plt.figure()
    plt.plot(acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.title("Epoch - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "epoch_accuracy.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Epoch - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "epoch_loss.png"), dpi=200, bbox_inches="tight")
    plt.close()

plot_history(history)

# =========================
# TEST EVAL + REPORT
# =========================
test_loss, test_acc = best_model.evaluate(test_ds_p, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_prob = best_model.predict(test_ds_p, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\nClassification Report:\n", report)

with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {test_acc:.6f}\nTest Loss: {test_loss:.6f}\n\n")
    f.write(report)

# =========================
# CONFUSION MATRIX (IMAGE)
# =========================
cm = confusion_matrix(y_true, y_pred)
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(num_classes), class_names, rotation=45, ha="right")
plt.yticks(range(num_classes), class_names)
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
plt.close()

# =========================
# ROC-AUC (One-vs-Rest)
# =========================
y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

plt.figure()
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curves (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "roc_auc.png"), dpi=200, bbox_inches="tight")
plt.close()

print("\n✅ Bitti.")
print("✅ Model:", BEST_MODEL_PATH)
print("✅ Sınıflar dosyası:", CLASSNAMES_PATH)
print("✅ Grafikler ve rapor:", OUT_DIR)
