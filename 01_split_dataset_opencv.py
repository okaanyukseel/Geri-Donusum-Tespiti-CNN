import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# =========================
# AYARLAR
# =========================
RAW_DIR = r"C:\DERSLER\goruntu isleme\trashnet\dataset-resized"   # raw_dataset klasörü (cam/metal/kagit/plastik)
OUT_DIR = r"output_dataset"

TEST_SIZE = 0.15
VAL_SIZE  = 0.15
SEED = 42

RESIZE_TO = (224, 224)   # (w, h)  None yaparsan resize kapalı
SAVE_EXT = ".jpg"        # .png da yapabilirsin

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def safe_imread(path: str):
    img = cv2.imread(path)
    if img is None or img.size == 0:
        return None
    return img

def ensure_dirs(out_dir: str, splits, class_names):
    for sp in splits:
        for c in class_names:
            os.makedirs(os.path.join(out_dir, sp, c), exist_ok=True)

def save_image(img, out_path: str):
    if out_path.lower().endswith((".jpg", ".jpeg")):
        cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        cv2.imwrite(out_path, img)

def main():
    raw_path = Path(RAW_DIR)
    if not raw_path.exists():
        raise FileNotFoundError(f"RAW_DIR bulunamadı: {RAW_DIR}")

    class_names = sorted([p.name for p in raw_path.iterdir() if p.is_dir()])
    if len(class_names) < 2:
        raise ValueError("RAW_DIR içinde sınıf klasörleri yok (cam/metal/kagit/plastik gibi).")

    all_files, all_labels = [], []
    for ci, cname in enumerate(class_names):
        files = [p for p in (raw_path / cname).rglob("*") if p.is_file() and is_image_file(p)]
        for f in files:
            all_files.append(str(f))
            all_labels.append(ci)

    all_files = np.array(all_files)
    all_labels = np.array(all_labels)

    if len(all_files) == 0:
        raise ValueError("Hiç görsel bulunamadı. Klasör ve uzantıları kontrol et.")

    print("Sınıflar:", class_names)
    for ci, cname in enumerate(class_names):
        print(f"  {cname}: {(all_labels==ci).sum()}")

    # 1) test ayır
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        all_files, all_labels,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=all_labels
    )

    # 2) trainval içinden val ayır (toplam oranı tutturmak için)
    val_ratio_of_trainval = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio_of_trainval,
        random_state=SEED,
        stratify=y_trainval
    )

    splits = {
        "train": (X_train, y_train),
        "val":   (X_val, y_val),
        "test":  (X_test, y_test),
    }

    print("\nSplit sayıları:")
    for sp, (X, _) in splits.items():
        print(f"  {sp}: {len(X)}")

    ensure_dirs(OUT_DIR, splits=["train", "val", "test"], class_names=class_names)

    def export_split(split_name, X, y):
        bad = 0
        for idx, (src_path, label) in enumerate(zip(X, y)):
            img = safe_imread(src_path)
            if img is None:
                bad += 1
                continue

            if RESIZE_TO is not None:
                img = cv2.resize(img, RESIZE_TO, interpolation=cv2.INTER_AREA)

            cname = class_names[int(label)]
            out_name = f"{cname}_{idx:06d}{SAVE_EXT}"
            out_path = os.path.join(OUT_DIR, split_name, cname, out_name)
            save_image(img, out_path)

        print(f"{split_name} tamamlandı. Bozuk/okunamayan: {bad}")

    for sp, (X, y) in splits.items():
        export_split(sp, X, y)

    print("\n✅ Bitti. Çıktı:", OUT_DIR)
    print("Not: Eğitim kodu sınıf isimlerini otomatik okuyacak.")

if __name__ == "__main__":
    main()
