# TrioLearn/evaluation/smoke_check_modality.py
import argparse, os, pickle, joblib
from pathlib import Path
import numpy as np
import pandas as pd

def find_repo_root(script_path: Path) -> Path:
    return script_path.resolve().parents[1]

def first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

def build_counts_row(course=10, reading=5, video=20) -> pd.DataFrame:
    return pd.DataFrame([{
        "course_count": int(course),
        "reading_count": int(reading),
        "video_count": int(video),
    }])

def build_demo_row(gender="U", region="Unknown", highest_education="Unknown", age_band="Unknown") -> pd.DataFrame:
    return pd.DataFrame([{
        "gender": str(gender).strip(),
        "region": str(region).strip(),
        "highest_education": str(highest_education).strip(),
        "age_band": str(age_band).strip(),
    }])

def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def load_label_encoder(enc_path: Path | None):
    if not enc_path:
        return None
    try:
        return joblib.load(enc_path)
    except Exception:
        return None

def make_label_mapper(model, label_encoder):
    """
    Returns a function: int -> label string.
    Preference:
      1) If model.classes_ are strings, use them.
      2) Else if LabelEncoder is present (string classes_), use it.
      3) Else fallback {0:'video',1:'book',2:'course'}.
    """
    model_classes = getattr(model, "classes_", None)

    # 1) If model.classes_ exist and look like strings, use them
    if model_classes is not None:
        as_list = list(model_classes)
        if any(isinstance(x, str) for x in as_list):
            return lambda i: str(as_list[int(i)])

    # 2) Prefer LabelEncoder if present and has string classes
    if label_encoder is not None and getattr(label_encoder, "classes_", None) is not None:
        le_classes = list(label_encoder.classes_)
        # If LabelEncoder classes are strings, use them (typical case)
        if any(isinstance(x, str) for x in le_classes):
            return lambda i: str(le_classes[int(i)])

    # 3) Final fallback (only use if you're sure of the order)
    fallback = {0: "video", 1: "book", 2: "course"}
    return lambda i: fallback.get(int(i), f"class_{int(i)}")

def normalize_to_class_index(pred) -> int | str:
    """
    Accepts many shapes:
      - scalar int               -> class id
      - scalar str               -> class label (return str)
      - 1D vector probs/one-hot -> argmax
      - 2D (1, n_classes)       -> argmax
      - 1D single element       -> that value
    """
    arr = np.asarray(pred)
    if arr.ndim == 0:
        # scalar (could be int id or str)
        val = arr.item()
        try:
            return int(val)
        except Exception:
            return str(val)
    if arr.ndim == 1:
        if arr.size == 1:
            return int(arr[0])
        # vector -> argmax
        return int(np.argmax(arr))
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return int(np.argmax(arr[0]))
        # generic fallback
        return int(np.argmax(arr))
    # unknown, return string
    return str(arr)

def main():
    parser = argparse.ArgumentParser(description="Modality classifier smoke check (robust to outputs).")
    parser.add_argument("--model", type=str, default=None, help="Path to model pickle (.pkl).")
    parser.add_argument("--encoder", type=str, default=None, help="Optional label_encoder.pkl path (not required).")
    parser.add_argument("--mode", choices=["counts", "demo"], default="counts",
                        help="Feature mode: counts (default) or demo (demographics).")
    parser.add_argument("--course_count", type=int, default=10)
    parser.add_argument("--reading_count", type=int, default=5)
    parser.add_argument("--video_count", type=int, default=20)
    parser.add_argument("--gender", type=str, default="U")
    parser.add_argument("--region", type=str, default="Unknown")
    parser.add_argument("--highest_education", type=str, default="Unknown")
    parser.add_argument("--age_band", type=str, default="Unknown")
    args = parser.parse_args()

    here = Path(__file__).resolve()
    repo = find_repo_root(here)

    default_model_candidates = [
        args.model,
        repo / "outputs" / "models" / "xgb_best_model.pkl",
        repo / "outputs" / "models" / "xgb_best_media_type_model.pkl",
        repo / "outputs" / "models" / "xgb_smote_model.pkl",
    ]
    model_path = first_existing(default_model_candidates)
    if not model_path:
        print("[ERROR] Could not locate a model. Tried:")
        for c in default_model_candidates:
            if c:
                print("  -", c)
        print("\nTip: pass --model C:\\Users\\jvlas\\source\\repos\\TrioLearn\\outputs\\models\\xgb_best_model.pkl")
        return

    default_encoder_candidates = [
        args.encoder,
        repo / "outputs" / "models" / "label_encoder.pkl",
    ]
    enc_path = first_existing(default_encoder_candidates)

    print("[info] repo root:", repo)
    print("[info] model path:", model_path)
    print("[info] encoder path:", enc_path if enc_path else "(none)")

    model = load_model(model_path)
    label_encoder = load_label_encoder(enc_path)
    to_label = make_label_mapper(model, label_encoder)

    if args.mode == "counts":
        X = build_counts_row(args.course_count, args.reading_count, args.video_count)
    else:
        X = build_demo_row(args.gender, args.region, args.highest_education, args.age_band)

    raw_pred = model.predict(X)
    # Some models return a scalar, some return array-like; handle both
    class_or_label = normalize_to_class_index(raw_pred)

    if isinstance(class_or_label, str) and not class_or_label.isdigit():
        # Model already returned a string label
        print("Predicted modality:", class_or_label)
    else:
        print("Predicted modality:", to_label(int(class_or_label)))

if __name__ == "__main__":
    main()
