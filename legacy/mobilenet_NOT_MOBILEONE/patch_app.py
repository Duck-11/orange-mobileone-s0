"""Vá app.py: thay OrangeClassifier (TFLite/Keras) bằng ONNX Runtime MobileOne-S0.
Tạo app.py.bak_mobilenet trước. python tools/patch_app.py path/to/app.py"""
import re, shutil, sys
from pathlib import Path

NEW_CLASS = '''class OrangeClassifier:
    """Load MobileOne-S0 (ONNX, FP32, đã reparameterize) và dự đoán xác suất BAD.
    Input ONNX: RGB NCHW float32 0..255; mean/std nằm trong graph (xem model_meta.json)."""

    def __init__(self) -> None:
        import onnxruntime as ort
        if not MODEL_ONNX_PATH.exists():
            raise FileNotFoundError(
                "Không tìm thấy model.onnx.\\n"
                f"Chạy convert_to_onnx.py để tạo tại: {APP_DIR}")
        self.session = ort.InferenceSession(
            str(MODEL_ONNX_PATH), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.backend = "onnxruntime"
        meta = {}
        if MODEL_META_PATH.exists():
            meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
        self.model_id = meta.get("model_id", "UNKNOWN")
        print(f"[Model] {self.model_id} | dtype={meta.get('dtype', '?')} | "
              f"backend={self.backend}")

    def predict_probability_bad(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Nhận crop BGR và trả về xác suất BAD."""
        if not crops_bgr:
            return np.array([], dtype=np.float32)
        batch = np.zeros((len(crops_bgr), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        for i, crop in enumerate(crops_bgr):
            resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            # KHÔNG chia 255 / không chuẩn hoá ở đây: mean/std nằm trong model.
            batch[i] = rgb.astype(np.float32).transpose(2, 0, 1)
        out = self.session.run(None, {self.input_name: batch})[0].reshape(-1)
        return np.clip(out.astype(np.float32), 0.0, 1.0)

'''

p = Path(sys.argv[1])
src = p.read_text(encoding="utf-8")
a = src.find("class OrangeClassifier:")
b = src.find("# PHÁT HIỆN QUẢ CAM")
if a < 0 or b < 0 or b < a:
    sys.exit("Không tìm thấy mốc 'class OrangeClassifier:' / '# PHÁT HIỆN QUẢ CAM' — app.py khác bản mẫu, dừng.")
shutil.copy(p, p.with_name(p.name + ".bak_mobilenet"))
src = src[:a] + NEW_CLASS + src[b:]
src, n = re.subn(r'MODEL_TFLITE_PATH = APP_DIR / "model.tflite"\s*\nMODEL_KERAS_PATH = APP_DIR / "model.keras"',
                 'MODEL_ONNX_PATH = APP_DIR / "model.onnx"\nMODEL_META_PATH = APP_DIR / "model_meta.json"', src)
if n != 1:
    sys.exit("Không thay được khối đường dẫn model — dừng (đã có file .bak).")
src = src.replace("MobileNetV3-Small", "MobileOne-S0").replace("vào MobileNet.", "vào MobileOne-S0.")
p.write_text(src, encoding="utf-8")
print("Đã vá", p, "| backup:", p.name + ".bak_mobilenet")
