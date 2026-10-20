from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# CẤU HÌNH

APP_DIR = Path(__file__).resolve().parent

IMAGE_SIZE = 224
LABEL_GOOD = "GOOD"
LABEL_BAD = "BAD"
DEFAULT_THRESHOLD = 0.5

MODEL_ONNX_PATH = APP_DIR / "model.onnx"
MODEL_META_PATH = APP_DIR / "model_meta.json"
THRESHOLD_JSON_PATH = APP_DIR / "threshold.json"

# Cấu hình detector.
#
# Detector không phải model AI object detection. Nó dùng HSV + contour để
# tìm vùng có hình dạng giống quả cam trước khi đưa crop vào MobileOne-S0.

# HSV rộng để nhận cam vàng, cam, cam xanh và cam bị bệnh.
HSV_LOWER = np.array([0, 75, 45], dtype=np.uint8)
HSV_UPPER = np.array([45, 255, 255], dtype=np.uint8)

# Diện tích contour so với ảnh detection.
MIN_CONTOUR_AREA_RATIO = 0.004
MAX_CONTOUR_AREA_RATIO = 0.12

# Quả cam thường gần tròn hoặc bầu dục.
MIN_ASPECT_RATIO = 0.60
MAX_ASPECT_RATIO = 1.65

# Hình dạng contour.
MIN_CIRCULARITY = 0.52
MIN_EXTENT = 0.48
MIN_SOLIDITY = 0.72

# Màu trung bình của vùng contour.
# Tăng saturation giúp loại nhiều vùng da, áo và nền.
MIN_MEAN_SATURATION = 75.0

BOX_PADDING_RATIO = 0.08
DETECTION_MAX_WIDTH = 640

# THRESHOLD
def load_threshold() -> float:
    """Đọc threshold đã chọn khi train."""

    if THRESHOLD_JSON_PATH.exists():
        try:
            data = json.loads(THRESHOLD_JSON_PATH.read_text(encoding="utf-8"))
            value = float(data.get("threshold", data.get("final_threshold", DEFAULT_THRESHOLD),))

            if 0.0 <= value <= 1.0:
                return value

        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    print(
        f"[Cảnh báo] Không đọc được {THRESHOLD_JSON_PATH.name}. "
        f"Dùng threshold mặc định = {DEFAULT_THRESHOLD:.2f}."
    )
    return DEFAULT_THRESHOLD

# MODEL
class OrangeClassifier:
    """Load MobileOne-S0 (ONNX, FP32, đã reparameterize) và dự đoán xác suất BAD.
    Input ONNX: RGB NCHW float32 0..255; mean/std nằm trong graph (xem model_meta.json)."""

    def __init__(self) -> None:
        import onnxruntime as ort
        if not MODEL_ONNX_PATH.exists():
            raise FileNotFoundError(
                "Không tìm thấy model.onnx.\n"
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

# PHÁT HIỆN QUẢ CAM
def detect_orange_boxes(frame_bgr: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    height, width = frame_bgr.shape[:2]

    # Resize để detection nhanh.
    scale = 1.0
    detect_frame = frame_bgr

    if width > DETECTION_MAX_WIDTH:
        scale = DETECTION_MAX_WIDTH / float(width)
        detect_frame = cv2.resize(
            frame_bgr,(DETECTION_MAX_WIDTH,int(height * scale),),
            interpolation=cv2.INTER_AREA,)

    # Làm mượt + HSV.
    blurred = cv2.GaussianBlur(detect_frame,(5, 5),0,)

    hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV,)
    mask = cv2.inRange(hsv,HSV_LOWER,HSV_UPPER,)

    # Morphology.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7),)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations=1,)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=2,)

    # Tìm contour.
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,)
    frame_area = detect_frame.shape[0] * detect_frame.shape[1]
    min_area = frame_area * MIN_CONTOUR_AREA_RATIO
    max_area = frame_area * MAX_CONTOUR_AREA_RATIO
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # 1. Diện tích.
        if area < min_area or area > max_area:
            continue
        # 2. Chu vi + circularity.
        perimeter = cv2.arcLength(contour, True,)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue
        # 3. Bounding box + aspect ratio.
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        aspect_ratio = w / float(h)
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            continue
        # 4. Extent.
        bounding_area = w * h
        extent = area / float(bounding_area)
        if extent < MIN_EXTENT:
            continue
        # 5. Solidity.
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = area / float(hull_area)
        if solidity < MIN_SOLIDITY:
            continue
        # 6. Saturation trung bình.
        # Giúp loại nhiều vùng da người, áo và nền có màu nâu/đỏ.
        contour_mask = np.zeros(hsv.shape[:2],dtype=np.uint8,)
        cv2.drawContours(contour_mask,[contour],-1,255,-1,)
        mean_h, mean_s, mean_v, _ = cv2.mean(hsv, mask=contour_mask,)
        if mean_s < MIN_MEAN_SATURATION:
            continue
        # Điểm hình dạng.
        # Dùng để ưu tiên contour giống quả cam hơn khi có nhiều ứng viên.
        shape_score = 0.45 * circularity + 0.30 * solidity + 0.25 * extent
        candidates.append(
            {
                "box": (x, y, w, h),
                "area": area,
                "score": shape_score,
                "mean_h": mean_h,
                "mean_s": mean_s,
                "mean_v": mean_v,
            }
        )
    # Sắp xếp ứng viên theo score.
    candidates.sort(key=lambda item: item["score"],reverse=True,)
    selected = []
    # Non-Maximum Suppression đơn giản.
    # Loại box chồng lên nhau quá nhiều.
    for candidate in candidates:
        x, y, w, h = candidate["box"]
        keep = True
        for selected_box in selected:
            sx, sy, sw, sh = selected_box
            ix1 = max(x, sx)
            iy1 = max(y, sy)
            ix2 = min(x + w, sx + sw)
            iy2 = min(y + h, sy + sh)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            intersection = (ix2 - ix1) * (iy2 - iy1)
            area1 = w * h
            area2 = sw * sh
            overlap = intersection / float(min(area1, area2))
            if overlap > 0.60:
                keep = False
                break
        if keep:
            selected.append(candidate["box"])
    # Map tọa độ từ ảnh detection về ảnh gốc.
    boxes = []
    for x, y, w, h in selected:
        x0 = x
        y0 = y
        x1 = x + w
        y1 = y + h
        if scale != 1.0:
            x0 = int(x0 / scale)
            y0 = int(y0 / scale)
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
        # Padding nhỏ để crop không sát mép quả.
        pad_w = int((x1 - x0) * BOX_PADDING_RATIO)
        pad_h = int((y1 - y0) * BOX_PADDING_RATIO)
        x0 = max(0,x0 - pad_w,)
        y0 = max(0,y0 - pad_h,)
        x1 = min(width,x1 + pad_w,)
        y1 = min(height,y1 + pad_h,)
        boxes.append((x0,y0,x1 - x0,y1 - y0,))
    # Trái -> phải, trên -> dưới.
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes

# VẼ KẾT QUẢ
def draw_result(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    confidence: float,
) -> None:
    x, y, w, h = box
    if label == LABEL_GOOD:
        color = (0, 200, 0)
    else:
        color = (0, 0, 255)
    cv2.rectangle(frame,(x, y),(x + w, y + h),color,2,)
    text = f"{label} {confidence * 100:.1f}%"
    (text_w,text_h,),baseline = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.6,2,)
    label_y = max(y,text_h + baseline + 8,)
    cv2.rectangle(
        frame,(x,label_y - text_h - baseline - 4,),
        (x + text_w + 8,label_y + baseline - 4,),color,-1,)
    cv2.putText(frame,text,(x + 4, label_y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 255, 255),2,cv2.LINE_AA,)

# XỬ LÝ FRAME
def process_frame(
    frame: np.ndarray,
    classifier: OrangeClassifier,
    threshold: float,
) -> tuple[np.ndarray, int]:
    """Phát hiện, phân loại và vẽ kết quả cho một frame."""
    boxes = detect_orange_boxes(frame)
    crops = []
    valid_boxes = []
    for x, y, w, h in boxes:
        crop = frame[
            y: y + h, x: x + w,
        ]
        if crop.size == 0:
            continue
        crops.append(crop)
        valid_boxes.append((x, y, w, h))
    if crops:
        probabilities_bad = classifier.predict_probability_bad(crops)
        for box, probability_bad in zip(
            valid_boxes,probabilities_bad,
        ):
            if probability_bad >= threshold:
                label = LABEL_BAD
                confidence = probability_bad
            else:
                label = LABEL_GOOD
                confidence = 1.0 - probability_bad
            draw_result(frame,box,label,confidence,)
    return frame, len(valid_boxes)

# CAMERA
def run_camera(
    classifier: OrangeClassifier,
    threshold: float,
    camera_id: int,
) -> None:
    """Chạy webcam."""
    capture = cv2.VideoCapture(camera_id)
    if not capture.isOpened():
        raise RuntimeError(f"Không mở được webcam (camera_id={camera_id}).")
    # Giảm độ phân giải để detector và TFLite chạy nhẹ hơn.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280,)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720,)
    print("Đang chạy webcam. Nhấn 'q' để thoát.")
    previous_time = time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("[Cảnh báo] Không đọc được frame từ webcam.")
                break
            frame, num_oranges = process_frame(
                frame,classifier,threshold,
            )
            current_time = time.perf_counter()
            instant_fps = 1.0 / max(current_time - previous_time,1e-6,)
            previous_time = current_time
            if fps > 0:
                fps = fps * 0.9 + instant_fps * 0.1
            else:
                fps = instant_fps
            cv2.putText(
                frame, f"FPS: {fps:.1f} | Oranges: {num_oranges}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"Threshold: {threshold:.2f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.60, (255, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.imshow(
                "Orange Quality Check - Camera",
                frame,
            )
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

# VIDEO
def run_video(
    classifier: OrangeClassifier,
    threshold: float,
    video_path: Path,
) -> None:
    """Chạy video."""
    if not video_path.exists():
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")
    print(f"Đang chạy video: {video_path}. " "Nhấn 'q' để thoát.")
    previous_time = time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Đã hết video.")
                break
            frame, num_oranges = process_frame(frame, classifier, threshold,)
            current_time = time.perf_counter()
            instant_fps = 1.0 / max(current_time - previous_time, 1e-6,)
            previous_time = current_time
            if fps > 0:
                fps = fps * 0.9 + instant_fps * 0.1
            else:
                fps = instant_fps
            cv2.putText(
                frame, f"FPS: {fps:.1f} | Oranges: {num_oranges}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.imshow(
                "Orange Quality Check - Video", frame,
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

# IMAGE
def run_image(
    classifier: OrangeClassifier,
    threshold: float,
    image_path: Path,
    save: bool,
) -> None:
    """Chạy trên một ảnh."""
    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Không đọc được ảnh: {image_path}")
    frame, num_oranges = process_frame(
        frame, classifier, threshold,
    )
    print(f"Phát hiện {num_oranges} quả cam " f"trong ảnh: {image_path.name}")
    if save:
        output_path = image_path.with_name(
            image_path.stem + "_result" + image_path.suffix
        )
        cv2.imwrite(str(output_path), frame,)
        print(f"Đã lưu ảnh kết quả: {output_path}")
    cv2.imshow("Orange Quality Check - Image", frame,)
    print("Nhấn phím bất kỳ trên cửa sổ ảnh để đóng.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ARGUMENT PARSER
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Phân loại chất lượng quả cam " "GOOD/BAD bằng MobileOne-S0.")
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--camera", action="store_true", help="Chạy webcam.",
    )
    mode_group.add_argument(
        "--image", type=str, default=None, help="Đường dẫn ảnh cần test.",
    )
    mode_group.add_argument(
        "--video", type=str, default=None, help="Đường dẫn video cần test.",
    )
    parser.add_argument(
        "--camera-id", type=int, default=0, help="ID webcam, mặc định là 0.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="Ghi đè threshold GOOD/BAD, giá trị 0-1.",
    )
    parser.add_argument(
        "--save", action="store_true", help="Lưu ảnh kết quả, chỉ áp dụng với --image.",
    )
    return parser

# MAIN
def main() -> int:
    args = build_parser().parse_args()
    try:
        classifier = OrangeClassifier()
    except Exception as exc:
        print(f"[Lỗi model] {exc}")
        return 1
    threshold = args.threshold if args.threshold is not None else load_threshold()
    if not 0.0 <= threshold <= 1.0:
        print("[Lỗi] Threshold phải nằm trong khoảng 0 đến 1.")
        return 1
    print(f"[Threshold] {threshold:.4f} " "(BAD nếu xác suất BAD >= threshold)")
    try:
        if args.image:
            run_image(
                classifier, threshold,
                Path(args.image).expanduser().resolve(), args.save,
            )
        elif args.video:
            run_video(
                classifier, threshold, Path(args.video).expanduser().resolve(),
            )
        else:
            # --camera hoặc không truyền tham số đều mở webcam.
            run_camera(
                classifier, threshold, args.camera_id,
            )
        return 0
    except KeyboardInterrupt:
        print("Đã dừng theo yêu cầu người dùng.")
        return 130
    except Exception as exc:
        print(f"[Lỗi] {exc}")
        return 1
if __name__ == "__main__":
    sys.exit(main())