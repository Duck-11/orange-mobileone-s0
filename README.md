# Orange Quality Check — GOOD/BAD với MobileOne-S0

Phân loại chất lượng cam (GOOD/BAD) bằng **MobileOne-S0** (PyTorch/`timm`, export ONNX FP32).
`app.py` chạy camera/ảnh/video: OpenCV tách từng quả cam, MobileOne-S0 phân loại từng quả.

> **Lưu ý về bản cũ:** project trước đây dùng MobileNetV3-Small (Keras/TFLite). Các file đó
> nằm trong `legacy/mobilenet_NOT_MOBILEONE/` và **không phải MobileOne-S0**. Kết quả của bản cũ
> không được tính cho MobileOne-S0.

## 1. Model

| Mục | Giá trị |
|---|---|
| Kiến trúc | MobileOne-S0 (backbone `timm.create_model('mobileone_s0.apple_in1k', num_classes=0)`) + Dropout(0.2) + Linear(1024→1) |
| Weights khởi tạo | ImageNet, nguồn `https://huggingface.co/timm/mobileone_s0.apple_in1k` |
| Bằng chứng nguồn | `training/orange/mobileone_source.json` (phiên bản `timm`, `pretrained_cfg`, `weights_sha256`, số tham số backbone 4,268,272) |
| License | Xem LICENSE của `apple/ml-mobileone`, `timm` và model card trên Hugging Face (chưa đối chiếu trong README này) |
| Nhãn | `GOOD = 0`, `BAD = 1`; model xuất `p_BAD` |
| Input ONNX | RGB, NCHW, float32, giá trị 0–255. Mean/std (lấy từ `pretrained_cfg` của weights) nằm **trong** model |
| Deploy | Đã reparameterize (gộp nhánh) rồi export ONNX FP32 (`convert_to_onnx.py`) |
| Parity | train vs deploy (PyTorch FP32): 4.06e-08; ONNX vs PyTorch deploy: 9.80e-08 (dung sai đặt trước: 1e-4; đo trên tensor ngẫu nhiên) |

## 2. Cách `app.py` hoạt động

```
Camera/ảnh/video → OpenCV (ngưỡng HSV + contour) → mỗi vùng 1 box
  → crop → resize 224×224 (INTER_AREA), BGR→RGB → MobileOne-S0 (ONNX)
  → p_BAD >= threshold ? BAD : GOOD → vẽ box + nhãn + % tin cậy
```
Bước tách quả là **phát hiện bằng màu/hình dạng, không phải AI detector** (không dùng YOLO).
Các quả cam chạm/chồng nhau có thể bị gộp thành một khung. Chỉnh `HSV_LOWER`, `HSV_UPPER`,
`MIN_CONTOUR_AREA_RATIO` ở đầu `app.py` nếu phát hiện thiếu/thừa.

## 3. Dữ liệu và huấn luyện (chạy riêng, không phải baseline chung của nhóm)

- 5 lớp gốc: `fresh_orange` → GOOD; `blackspotted_orange`, `cankered_orange`, `rotten_orange`, `green_orange` → BAD.
- Chia ngẫu nhiên phân tầng 70/15/15, seed 42 (giữ nguyên split của lần chạy MobileNet cũ). **Không phải** manifest chung đã ký `7851/2059`; `green_orange` ở đây là BAD.
- Huấn luyện 2 giai đoạn: head (backbone đóng băng, tối đa 12 epoch, lr 1e-3) rồi fine-tune 2 stage cuối (tối đa 18 epoch, lr 1e-5). BatchNorm đóng băng cả hai giai đoạn. Adam, loss BCE có class weight, early stopping 5, giảm lr khi val_loss không cải thiện.
- Checkpoint tốt nhất theo `val_loss` toàn cục trên cả hai giai đoạn (`training/outputs/models/best_model.pt`).
- Threshold: chọn trên validation theo F1 BAD. Validation hoàn hảo trên toàn dải 0.10–0.90 nên chọn điểm giữa vùng hòa điểm = **0.50**.

## 4. Kết quả đã đo

| Tập | Ảnh | Threshold | Accuracy | Precision BAD | Recall BAD | F1 BAD | ROC-AUC |
|---|---|---|---|---|---|---|---|
| internal_test (split ngẫu nhiên) | — | 0.10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dataset ngoài `dataset_3` (fresh/rotten) | 4261 | 0.50 | 0.9162 | 0.9125 | 0.9254 | 0.9189 | 0.9722 |
| test_moi | 99 | 0.50 | 0.9596 | 0.9559 | 0.9848 | 0.9701 | — |

Dataset ngoài ở 0.50: 194 ảnh `fresh` bị đoán BAD, 163 ảnh `rotten` bị đoán GOOD (cùng tập ở 0.10: 254 và 123;
ROC-AUC không đổi theo threshold). Internal test được đo khi threshold còn là 0.10 và chưa đo lại ở 0.50.

`test_moi` (99 ảnh): 4 ảnh sai ở 0.50.
- Ba ảnh `fresh_orange` (`f (22)`, `f (23)`, `f (24)`) bị đoán BAD với p_BAD = 1.000. Khi mở ảnh, cả ba là **cam vỏ xanh**
  (green peel) nhưng đang nằm trong thư mục `fresh_orange`. Chính sách huấn luyện coi `green_orange` là BAD, nên model đoán
  nhất quán với chính sách; nhãn trong `test_moi` mâu thuẫn với chính sách đó. Nếu gán lại ba ảnh này sang BAD thì chỉ còn 1 ảnh sai
  (98/99 = 0.9899); con số này chỉ mang tính tham khảo vì nhãn được sửa sau khi đã xem lỗi. Kết quả chính thức giữ nguyên nhãn gốc.
- Một ảnh `green_orange` (`g (17)`): ảnh tối, cam xanh có nhiều đốm nâu, bị đoán GOOD với p_BAD = 0.334. Đây là lỗi thật của model.
- Một số ảnh trong `test_moi` có góc đen do xoay, dấu hiệu là ảnh đã augment.

Ghi chú khi đọc kết quả:
- Internal test hoàn hảo nhưng dataset ngoài chỉ đạt ~92%: internal test dễ hơn thực tế
  (split ngẫu nhiên có thể có ảnh gần giống nhau ở cả train và test — chưa kiểm tra). Dùng số của dataset ngoài và `test_moi` làm tham chiếu thực tế hơn.
- ROC-AUC của dataset ngoài (0.972) không đổi khi đổi threshold, nên phần lỗi còn lại chủ yếu do khác biệt giữa nguồn ảnh, không chỉnh được bằng threshold.
- Dataset ngoài chỉ kiểm tra `fresh` với `rotten`, không kiểm tra `blackspot`, `canker`, `green`. Chưa kiểm tra có ảnh trùng với tập train hay không.
- `test_moi`: mẫu nhỏ (99 ảnh), chưa đủ để kết luận chắc chắn.
- Chưa so sánh với MobileNetV3-Small cũ trên cùng dữ liệu.

## 5. Cấu trúc

```
orange_project/
├── app.py                 chạy camera/ảnh/video (ONNX Runtime)
├── model.onnx             model deploy (FP32, đã reparameterize)
├── model_meta.json        model ID, dtype, hash, parity, norm_cfg
├── threshold.json         threshold đang dùng
├── requirements.txt       thư viện CHẠY app.py
├── convert_to_onnx.py     best_model.pt → model.onnx (+ kiểm tra parity)
├── eval_folder.py         đánh giá model.onnx trên thư mục ảnh
├── tests/                 pytest (kiến trúc, parity, preprocessing, pipeline)
├── tools/                 prove_source.py, ...
├── legacy/                bản MobileNetV3-Small cũ — NOT_MOBILEONE
└── training/              pipeline train PyTorch
    ├── main.py, config.yaml, requirements.txt
    └── orange/            data.py, model.py, trainer.py, evaluation.py, mobileone_source.json
```

## 6. Cài đặt và chạy

Chạy ứng dụng (Windows):
```cmd
py -3.11 -m venv .venv-app
.venv-app\Scripts\activate
python -m pip install -r requirements.txt
python app.py --camera                # webcam; --camera-id 1 nếu có nhiều camera
python app.py --image test.jpg --save
python app.py --video test.mp4
python app.py --camera --threshold 0.5
```
Nhấn `q` để thoát.

Train lại (cần `torch`, nên dùng bản có CUDA nếu có GPU NVIDIA):
```cmd
cd training
python -m pip install -r requirements.txt
python main.py run          # giữ split cũ nếu outputs\splits đã có
cd ..
python convert_to_onnx.py --checkpoint training\outputs\models\best_model.pt
python eval_folder.py --root "D:\NCKH\test_moi"
```
Trước khi train, sửa `data.source_dir` (và `external_datasets.path`) trong `training/config.yaml`.

Kiểm thử: `pytest tests -v`.

## 7. Raspberry Pi

Cần copy: `app.py`, `requirements.txt`, `model.onnx`, `model_meta.json`, `threshold.json`.
`requirements.txt` dùng `onnxruntime`. **Chưa kiểm chứng trên Raspberry Pi** (cài đặt `onnxruntime` và tốc độ trên Pi cần thử riêng).
Chạy qua SSH không có màn hình sẽ lỗi `cv2.imshow`; cần chạy trên desktop của Pi hoặc dùng VNC.

## 8. Giới hạn đã biết

- Phát hiện quả bằng màu/hình dạng, phụ thuộc ánh sáng và nền.
- Không dùng manifest chung; kết quả chỉ là của lần train riêng này.
- Chưa có reparameterize parity trên ảnh thật (hiện đo trên tensor ngẫu nhiên) và chưa có kiểm thử tiền xử lý train/app trên cùng ảnh.