# Orange Quality Check — Runtime (GOOD/BAD, MobileNetV3-Small)

Bản chạy thực tế gọn nhẹ, chỉ **một file `app.py`**, dựng từ project training
gốc (`main.py`, `orange/model.py`, `orange/data.py`, `orange/evaluation.py`,
`config.yaml`). Mục tiêu: chạy trên máy tính để test, rồi copy nguyên cụm
sang thẻ SD Raspberry Pi.

## 0. Vì sao cần thêm bước detection bằng OpenCV?

Model `MobileNetV3-Small` hiện tại của bạn là **model phân loại ảnh**
(1 ảnh → GOOD hoặc BAD), **không phải object detection model**, nên nó
không thể tự vẽ nhiều bounding box cho nhiều quả cam trong một khung hình.

`app.py` giải quyết việc này bằng một bước detection **rất nhẹ, dùng
OpenCV** (ngưỡng màu HSV + contour) để tách từng quả cam thành từng vùng
riêng, rồi mới crop từng vùng đưa vào MobileNetV3-Small:

```
Camera/ảnh/video → OpenCV: tách từng vùng quả cam (contour) → mỗi vùng 1 box
                  → crop từng box → resize 224x224 → MobileNetV3-Small
                  → GOOD/BAD + confidence → vẽ box + nhãn → hiển thị
```

**Giới hạn cần biết:** đây là detection dựa theo màu sắc/hình dạng, không
phải AI detection, nên nếu nhiều quả cam **chạm/chồng sát nhau** thì có thể
bị dính vào một contour duy nhất. Đây là đánh đổi để giữ hệ thống đủ nhẹ
cho Raspberry Pi (không dùng YOLO). Nếu detect thiếu/dư quả trong điều kiện
ánh sáng/nền của bạn, chỉnh 3 hằng số ở đầu `app.py`:

```python
HSV_LOWER, HSV_UPPER          # khoảng màu vỏ cam (vàng cam → nâu sẫm)
MIN_CONTOUR_AREA_RATIO        # diện tích tối thiểu tính theo % khung hình
```

## 1. Model và preprocessing — giữ nguyên 100% so với project gốc

Đã kiểm tra kỹ `orange/model.py` và `orange/data.py` trước khi viết `app.py`:

| Thuộc tính | Giá trị (giữ nguyên) |
|---|---|
| Kiến trúc | MobileNetV3-Small (`include_top=False`) + GlobalAveragePooling + Dropout + Dense(1, sigmoid) |
| Input size | 224×224 |
| Preprocessing | **Không tự chia 255 / không tự chuẩn hoá thủ công.** Model được build với `include_preprocessing=True`, nghĩa là việc chuẩn hoá pixel (0–255 → khoảng chuẩn của MobileNetV3) đã nằm sẵn trong model. `app.py` chỉ resize 224×224, đổi BGR→RGB (vì OpenCV đọc ảnh là BGR còn lúc train ảnh được đọc theo RGB) và ép kiểu `float32`, giống hệt `orange/data.py` gốc. |
| Nhãn | `0 = GOOD`, `1 = BAD` — output model là **xác suất BAD** |
| Threshold | Không cố định cứng trong code, mà đọc từ `threshold.json` — chính là `final_threshold` được `main.py train` chọn trên tập validation (mặc định dò trong khoảng 0.10–0.90, bước 0.01, theo `config.yaml` gốc) |

Không có thay đổi nào về kiến trúc, input size, preprocessing, class mapping
hay threshold logic.

## 2. ⚠️ Lưu ý quan trọng: project ZIP bạn gửi CHƯA có model đã train

Đã kiểm tra toàn bộ ZIP: chỉ có **source code** (`main.py`, `orange/`,
`config.yaml`...), thư mục `outputs/` rỗng — **chưa có file
`best_model.keras` hay trọng số nào**. Vì vậy `app.py` chưa thể chạy ngay
cho tới khi bạn:

1. Train model (bước 3 dưới đây), hoặc
2. Đã có sẵn `best_model.keras` từ máy khác — chỉ cần copy file đó vào
   đúng vị trí `training/outputs/models/best_model.keras` rồi làm tiếp bước 4.

## 3. Cấu trúc project

```
orange_project/
├── app.py                  ← FILE CHẠY THỰC TẾ DUY NHẤT (camera/ảnh/video)
├── convert_to_tflite.py    ← chạy 1 lần sau khi train, để export model.tflite
├── requirements.txt        ← thư viện tối giản để CHẠY app.py
├── README.md                ← chính là file này
├── model.tflite              ← (sẽ được tạo ra ở bước 4) model dùng khi chạy
├── threshold.json             ← (sẽ được tạo ra ở bước 4) threshold đã chọn
└── training/                 ← giữ nguyên pipeline train/evaluate gốc, KHÔNG
    ├── main.py                 cần thiết khi chạy thực tế, chỉ dùng khi
    ├── config.yaml             muốn train lại / train model mới
    ├── requirements.txt         (đầy đủ TensorFlow + pandas + sklearn...)
    ├── README_TRAINING.md       (README gốc của project training)
    └── orange/
        ├── __init__.py
        ├── data.py
        ├── model.py
        └── evaluation.py
```

Khi chạy thực tế trên Raspberry Pi, bạn **chỉ cần copy 4 thứ**:
`app.py`, `requirements.txt`, `model.tflite`, `threshold.json`
(không cần thư mục `training/`).

## 4. Cài đặt & train trên Windows

### 4.1. Cài môi trường training (để tạo model)

```cmd
cd training
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Sửa `data.source_dir` trong `training/config.yaml` trỏ tới dataset của bạn,
rồi train:

```cmd
python main.py run
```

Lệnh này prepare data → train → chọn threshold trên validation → đánh giá
test/external, giống hệt project gốc. Kết quả quan trọng để dùng cho
runtime:

```
training/outputs/models/best_model.keras
training/outputs/selected_threshold.json
```

### 4.2. Export sang model.tflite (khuyến nghị cho Raspberry Pi)

Vẫn trong virtualenv `training` (đã có TensorFlow), quay về thư mục gốc
project rồi chạy:

```cmd
cd ..
python convert_to_tflite.py
```

Script này:

* Convert `training/outputs/models/best_model.keras` → `model.tflite`
  (mặc định **float32, không lượng tử hoá**, để kết quả gần như giống hệt
  bản `.keras`). Nếu muốn model nhẹ/nhanh hơn nữa và chấp nhận lệch nhẹ kết
  quả, thêm `--quantize`.
* Copy threshold đã chọn (`training/outputs/selected_threshold.json`)
  sang `threshold.json` cạnh `app.py`.

Sau bước này, thư mục gốc sẽ có `model.tflite` + `threshold.json`, sẵn sàng
chạy `app.py`.

> Nếu vì lý do nào đó bạn không muốn convert sang `.tflite`, có thể copy
> thẳng `training/outputs/models/best_model.keras` thành `model.keras`
> cạnh `app.py` — `app.py` tự nhận và dùng TensorFlow đầy đủ để chạy.
> Cách này chạy được trên Windows nhưng **không khuyến nghị cho Raspberry
> Pi** vì TensorFlow đầy đủ rất nặng và chậm trên Pi.

### 4.3. Cài môi trường để CHẠY `app.py` trên Windows

```cmd
cd orange_project
py -3.11 -m venv .venv-app
.venv-app\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Chạy webcam

```cmd
python app.py --camera
```

hoặc không cần truyền gì (mặc định webcam):

```cmd
python app.py
```

* Mỗi quả cam trong khung hình có **bounding box riêng** kèm nhãn
  `GOOD`/`BAD` + phần trăm confidence.
* Có hiển thị FPS và số quả cam đang phát hiện ở góc trên bên trái.
* Nhấn phím `q` để thoát.
* Nếu máy có nhiều webcam: `python app.py --camera --camera-id 1`.

## 6. Test ảnh

```cmd
python app.py --image test.jpg
```

Thêm `--save` nếu muốn lưu ảnh đã vẽ kết quả (lưu cạnh ảnh gốc, tên
`test_result.jpg`):

```cmd
python app.py --image test.jpg --save
```

## 7. Test video

```cmd
python app.py --video test.mp4
```

Xử lý từng frame giống webcam: detect từng quả, classify, vẽ bounding box,
hiển thị liên tục. Nhấn `q` để thoát giữa chừng.

## 8. Chuyển sang Raspberry Pi

1. Copy 4 file/thư mục sau vào thẻ SD (hoặc qua `scp`):
   ```
   app.py
   requirements.txt
   model.tflite
   threshold.json
   ```
2. Trên Raspberry Pi (Raspberry Pi OS, Python 3.9+), cài thư viện — **không
   cài TensorFlow đầy đủ**, chỉ cần:

   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-opencv libatlas-base-dev
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install opencv-python-headless numpy
   pip install tflite-runtime
   ```

   > `tflite-runtime` là gói chính thức của Google cho thiết bị nhúng, nhẹ
   > hơn TensorFlow đầy đủ rất nhiều — phù hợp Raspberry Pi. `app.py` tự
   > phát hiện và ưu tiên dùng `tflite_runtime` nếu có; nếu Pi của bạn
   > không cài được `tflite-runtime` (một số phiên bản Raspberry Pi OS /
   > Python không có wheel sẵn), có thể cài `tensorflow` đầy đủ như phương
   > án dự phòng — chạy được nhưng chậm hơn.

3. Chạy với USB camera hoặc Raspberry Pi Camera (được OpenCV nhận diện như
   webcam thông thường):

   ```bash
   python3 app.py --camera
   ```

   Nếu chạy qua SSH không có màn hình, `cv2.imshow` sẽ báo lỗi vì không có
   giao diện đồ hoạ — cần chạy trực tiếp trên desktop của Pi (có màn hình
   HDMI) hoặc dùng VNC.

4. Test ảnh/video trên Pi giống hệt Windows:

   ```bash
   python3 app.py --image test.jpg
   python3 app.py --video test.mp4
   ```

Không cần Internet khi chạy inference — `model.tflite` chạy hoàn toàn
offline trên máy.

## 9. Danh sách thư viện cần cài trên Raspberry Pi

```
opencv-python-headless
numpy
tflite-runtime
```

(hoặc `tensorflow` đầy đủ thay cho `tflite-runtime` nếu không cài được, xem
lưu ý ở mục 8.)

## 10. Tổng hợp các lệnh chính

| Việc cần làm | Lệnh |
|---|---|
| Train model | `cd training && python main.py run` |
| Export tflite + threshold | `python convert_to_tflite.py` |
| Chạy webcam | `python app.py` hoặc `python app.py --camera` |
| Test ảnh | `python app.py --image test.jpg` |
| Test ảnh và lưu kết quả | `python app.py --image test.jpg --save` |
| Test video | `python app.py --video test.mp4` |
| Đổi webcam khác | `python app.py --camera --camera-id 1` |
| Ghi đè threshold thủ công | `python app.py --threshold 0.4` |
| Thoát chương trình | phím `q` |

## 11. Đã kiểm tra trước khi bàn giao

* `app.py`, `convert_to_tflite.py` và toàn bộ code trong `training/` đã
  compile không lỗi cú pháp/import.
* Đã smoke-test toàn bộ pipeline `app.py` (detection OpenCV → crop → tiền
  xử lý → inference → vẽ bounding box) bằng model MobileNetV3-Small dựng
  đúng kiến trúc gốc, xác nhận: mỗi quả cam trong ảnh có bounding box
  riêng biệt, không bị gộp; cả hai backend `model.tflite` và `model.keras`
  đều load và infer đúng, không lỗi input size/đường dẫn.
* Vì ZIP gốc không kèm trọng số đã train, **chưa thể kiểm chứng độ chính
  xác thực tế của model** — điều này phụ thuộc vào bước train ở mục 4.1
  bạn tự chạy với dataset của bạn.
