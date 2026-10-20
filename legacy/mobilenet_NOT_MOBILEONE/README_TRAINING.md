# Orange MobileNetV3Small Clean

Project ngắn gọn để huấn luyện và đánh giá **một backbone MobileNetV3Small** cho bài toán:

```text
Ảnh quả cam → GOOD hoặc BAD
```

Project tập trung đúng giai đoạn hiện tại của nhóm: bốn thành viên huấn luyện bốn backbone trên cùng dữ liệu, đánh giá bằng cùng tập test và ghép kết quả để chọn model tốt nhất.

## 1. Project chỉ có các phần cần thiết

```text
orange_mobilenetv3_clean/
├── main.py
├── config.yaml
├── requirements.txt
├── README.md
├── orange/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   └── evaluation.py
└── outputs/
```

Đã bỏ các phần chưa cần ở giai đoạn so sánh backbone:

- INT8 và hiệu chỉnh threshold riêng cho INT8.
- Benchmark Raspberry Pi.
- Camera và servo.
- pHash, Union-Find và gom ảnh gần giống.
- Nhiều script chạy riêng lẻ.
- Các module báo cáo trùng chức năng.

## 2. Model được gọi trực tiếp

Trong `orange/model.py`, model được tạo trực tiếp bằng:

```python
backbone = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    include_preprocessing=True,
)
```

Sau backbone là Global Average Pooling, Dropout và một neuron sigmoid. Đầu ra là xác suất BAD:

```text
GOOD = 0
BAD  = 1
```

## 3. Cấu trúc dataset gốc

Ví dụ đường dẫn trong `config.yaml` là:

```yaml
data:
  source_dir: "E:/dataset"
```

Thư mục phải có dạng:

```text
E:/dataset/
├── fresh_orange/
├── blackspotted_orange/
├── cankered_orange/
└── rotten_orange/
```

Mapping:

```text
fresh_orange        → GOOD
blackspotted_orange → BAD
cankered_orange     → BAD
rotten_orange       → BAD
```

Chương trình tự chia:

```text
Train      70%
Validation 15%
Test       15%
```

Việc chia có seed cố định và phân tầng theo bốn lớp gốc để các tập giữ phân bố tương đối giống nhau.

## 4. Cài đặt trên Windows

Dùng Python 3.11.

Mở CMD hoặc Terminal của VS Code trong thư mục project:

```cmd
py -3.11 -m venv .venv
```

Kích hoạt:

```cmd
.venv\Scripts\activate
```

Cài thư viện:

```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Chạy toàn bộ bằng một lệnh

Sửa đường dẫn trong `config.yaml`, sau đó chạy:

```cmd
python main.py run
```

Lệnh này tự thực hiện:

1. Quét và kiểm tra ảnh.
2. Chia train/validation/test.
3. Tính class weight.
4. Gọi trực tiếp MobileNetV3Small pretrained ImageNet.
5. Train classifier head.
6. Fine-tune các tầng cuối.
7. Chọn threshold trên validation.
8. Đánh giá internal test.
9. Đánh giá tất cả external dataset đã khai báo.
10. Xuất báo cáo và bảng tổng hợp.

## 6. Các lệnh riêng

Chỉ chuẩn bị và chia dữ liệu:

```cmd
python main.py prepare
```

Train bằng các split đã tạo:

```cmd
python main.py train
```

Đánh giá đồng thời tất cả dataset ngoài:

```cmd
python main.py evaluate-all
```

Dự đoán nhanh một ảnh:

```cmd
python main.py predict --image "E:/anh_test/cam_01.jpg"
```

Các lệnh trên chạy giống nhau trong CMD, PowerShell và Terminal của VS Code. Trong PowerShell, kích hoạt bằng:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 7. Chạy nhiều dataset cùng lúc

Thêm nhiều mục vào `external_datasets` trong `config.yaml`:

```yaml
external_datasets:
  - name: "dataset_kaggle_1"
    path: "E:/orange_dataset_1/test"
    class_mapping:
      fresh: GOOD
      blackspot: BAD
      canker: BAD
      grenning: BAD

  - name: "dataset_camera_lab"
    path: "E:/orange_dataset_2"
    class_mapping:
      GOOD: GOOD
      BAD: BAD
```

Sau đó chỉ chạy một lần:

```cmd
python main.py evaluate-all
```

`path` có thể trỏ:

- Thẳng tới thư mục chứa các lớp.
- Hoặc tới thư mục cha có thư mục `test`; chương trình tự tìm `test`.

External dataset **không được dùng để chọn threshold hoặc train**. Chương trình dùng threshold đã chọn từ validation của dataset gốc.

## 8. Kết quả đầu ra

Model:

```text
outputs/models/best_model.keras
outputs/models/final_model.keras
```

Dữ liệu đã chia:

```text
outputs/splits/train.csv
outputs/splits/validation.csv
outputs/splits/test.csv
outputs/splits/dataset_manifest.csv
outputs/splits/invalid_images.csv
```

Ngưỡng được chọn trên validation:

```text
outputs/selected_threshold.json
```

Biểu đồ train:

```text
outputs/plots/loss_curve.png
outputs/plots/accuracy_curve.png
outputs/plots/auc_curve.png
```

Mỗi dataset có thư mục báo cáo riêng:

```text
outputs/reports/internal_test/
outputs/reports/orange_disease_external/
```

Mỗi thư mục có:

```text
metrics.json
predictions.csv
per_original_class.csv
classification_report.txt
confusion_matrix.png
normalized_confusion_matrix.png
roc_curve.png
precision_recall_curve.png
```

Bảng quan trọng nhất để ghép kết quả bốn backbone:

```text
outputs/backbone_result.csv
```

Các cột gồm:

```text
backbone
dataset
accuracy
precision_bad
recall_bad
f1_bad
macro_f1
weighted_f1
roc_auc
good_predicted_bad
bad_predicted_good
threshold
parameter_count
keras_model_size_mb
average_inference_ms_per_image
```

## 9. Chọn model tốt nhất trong nhóm

Cả bốn người phải dùng chung:

- `train.csv`
- `validation.csv`
- `test.csv`
- Kích thước ảnh 224 × 224.
- GOOD = 0 và BAD = 1.
- Cùng external datasets.

Khi ghép bảng, ưu tiên xem đồng thời:

1. Recall BAD: giảm cam hỏng bị lọt sang ô GOOD.
2. F1 BAD: cân bằng Precision và Recall của lớp BAD.
3. Macro F1: tránh model chỉ tốt với lớp đông hơn.
4. Accuracy.
5. `bad_predicted_good`: số cam BAD bị đoán nhầm thành GOOD.
6. Thời gian suy luận, số tham số và dung lượng model.

Không chọn model chỉ dựa vào Recall BAD. Một model dự đoán mọi ảnh là BAD có thể có Recall BAD cao nhưng phân loại thực tế rất kém.

## 10. Lưu ý về chất lượng

Code có thể chạy đúng quy trình nhưng không thể bảo đảm mọi ảnh ngoài thực tế đều được dự đoán đúng. Chất lượng phụ thuộc vào:

- Độ chính xác của nhãn dataset.
- Sự đa dạng góc chụp, nền và ánh sáng.
- Dataset mới có giống điều kiện camera thật hay không.
- Cam có nằm rõ và đủ lớn trong ảnh hay không.

Sau khi nhóm chọn được backbone tốt nhất, bước kế tiếp mới là thu thập ảnh từ camera và dây chuyền thật để fine-tune model chiến thắng.
