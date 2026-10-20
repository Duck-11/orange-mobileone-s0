# Dọn project orange_project. Chạy từ thư mục gốc:  powershell -ExecutionPolicy Bypass -File .\tidy_project.ps1
# Chỉ: (1) CHUYỂN file bản cũ vào legacy\mobilenet_NOT_MOBILEONE\  (2) xoá thư mục cache.
# KHÔNG xoá file nào khác, KHÔNG đụng tới môi trường ảo (.venv-app, venv, training\.venv, training\.venv_new).
$ErrorActionPreference = "Stop"
$root = (Get-Location).Path
$dest = Join-Path $root "legacy\mobilenet_NOT_MOBILEONE"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$toMove = @(
  "model.tflite",
  "convert_to_tflite.py",
  "app.py.bak_mobilenet",
  "training\README_TRAINING.md",
  "training\remap_split_paths.py",
  "tools\patch_app.py",
  "training\outputs\models\best_model.keras"
)
foreach ($rel in $toMove) {
  $src = Join-Path $root $rel
  if (Test-Path $src) {
    $target = Join-Path $dest (Split-Path $rel -Leaf)
    if (Test-Path $target) { Write-Host "BỎ QUA (đã có trong legacy): $rel"; continue }
    Move-Item -Path $src -Destination $target
    Write-Host "Đã chuyển: $rel"
  }
}

"Các file trong thư mục này là của bản MobileNetV3-Small cũ / script dùng một lần. KHÔNG phải MobileOne-S0. Không đưa kết quả cũ vào bảng MobileOne-S0." |
  Set-Content -Path (Join-Path $dest "README_LEGACY.txt") -Encoding UTF8

# Xoá cache (bỏ qua mọi thư mục môi trường ảo)
Get-ChildItem -Path $root -Recurse -Directory -Force -ErrorAction SilentlyContinue |
  Where-Object { ($_.Name -in "__pycache__", ".pytest_cache") -and ($_.FullName -notmatch "\\(\.venv|\.venv-app|\.venv_new|venv)\\") } |
  ForEach-Object { Remove-Item -Recurse -Force $_.FullName; Write-Host "Đã xoá cache: $($_.FullName.Substring($root.Length))" }

Write-Host "`nXong. Kiểm tra: python app.py --image <ảnh>  và  pytest tests -v"
