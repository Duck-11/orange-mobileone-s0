# Cấu trúc MobileOne-S0 (đo bằng forward 1x3x224x224)

- timm: `mobileone_s0.apple_in1k`; backbone class: `ByobNet`; MobileOneBlock: 43
- tham số backbone 4,268,272; toàn model 4,269,297

| module | class | params | output shape | blocks | nhánh kxk (block đầu) |
|---|---|---|---|---|---|
| stem | MobileOneBlock | 1,632 | [1, 48, 112, 112] |  |  |
| stages | Sequential | 4,266,640 | [1, 1024, 7, 7] |  |  |
| final_conv | Identity | 0 | [1, 1024, 7, 7] |  |  |
| head | ClassifierHead | 0 | [1, 1024] |  |  |
| stages.0 | Sequential | 24,000 | [1, 48, 56, 56] | 4 | 4 |
| stages.1 | Sequential | 539,472 | [1, 128, 28, 28] | 16 | 4 |
| stages.2 | Sequential | 2,634,368 | [1, 256, 14, 14] | 20 | 4 |
| stages.3 | Sequential | 1,068,800 | [1, 1024, 7, 7] | 2 | 4 |