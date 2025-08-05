# Auto ROI - 自動ROI検出システム

医療画像における皮膚病変の自動検出・抽出システム

## 主要機能

### T字型マスクによる匿名化
- **mask.py**: プライバシー保護しながら病変部位は保持
- 眉毛・目元・鼻・口を黒マスクで匿名化
- 額・頬・顎の病変検出に重要な部位は保持

### 病変検出
- YOLO技術による高精度な皮膚病変検出
- 複数の病変タイプに対応

## 対応病変

- ADM (後天性真皮メラノサイトーシス)
- Basal cell carcinoma (基底細胞癌)  
- Ephelis (そばかす)
- Malignant melanoma (悪性黒色腫)
- Melasma (肝斑)
- Nevus (色素性母斑)
- Seborrheic keratosis (脂漏性角化症)
- Solar lentigo (日光性色素斑)

## 使用方法

```bash
python mask.py input_image.jpg -o output_masked.jpg
```

## 技術スタック

- Python 3.x
- OpenCV
- dlib
- PyTorch
- YOLO

---
🤖 Generated with Claude Code