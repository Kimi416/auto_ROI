# Auto ROI - 自動ROI検出システム

医療画像における皮膚病変の自動検出・抽出システム

## 主要機能

### T字型マスクによる匿名化
- **mask_t_shape.py**: プライバシー保護しながら病変部位は保持
- 眉毛・目元・鼻・口を黒マスクで匿名化（T字型）
- 額・頬・顎の病変検出に重要な部位は保持

### 病変検出・学習
- **train_yolo.py**: YOLOv8による皮膚病変検出モデルの学習
- **detect_lesions.py**: 学習済みモデルによる病変検出
- **lesion_extractor.py**: 検出された病変のくり抜き・抽出

## プロジェクト構成

### 主要ファイル
- `mask_t_shape.py` - T字型顔マスク（メイン機能）
- `mask_t_shape_advanced.py` - 高度なT字型マスク
- `train_yolo.py` - YOLO学習スクリプト（メイン機能）
- `detect_lesions.py` - 病変検出実行
- `lesion_extractor.py` - 病変抽出

### フォルダ構成
```
├── tests/          # テスト用ファイル
│   ├── samples/    # テスト画像・結果
│   ├── input/      # テスト入力データ
│   ├── output/     # テスト出力結果
│   ├── visualizations/  # 可視化結果
│   └── development/ # 開発・実験用コード
├── development/    # 開発中・実験的なスクリプト
├── archive/        # アーカイブ（過去データ、バックアップ）
└── yolo_dataset*/  # YOLO学習用データセット
```

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

### T字型マスク
```bash
# 単一画像のマスク処理
python mask_t_shape.py input_image.jpg -o output_masked.jpg

# プレビューモード（マスク範囲を赤枠で表示）
python mask_t_shape.py input_image.jpg --preview

# ディレクトリ一括処理
python mask_t_shape.py input_dir/ -o output_dir/
```

### YOLO学習・検出
```bash
# モデル学習
python train_yolo.py lesion_detection.yaml

# 病変検出
python detect_lesions.py input_dir/ -o results_dir/
```

## 技術スタック

- Python 3.x
- OpenCV
- PyTorch
- Ultralytics YOLO
- NumPy
- Pathlib