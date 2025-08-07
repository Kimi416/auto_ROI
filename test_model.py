#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学習済みモデルの精度確認と検出テスト
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def evaluate_model():
    """モデル評価と精度確認"""
    print("=" * 60)
    print("YOLOv8 病変検出モデル - 精度評価レポート")
    print("=" * 60)
    
    # 最良モデルのパス
    model_path = '/Users/iinuma/Desktop/自動ROI/yolo_dataset/models/lesion_detection_v2_50epochs/weights/best.pt'
    
    # モデル読み込み
    print(f"\n📊 モデル読み込み: {model_path}")
    model = YOLO(model_path)
    
    # データセット設定
    data_yaml = '/Users/iinuma/Desktop/自動ROI/yolo_dataset/dataset.yaml'
    
    # 1. バリデーションセットでの評価
    print("\n" + "=" * 40)
    print("1. バリデーションセット評価（80枚）")
    print("=" * 40)
    
    val_results = model.val(data=data_yaml, split='val')
    
    print(f"\n📈 全体精度:")
    print(f"  - mAP50: {val_results.box.map50:.3f} (50%IoUでの平均精度)")
    print(f"  - mAP50-95: {val_results.box.map:.3f} (50-95%IoUでの平均精度)")
    print(f"  - Precision: {val_results.box.mp:.3f}")
    print(f"  - Recall: {val_results.box.mr:.3f}")
    
    # クラス別の精度
    print("\n📊 病変タイプ別精度 (mAP50):")
    class_names = [
        'Melasma (肝斑)',
        'Solar_lentigo (日光性色素斑)',
        'Nevus (母斑)',
        'ADM (後天性真皮メラノサイトーシス)',
        'Ephelis (雀卵斑)',
        'Seborrheic_keratosis (脂漏性角化症)',
        'Basal_cell_carcinoma (基底細胞癌)',
        'Malignant_melanoma (悪性黒色腫)'
    ]
    
    for i, name in enumerate(class_names):
        if i < len(val_results.box.ap50):
            ap50 = val_results.box.ap50[i]
            print(f"  {i+1}. {name}: {ap50:.3f}")
    
    # 2. テストセットでの評価
    print("\n" + "=" * 40)
    print("2. テストセット評価（40枚）")
    print("=" * 40)
    
    test_results = model.val(data=data_yaml, split='test')
    
    print(f"\n📈 テストセット精度:")
    print(f"  - mAP50: {test_results.box.map50:.3f}")
    print(f"  - mAP50-95: {test_results.box.map:.3f}")
    print(f"  - Precision: {test_results.box.mp:.3f}")
    print(f"  - Recall: {test_results.box.mr:.3f}")
    
    # 3. 実際の画像での検出テスト
    print("\n" + "=" * 40)
    print("3. サンプル画像での検出テスト")
    print("=" * 40)
    
    # テスト画像を選択
    test_images_dir = Path('/Users/iinuma/Desktop/自動ROI/yolo_dataset/test/images')
    test_images = list(test_images_dir.glob('*'))[:5]  # 最初の5枚
    
    if test_images:
        print(f"\n🔍 {len(test_images)}枚のテスト画像で検出実行:")
        
        for idx, img_path in enumerate(test_images, 1):
            print(f"\n画像 {idx}: {img_path.name}")
            
            # 推論実行
            results = model(str(img_path), conf=0.25)
            
            for r in results:
                if len(r.boxes) > 0:
                    print(f"  検出数: {len(r.boxes)}個の病変")
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = class_names[class_id].split('(')[0].strip()
                        print(f"    - {class_name}: 信頼度 {confidence:.2%}")
                else:
                    print(f"  検出なし")
            
            # 結果画像保存
            output_dir = Path('/Users/iinuma/Desktop/自動ROI/detection_results')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f'result_{idx}_{img_path.name}'
            results[0].save(str(output_path))
    
    # 4. モデル情報
    print("\n" + "=" * 40)
    print("4. モデル情報")
    print("=" * 40)
    
    print(f"\n📦 モデル詳細:")
    print(f"  - アーキテクチャ: YOLOv8n (nano)")
    print(f"  - 学習エポック数: 50")
    print(f"  - 学習画像数: 282枚")
    print(f"  - バリデーション画像数: 80枚")
    print(f"  - テスト画像数: 40枚")
    print(f"  - 総パラメータ数: 3,007,208")
    print(f"  - モデルサイズ: 6.2MB")
    
    # 5. 性能サマリー
    print("\n" + "=" * 60)
    print("📊 性能サマリー")
    print("=" * 60)
    
    print(f"\n✅ 主要指標:")
    print(f"  - 平均検出精度 (mAP50): {val_results.box.map50:.1%}")
    print(f"  - 高精度検出病変タイプ:")
    
    for i, name in enumerate(class_names):
        if i < len(val_results.box.ap50):
            ap50 = val_results.box.ap50[i]
            if ap50 > 0.6:
                print(f"    • {name.split('(')[0]}: {ap50:.1%}")
    
    print(f"\n⚠️ 改善が必要な病変タイプ:")
    for i, name in enumerate(class_names):
        if i < len(val_results.box.ap50):
            ap50 = val_results.box.ap50[i]
            if ap50 < 0.3:
                print(f"    • {name.split('(')[0]}: {ap50:.1%}")
    
    # レポート保存
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model_path),
        'validation_metrics': {
            'mAP50': float(val_results.box.map50),
            'mAP50-95': float(val_results.box.map),
            'precision': float(val_results.box.mp),
            'recall': float(val_results.box.mr)
        },
        'test_metrics': {
            'mAP50': float(test_results.box.map50),
            'mAP50-95': float(test_results.box.map),
            'precision': float(test_results.box.mp),
            'recall': float(test_results.box.mr)
        }
    }
    
    report_path = Path('/Users/iinuma/Desktop/自動ROI/evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 評価レポート保存: {report_path}")
    print("\n" + "=" * 60)
    print("評価完了")
    print("=" * 60)

if __name__ == '__main__':
    evaluate_model()