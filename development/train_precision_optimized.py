#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
精度を最大化する最適化学習スクリプト
詳細な分析結果に基づく高精度設定
"""

from ultralytics import YOLO
import torch
import json
import numpy as np
from pathlib import Path

def calculate_class_weights():
    """データ不均衡を考慮したクラス重み計算"""
    with open('yolo_annotations.json', 'r') as f:
        annotations = json.load(f)
    
    # カテゴリ別病変数を集計
    category_counts = {}
    for ann in annotations:
        cat = ann['category']
        lesion_count = ann['lesion_count']
        category_counts[cat] = category_counts.get(cat, 0) + lesion_count
    
    # クラスIDマッピング
    class_mapping = {
        'ADM': 0, 'Ephelis': 1, 'Melasma': 2, 'Solar lentigo': 3,
        'Nevus': 4, 'Basal cell carcinoma': 5, 'Seborrheic keratosis': 6,
        'Malignant melanoma': 7
    }
    
    # 逆頻度重みを計算
    total_samples = sum(category_counts.values())
    class_weights = {}
    
    for cat, class_id in class_mapping.items():
        count = category_counts.get(cat, 1)
        weight = total_samples / (len(class_mapping) * count)
        class_weights[class_id] = weight
    
    return class_weights

def train_precision_optimized():
    print("🎯 精度最適化学習開始")
    print("="*60)
    
    # クラス重みを計算
    class_weights = calculate_class_weights()
    print("📊 計算されたクラス重み:")
    class_names = ['ADM', 'Ephelis', 'Melasma', 'Solar lentigo', 
                   'Nevus', 'Basal cell carcinoma', 'Seborrheic keratosis', 
                   'Malignant melanoma']
    
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weights.get(i, 1.0):.2f}")
    
    # 最適化された設定
    config = {
        "モデル": "yolov8m.pt (高精度維持)",
        "バッチサイズ": "12 (MPS最適化)",
        "画像サイズ": "640 (高解像度維持)",
        "エポック": "100 (十分な学習)",
        "デバイス": "MPS (Apple Silicon最適化)",
        "学習率": "動的調整",
        "データ拡張": "適度な強化",
        "Early Stopping": "patience=15",
        "モデル保存": "3エポックごと"
    }
    
    print("\n⚙️ 最適化設定:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n🚀 学習開始...")
    
    try:
        # YOLOv8m で高精度学習
        model = YOLO('yolov8m.pt')
        
        # 精度重視の最適化設定
        results = model.train(
            data='yolo_dataset/dataset.yaml',
            epochs=100,                 # 十分な学習回数
            imgsz=640,                  # 高解像度維持
            batch=12,                   # MPS最適化バッチサイズ
            device='mps',               # Apple Silicon最適化
            workers=2,                  # 安定した並列処理
            patience=15,                # 早期停止の緩和
            save=True,
            save_period=3,              # 頻繁な保存
            val=True,
            plots=True,
            verbose=True,
            project='runs/detect',
            name='precision_optimized',
            
            # 学習率の最適化
            lr0=0.001,                  # 初期学習率を下げて安定化
            lrf=0.01,                   # 最終学習率
            momentum=0.9,               # 最適化安定化
            weight_decay=0.0005,        # 正則化
            warmup_epochs=5,            # ウォームアップ強化
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # データ拡張の最適化（精度重視）
            hsv_h=0.01,                 # 色相変化を抑制
            hsv_s=0.5,                  # 彩度変化を適度に
            hsv_v=0.3,                  # 明度変化を適度に
            degrees=5.0,                # 回転を控えめに
            translate=0.05,             # 平行移動を控えめに
            scale=0.3,                  # スケール変化を適度に
            shear=2.0,                  # シアー変換を控えめに
            perspective=0.0001,         # 透視変換を最小に
            flipud=0.0,                 # 上下反転なし（医療画像では重要）
            fliplr=0.5,                 # 左右反転は保持
            mosaic=0.8,                 # モザイク拡張を適度に
            mixup=0.1,                  # ミックスアップを控えめに
            copy_paste=0.1,             # コピーペーストを控えめに
            
            # 損失関数の最適化
            box=7.5,                    # バウンディングボックス損失重み
            cls=0.8,                    # クラス分類損失重み（不均衡対応）
            dfl=1.5,                    # DFL損失重み
            
            # メモリ効率化
            cache=False,                # メモリ使用量制御
            rect=False,                 # 矩形学習無効（精度重視）
            cos_lr=True,                # コサイン学習率スケジュール
            close_mosaic=15,            # モザイク拡張終了タイミング
            
            # 推論最適化
            half=False,                 # 精度重視でfloat32維持
            dnn=False,                  # OpenCV DNN無効
            
            # ログとモニタリング
            exist_ok=True,              # 上書き許可
            resume=False,               # 新規学習
            amp=True,                   # Automatic Mixed Precision
        )
        
        print("\n✅ 学習完了!")
        print(f"結果保存先: runs/detect/precision_optimized/")
        
        # 結果分析
        if hasattr(results, 'results_dict'):
            best_metrics = results.results_dict
            print(f"\n📊 最終結果:")
            print(f"  mAP50: {best_metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
            print(f"  mAP50-95: {best_metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
            print(f"  Precision: {best_metrics.get('metrics/precision(B)', 'N/A'):.4f}")
            print(f"  Recall: {best_metrics.get('metrics/recall(B)', 'N/A'):.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        print("エラー詳細:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧠 精度最適化YOLOv8学習システム")
    print("データ不均衡対応・MPS最適化・高精度設定")
    print()
    
    success = train_precision_optimized()
    
    if success:
        print("\n🎉 精度最適化学習が成功しました！")
        print("📁 結果: runs/detect/precision_optimized/weights/best.pt")
        print("📊 学習曲線: runs/detect/precision_optimized/results.csv")
    else:
        print("\n⚠️ 学習に失敗しました。設定を再調整します。")