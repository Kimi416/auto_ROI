#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2 精度向上版
安定したPhase 2基盤で精度を最大化
"""

from ultralytics import YOLO
import torch
import gc
import json
import numpy as np
from pathlib import Path

def analyze_dataset_quality():
    """データセット品質分析"""
    print("📊 データセット品質分析")
    
    with open('yolo_annotations.json', 'r') as f:
        annotations = json.load(f)
    
    # クラス分布分析
    class_counts = {}
    bbox_sizes = []
    
    for ann in annotations:
        category = ann['category']
        class_counts[category] = class_counts.get(category, 0) + ann['lesion_count']
        
        # バウンディングボックスサイズ分析（もしあれば）
        if 'bbox_info' in ann:
            bbox_sizes.extend(ann['bbox_info'])
    
    print("クラス分布:")
    total_samples = sum(class_counts.values())
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_samples) * 100
        print(f"  {cls}: {count}個 ({percentage:.1f}%)")
    
    # 不均衡度計算
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\\nデータ不均衡比: {imbalance_ratio:.1f}:1")
    if imbalance_ratio > 10:
        print("⚠️ 重度の不均衡 - 重み調整必要")
    elif imbalance_ratio > 3:
        print("⚠️ 中度の不均衡 - 軽い重み調整推奨")
    
    return class_counts

def calculate_optimized_class_weights(class_counts):
    """最適化されたクラス重み計算"""
    
    class_mapping = {
        'ADM': 0, 'Ephelis': 1, 'Melasma': 2, 'Solar lentigo': 3,
        'Nevus': 4, 'Basal cell carcinoma': 5, 'Seborrheic keratosis': 6,
        'Malignant melanoma': 7
    }
    
    total_samples = sum(class_counts.values())
    class_weights = {}
    
    # より洗練された重み計算
    for cat, class_id in class_mapping.items():
        count = class_counts.get(cat, 1)
        # sqrt重みで極端な重み付けを避ける
        weight = np.sqrt(total_samples / (len(class_mapping) * count))
        class_weights[class_id] = weight
    
    print("\\n最適化クラス重み:")
    for cat, class_id in class_mapping.items():
        weight = class_weights[class_id]
        print(f"  {cat}: {weight:.3f}")
    
    return class_weights

def train_phase2_precision_boost():
    """Phase 2 精度向上版"""
    print("🚀 Phase 2 精度向上版開始")
    print("="*60)
    
    # データ分析
    class_counts = analyze_dataset_quality()
    class_weights = calculate_optimized_class_weights(class_counts)
    
    # メモリクリア
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # 新しいベースモデル使用（より大きなモデル）
    print("\\n📁 モデル選択: YOLOv8l (大型モデル)")
    model = YOLO('yolov8l.pt')  # largeモデルで精度向上
    
    # デバイス設定
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🖥️ デバイス: {device}")
    
    print("\\n⚙️ 精度最適化設定:")
    print("- モデル: YOLOv8l (高精度)")
    print("- エポック: 30 (十分な学習)")
    print("- バッチサイズ: 4 (精度重視)")
    print("- 画像サイズ: 640 (標準)")
    print("- 最適化器: AdamW")
    print("- 学習率スケジューリング: Cosine")
    
    try:
        results = model.train(
            data='lesion_detection.yaml',
            epochs=30,  # 十分な学習
            imgsz=640,
            batch=4,    # 精度重視の小バッチ
            device=device,
            optimizer='AdamW',
            lr0=0.001,  # 初期学習率
            lrf=0.01,   # 最終学習率比
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,    # ウォームアップ
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            # 損失重み最適化
            box=7.5,    # バウンディングボックス損失
            cls=2.0,    # クラス分類損失重視
            dfl=1.5,    # 分布焦点損失
            # 高精度データ拡張
            hsv_h=0.015,    # 色相
            hsv_s=0.7,      # 彩度
            hsv_v=0.4,      # 明度
            translate=0.1,   # 移動
            scale=0.5,      # スケール
            mosaic=1.0,     # モザイク最大
            mixup=0.1,      # ミックスアップ
            copy_paste=0.1, # コピーペースト
            # 学習制御
            patience=15,    # 早期停止
            save=True,
            save_period=5,  # 5エポックごと保存
            val=True,
            plots=True,
            exist_ok=True,
            project='runs/detect',
            name='phase2_precision_boost',
            workers=4,
            verbose=True,
            # Test Time Augmentation
            augment=True,   # 推論時拡張
            # モデル固有設定
            cos_lr=True,    # Cosine学習率スケジューリング
            close_mosaic=10 # 最後の10エポックはモザイク無効
        )
        
        print("✅ Phase 2 精度向上版完了!")
        return 'runs/detect/phase2_precision_boost/weights/best.pt'
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results():
    """結果比較"""
    try:
        import pandas as pd
        
        print("\\n📊 性能比較")
        print("="*50)
        
        # 元のPhase 2
        original_df = pd.read_csv('runs/detect/optimal_stable_phase2/results.csv')
        original_final = original_df.iloc[-1]
        
        print("Phase 2 元版:")
        print(f"  mAP50: {original_final['metrics/mAP50(B)']:.4f}")
        print(f"  mAP50-95: {original_final['metrics/mAP50-95(B)']:.4f}")
        print(f"  Precision: {original_final['metrics/precision(B)']:.4f}")
        print(f"  Recall: {original_final['metrics/recall(B)']:.4f}")
        
        # 精度向上版
        if Path('runs/detect/phase2_precision_boost/results.csv').exists():
            boost_df = pd.read_csv('runs/detect/phase2_precision_boost/results.csv')
            boost_final = boost_df.iloc[-1]
            
            print("\\nPhase 2 精度向上版:")
            print(f"  mAP50: {boost_final['metrics/mAP50(B)']:.4f}")
            print(f"  mAP50-95: {boost_final['metrics/mAP50-95(B)']:.4f}")
            print(f"  Precision: {boost_final['metrics/precision(B)']:.4f}")
            print(f"  Recall: {boost_final['metrics/recall(B)']:.4f}")
            
            # 改善度
            map50_improvement = boost_final['metrics/mAP50(B)'] - original_final['metrics/mAP50(B)']
            precision_improvement = boost_final['metrics/precision(B)'] - original_final['metrics/precision(B)']
            
            print(f"\\n🎯 改善度:")
            print(f"  mAP50: {map50_improvement:+.4f}")
            print(f"  Precision: {precision_improvement:+.4f}")
            
            if map50_improvement > 0.02:
                print("🎉 大幅改善達成!")
            elif map50_improvement > 0:
                print("✅ 改善確認")
            else:
                print("⚠️ 改善なし")
                
    except Exception as e:
        print(f"比較エラー: {e}")

def main():
    print("🎯 Phase 2 精度向上プログラム")
    print("大型モデル + 最適化設定で精度最大化")
    
    result = train_phase2_precision_boost()
    
    if result:
        print(f"\\n🎉 精度向上版完了: {result}")
        compare_results()
        
        print("\\n💡 さらなる改善案:")
        print("1. アンサンブル学習 (複数モデル組み合わせ)")
        print("2. データ追加収集")
        print("3. 疑似ラベリング")
        print("4. Knowledge Distillation")
        
    else:
        print("❌ 精度向上版失敗")

if __name__ == "__main__":
    main()