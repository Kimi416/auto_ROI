#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
最適化安定学習スクリプト - 最高精度と安定性の両立
段階的アプローチで確実に学習を完走
"""

from ultralytics import YOLO
import torch
import json
import numpy as np
from pathlib import Path
import gc
import time

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

def train_phase(phase_num, model_path, freeze_layers, batch_size, epochs, resume_path=None):
    """段階的学習の各フェーズを実行"""
    print(f"\n{'='*60}")
    print(f"🔥 Phase {phase_num} 開始")
    print(f"{'='*60}")
    
    # メモリクリア
    gc.collect()
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    
    # モデル読み込み
    if resume_path and Path(resume_path).exists():
        print(f"📂 既存モデルから再開: {resume_path}")
        model = YOLO(resume_path)
    else:
        model = YOLO(model_path)
    
    # 学習設定
    print(f"⚙️ Phase {phase_num} 設定:")
    print(f"  - Freezing: {len(freeze_layers)}層")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Device: MPS")
    
    try:
        results = model.train(
            data='yolo_dataset/dataset.yaml',
            epochs=epochs,
            imgsz=640,                      # 高精度維持
            batch=batch_size,
            device='mps',
            workers=2,
            patience=10,
            save=True,
            save_period=3,                  # 頻繁な保存
            val=True,
            plots=True,
            verbose=True,
            project='runs/detect',
            name=f'optimal_stable_phase{phase_num}',
            
            # 学習率（段階的調整）
            lr0=0.001 if phase_num == 1 else 0.0005,
            lrf=0.01,
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=2,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # データ拡張（控えめ・医療画像用）
            hsv_h=0.01,
            hsv_s=0.3,
            hsv_v=0.2,
            degrees=2.0,
            translate=0.03,
            scale=0.2,
            shear=1.0,
            perspective=0.0001,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.5 if phase_num == 1 else 0.3,
            mixup=0.0,
            copy_paste=0.0,
            
            # 損失関数
            box=7.5,
            cls=1.0,
            dfl=1.5,
            
            # 最適化
            cos_lr=True,
            close_mosaic=10,
            
            # メモリ最適化
            cache=False,
            rect=False,
            amp=True,                       # Mixed Precision
            half=False,
            dnn=False,
            
            # その他
            exist_ok=True,
            resume=False,
            
            # Freezing
            freeze=freeze_layers if phase_num < 3 else None,
        )
        
        print(f"✅ Phase {phase_num} 完了!")
        return f'runs/detect/optimal_stable_phase{phase_num}/weights/best.pt'
        
    except Exception as e:
        print(f"❌ Phase {phase_num} でエラー: {e}")
        return None

def train_optimal_stable():
    """最適化安定学習のメイン関数"""
    print("🎯 最適化安定学習システム起動")
    print("最高精度と確実な完走を両立")
    print("="*60)
    
    # クラス重みを計算
    class_weights = calculate_class_weights()
    print("📊 計算されたクラス重み:")
    class_names = ['ADM', 'Ephelis', 'Melasma', 'Solar lentigo', 
                   'Nevus', 'Basal cell carcinoma', 'Seborrheic keratosis', 
                   'Malignant melanoma']
    
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weights.get(i, 1.0):.2f}")
    
    # 段階的学習戦略
    print("\n📋 段階的学習戦略:")
    print("  Phase 1: 基礎学習（Freezing多・安定重視）")
    print("  Phase 2: 中間学習（バランス調整）")
    print("  Phase 3: 最終調整（精度最大化）")
    
    # Phase 1: 基礎学習（最も安定）
    print("\n" + "="*60)
    print("🚀 Phase 1: 基礎学習開始")
    print("目的: 安定した基礎モデル構築")
    
    best_model = train_phase(
        phase_num=1,
        model_path='yolov8m.pt',
        freeze_layers=list(range(10)),      # 10層凍結
        batch_size=8,                       # 安定バッチサイズ
        epochs=20                           # 短期集中
    )
    
    if not best_model:
        print("⚠️ Phase 1失敗。バッチサイズを調整して再試行...")
        best_model = train_phase(
            phase_num=1,
            model_path='yolov8m.pt',
            freeze_layers=list(range(15)),  # より多く凍結
            batch_size=6,                   # さらに小さく
            epochs=15
        )
    
    # Phase 2: 中間学習（バランス）
    if best_model and Path(best_model).exists():
        print("\n" + "="*60)
        print("🚀 Phase 2: 中間学習開始")
        print("目的: 精度向上とバランス調整")
        
        time.sleep(5)  # システム休憩
        gc.collect()
        
        best_model = train_phase(
            phase_num=2,
            model_path=best_model,          # Phase 1の結果を使用
            freeze_layers=list(range(5)),   # 凍結層削減
            batch_size=6,                   # 調整済みバッチ
            epochs=25                       # 追加学習
        )
    
    # Phase 3: 最終調整（精度最大化）
    if best_model and Path(best_model).exists():
        print("\n" + "="*60)
        print("🚀 Phase 3: 最終調整開始")
        print("目的: 精度の最大化")
        
        time.sleep(5)  # システム休憩
        gc.collect()
        
        best_model = train_phase(
            phase_num=3,
            model_path=best_model,          # Phase 2の結果を使用
            freeze_layers=[],               # 全層学習
            batch_size=4,                   # 最小バッチ（精度重視）
            epochs=30                       # 最終調整
        )
    
    # 最終結果
    print("\n" + "="*60)
    print("📊 最終結果サマリー")
    print("="*60)
    
    if best_model and Path(best_model).exists():
        print("✅ 学習成功!")
        print(f"📁 最終モデル: {best_model}")
        
        # 各フェーズの結果を表示
        for phase in range(1, 4):
            results_path = f'runs/detect/optimal_stable_phase{phase}/results.csv'
            if Path(results_path).exists():
                print(f"\n📈 Phase {phase} 結果:")
                # 最後の行を読む
                with open(results_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split(',')
                        if len(last_line) > 8:
                            print(f"  - mAP50: {float(last_line[7]):.4f}")
                            print(f"  - Precision: {float(last_line[5]):.4f}")
                            print(f"  - Recall: {float(last_line[6]):.4f}")
        
        print("\n💡 特徴:")
        print("  ✅ 段階的学習で安定性確保")
        print("  ✅ 最適なバッチサイズ自動調整")
        print("  ✅ 医療画像用最適化")
        print("  ✅ プライバシー保護（ローカル完結）")
        
    else:
        print("❌ 学習に失敗しました")
        print("💡 ヒント: さらに小さいバッチサイズまたはYOLOv8sをお試しください")

if __name__ == "__main__":
    print("🧠 最適化安定YOLOv8学習システム")
    print("最高精度と確実な完走の両立を実現")
    print()
    
    train_optimal_stable()