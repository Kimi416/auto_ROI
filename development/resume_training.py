#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学習再開スクリプト - Phase 2の続きとPhase 3実行
"""

from ultralytics import YOLO
import torch
import json
import gc
import time
from pathlib import Path

def calculate_class_weights():
    """データ不均衡を考慮したクラス重み計算"""
    with open('yolo_annotations.json', 'r') as f:
        annotations = json.load(f)
    
    category_counts = {}
    for ann in annotations:
        cat = ann['category']
        lesion_count = ann['lesion_count']
        category_counts[cat] = category_counts.get(cat, 0) + lesion_count
    
    class_mapping = {
        'ADM': 0, 'Ephelis': 1, 'Melasma': 2, 'Solar lentigo': 3,
        'Nevus': 4, 'Basal cell carcinoma': 5, 'Seborrheic keratosis': 6,
        'Malignant melanoma': 7
    }
    
    total_samples = sum(category_counts.values())
    class_weights = {}
    
    for cat, class_id in class_mapping.items():
        count = category_counts.get(cat, 1)
        weight = total_samples / (len(class_mapping) * count)
        class_weights[class_id] = weight
    
    return class_weights

def resume_phase2():
    """Phase 2の続きを実行（残り12エポック）"""
    print("="*60)
    print("🔄 Phase 2 再開（エポック14から25まで）")
    print("="*60)
    
    # 最後のチェックポイントから再開
    model = YOLO('runs/detect/optimal_stable_phase2/weights/last.pt')
    
    # デバイス設定
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🖥️ デバイス: {device}")
    
    # クラス重み
    class_weights = calculate_class_weights()
    
    # 残りのエポック数を計算（全25エポック中、13エポック完了）
    remaining_epochs = 25 - 13
    
    print(f"📊 残りエポック数: {remaining_epochs}")
    print("🚀 学習再開...")
    
    try:
        results = model.train(
            data='lesion_detection.yaml',
            epochs=remaining_epochs,
            imgsz=640,
            batch=6,
            device=device,
            optimizer='AdamW',
            lr0=0.0001,  # 低い学習率で継続
            lrf=0.01,
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=0,  # ウォームアップ不要
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=1.0,  # クラス重みは後で調整
            dfl=1.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            mosaic=0.5,
            mixup=0.0,
            copy_paste=0.0,
            patience=10,
            save=True,
            save_period=5,
            val=True,
            plots=True,
            exist_ok=True,
            project='runs/detect',
            name='optimal_stable_phase2_resumed',
            resume=True  # チェックポイントから再開
        )
        
        print("✅ Phase 2 再開分完了!")
        return 'runs/detect/optimal_stable_phase2_resumed/weights/best.pt'
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        return None

def train_phase3(model_path):
    """Phase 3: 軽量最適化（安定版）"""
    print("\n" + "="*60)
    print("🚀 Phase 3: 軽量最適化開始")
    print("="*60)
    
    # モデル読み込み
    model = YOLO(model_path)
    
    # デバイス設定
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # クラス重み
    class_weights = calculate_class_weights()
    
    print("📊 軽量設定:")
    print("- エポック数: 15 (短縮)")
    print("- バッチサイズ: 2 (メモリ節約)")
    print("- 安定学習")
    print("- デバイス:", device)
    
    try:
        results = model.train(
            data='lesion_detection.yaml',
            epochs=15,  # エポック数を半分に
            imgsz=640,
            batch=2,  # バッチサイズを2に削減
            device=device,
            optimizer='AdamW',
            lr0=0.00005,  # やや高めの学習率で効率化
            lrf=0.01,
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=1,  # ウォームアップ短縮
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=1.0,
            dfl=1.5,
            hsv_h=0.01,  # データ拡張を軽減
            hsv_s=0.5,
            hsv_v=0.3,
            translate=0.05,
            scale=0.3,
            mosaic=0.2,
            mixup=0.0,
            copy_paste=0.0,
            patience=8,  # 早期停止を短縮
            save=True,
            save_period=3,  # 保存頻度を上げる
            val=True,
            plots=True,
            project='runs/detect',
            name='optimal_stable_phase3_light'
        )
        
        print("✅ Phase 3 軽量版完了!")
        return 'runs/detect/optimal_stable_phase3_light/weights/best.pt'
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        return None

def main():
    print("🤖 YOLO学習再開プログラム")
    print("="*60)
    
    # メモリクリア
    gc.collect()
    
    # Phase 2は既に完了しているので、Phase 3から開始
    print("\n📌 Phase 2は完了済み。Phase 3を開始します...")
    
    # Phase 2の最終モデルパスを使用
    phase2_model = 'runs/detect/optimal_stable_phase2/weights/best.pt'
    
    if Path(phase2_model).exists():
        print(f"✅ Phase 2モデル確認: {phase2_model}")
        
        # 休憩
        print("\n⏸️ 5秒間システム休憩...")
        time.sleep(5)
        gc.collect()
        
        # Phase 3実行
        phase3_model = train_phase3(phase2_model)
        
        if phase3_model:
            print("\n" + "="*60)
            print("🎉 全フェーズ完了!")
            print(f"📁 最終モデル: {phase3_model}")
            print("="*60)
        else:
            print("\n⚠️ Phase 3で問題が発生しました")
    else:
        print(f"\n⚠️ Phase 2モデルが見つかりません: {phase2_model}")

if __name__ == "__main__":
    main()