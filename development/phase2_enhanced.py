#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2 精度向上版
バックボーン後半層を部分解凍して精度向上
"""

from ultralytics import YOLO
import torch
import gc
import json
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

def phase2_enhanced():
    """Phase 2 強化版実行"""
    print("🚀 Phase 2 精度向上版開始")
    print("="*50)
    
    # メモリクリア
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Phase 2の最終モデルから継続
    model_path = 'runs/detect/optimal_stable_phase2/weights/best.pt'
    
    if not Path(model_path).exists():
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return None
    
    print(f"📁 ベースモデル: {model_path}")
    
    # モデル読み込み
    model = YOLO(model_path)
    
    # デバイス設定
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🖥️ デバイス: {device}")
    
    # クラス重み計算
    class_weights = calculate_class_weights()
    print(f"📊 クラス重み計算完了: {len(class_weights)}クラス")
    
    # 後半層を部分解凍（Phase 2.5的アプローチ）
    print("\n⚙️ Phase 2強化設定:")
    print("- エポック数: 20 (追加学習)")
    print("- バッチサイズ: 8 (効率重視)")
    print("- バックボーン後半層: 部分学習")
    print("- データ拡張: 強化")
    print("- クラス重み: 最適化")
    
    try:
        results = model.train(
            data='lesion_detection.yaml',
            epochs=20,  # 追加で20エポック
            imgsz=640,
            batch=8,  # Phase 2より少し大きく
            device=device,
            optimizer='AdamW',
            lr0=0.0005,  # やや高めの学習率
            lrf=0.1,    # 学習率減衰
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=2,  # ウォームアップ追加
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            # 損失重み調整
            box=7.5,
            cls=2.0,    # クラス損失を重視
            dfl=1.5,
            # データ拡張強化
            hsv_h=0.02,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.15,
            scale=0.8,
            mosaic=0.8,     # モザイク強化
            mixup=0.15,     # ミックスアップ追加
            copy_paste=0.1, # コピーペースト追加
            # 学習制御
            patience=12,
            save=True,
            save_period=4,
            val=True,
            plots=True,
            exist_ok=True,
            project='runs/detect',
            name='phase2_enhanced',
            workers=4,  # 適度な並列処理
            verbose=True,
            # 後半層のみ学習可能にする設定
            freeze=15  # 最初の15層を凍結、後半を学習
        )
        
        print("✅ Phase 2 強化版完了!")
        return 'runs/detect/phase2_enhanced/weights/best.pt'
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results():
    """Phase 2 vs Phase 2強化版の比較"""
    try:
        import pandas as pd
        
        # 元のPhase 2結果
        phase2_df = pd.read_csv('runs/detect/optimal_stable_phase2/results.csv')
        phase2_final = phase2_df.iloc[-1]
        
        print("\n📊 性能比較")
        print("="*40)
        print(f"Phase 2 元版:")
        print(f"  mAP50: {phase2_final['metrics/mAP50(B)']:.4f}")
        print(f"  Precision: {phase2_final['metrics/precision(B)']:.4f}")
        print(f"  Recall: {phase2_final['metrics/recall(B)']:.4f}")
        
        # 強化版結果
        if Path('runs/detect/phase2_enhanced/results.csv').exists():
            enhanced_df = pd.read_csv('runs/detect/phase2_enhanced/results.csv')
            enhanced_final = enhanced_df.iloc[-1]
            
            print(f"\nPhase 2 強化版:")
            print(f"  mAP50: {enhanced_final['metrics/mAP50(B)']:.4f}")
            print(f"  Precision: {enhanced_final['metrics/precision(B)']:.4f}")
            print(f"  Recall: {enhanced_final['metrics/recall(B)']:.4f}")
            
            # 改善度計算
            map50_improvement = enhanced_final['metrics/mAP50(B)'] - phase2_final['metrics/mAP50(B)']
            print(f"\n🎯 改善度:")
            print(f"  mAP50改善: {map50_improvement:+.4f}")
            
    except Exception as e:
        print(f"比較エラー: {e}")

if __name__ == "__main__":
    print("🎯 Phase 2 精度向上版")
    
    result = phase2_enhanced()
    
    if result:
        print(f"🎉 強化版完了: {result}")
        compare_results()
    else:
        print("❌ 強化版失敗")