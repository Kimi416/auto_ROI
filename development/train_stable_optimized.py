#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
安定化最適学習スクリプト
batch=8, epochs=30, ハイパーパラメータ整理版
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

def train_stable_optimized():
    print("🎯 安定化最適学習開始")
    print("="*60)
    
    # クラス重みを計算
    class_weights = calculate_class_weights()
    print("📊 計算されたクラス重み:")
    class_names = ['ADM', 'Ephelis', 'Melasma', 'Solar lentigo', 
                   'Nevus', 'Basal cell carcinoma', 'Seborrheic keratosis', 
                   'Malignant melanoma']
    
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weights.get(i, 1.0):.2f}")
    
    # 安定化設定
    config = {
        "モデル": "yolov8m.pt (高精度維持)",
        "バッチサイズ": "8 (安定化)",
        "画像サイズ": "640 (高解像度維持)",
        "エポック": "30 (短縮・実用的)",
        "デバイス": "MPS (Apple Silicon最適化)",
        "保存間隔": "5エポックごと",
        "Early Stopping": "patience=10"
    }
    
    print("\\n⚙️ 安定化設定:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # ハイパーパラメータを辞書でまとめて定義
    hyp = {
        # 学習率関連
        'lr0': 0.001,           # 初期学習率（安定化のため低めに）
        'lrf': 0.01,            # 最終学習率
        'momentum': 0.9,        # モメンタム
        'weight_decay': 0.0005, # 重み減衰
        'warmup_epochs': 3,     # ウォームアップエポック数（短縮）
        'warmup_momentum': 0.8, # ウォームアップモメンタム
        'warmup_bias_lr': 0.1,  # ウォームアップバイアス学習率
        
        # データ拡張（医療画像特化・控えめ）
        'hsv_h': 0.01,          # 色相変化（最小限）
        'hsv_s': 0.3,           # 彩度変化（控えめ）
        'hsv_v': 0.2,           # 明度変化（控えめ）
        'degrees': 2.0,         # 回転角度（最小限）
        'translate': 0.03,      # 平行移動（控えめ）
        'scale': 0.2,           # スケール変化（適度）
        'shear': 1.0,           # シアー変換（最小限）
        'perspective': 0.0001,  # 透視変換（最小限）
        'flipud': 0.0,          # 上下反転なし（医療画像では重要）
        'fliplr': 0.5,          # 左右反転は保持
        'mosaic': 0.5,          # モザイク拡張（控えめ）
        'mixup': 0.0,           # ミックスアップ無効
        'copy_paste': 0.0,      # コピーペースト無効
        
        # 損失関数重み（データ不均衡対応）
        'box': 7.5,             # バウンディングボックス損失重み
        'cls': 1.0,             # クラス分類損失重み
        'dfl': 1.5,             # DFL損失重み
        
        # その他最適化
        'cos_lr': True,         # コサイン学習率スケジュール
        'close_mosaic': 5,      # モザイク拡張終了タイミング（短縮）
    }
    
    print("\\n📋 ハイパーパラメータ:")
    print("  学習率設定:")
    print(f"    lr0: {hyp['lr0']} (初期学習率)")
    print(f"    lrf: {hyp['lrf']} (最終学習率)")
    print(f"    momentum: {hyp['momentum']}")
    print(f"    weight_decay: {hyp['weight_decay']}")
    
    print("  データ拡張:")
    print(f"    色調変化: H={hyp['hsv_h']}, S={hyp['hsv_s']}, V={hyp['hsv_v']}")
    print(f"    幾何変換: rotation={hyp['degrees']}°, scale={hyp['scale']}")
    print(f"    フリップ: 左右={hyp['fliplr']}, 上下={hyp['flipud']}")
    
    print("  損失重み:")
    print(f"    box={hyp['box']}, cls={hyp['cls']}, dfl={hyp['dfl']}")
    
    print("\\n🚀 学習開始...")
    
    try:
        # YOLOv8m で学習実行
        model = YOLO('yolov8m.pt')
        
        # 学習実行（ハイパーパラメータを個別に設定）
        results = model.train(
            data='yolo_dataset/dataset.yaml',
            epochs=30,                      # 短縮・実用的
            imgsz=640,                      # 高解像度維持
            batch=8,                        # 安定化バッチサイズ
            device='mps',                   # Apple Silicon最適化
            workers=2,                      # 安定化
            patience=10,                    # 早期停止緩和（短縮版）
            save=True,
            save_period=5,                  # 5エポックごとに保存
            val=True,
            plots=True,
            verbose=True,
            project='runs/detect',
            name='stable_optimized',
            
            # ハイパーパラメータを個別に設定
            lr0=hyp['lr0'],                 # 初期学習率
            lrf=hyp['lrf'],                 # 最終学習率
            momentum=hyp['momentum'],       # モメンタム
            weight_decay=hyp['weight_decay'], # 重み減衰
            warmup_epochs=hyp['warmup_epochs'], # ウォームアップエポック数
            warmup_momentum=hyp['warmup_momentum'], # ウォームアップモメンタム
            warmup_bias_lr=hyp['warmup_bias_lr'], # ウォームアップバイアス学習率
            
            # データ拡張
            hsv_h=hyp['hsv_h'],             # 色相変化
            hsv_s=hyp['hsv_s'],             # 彩度変化
            hsv_v=hyp['hsv_v'],             # 明度変化
            degrees=hyp['degrees'],         # 回転角度
            translate=hyp['translate'],     # 平行移動
            scale=hyp['scale'],             # スケール変化
            shear=hyp['shear'],             # シアー変換
            perspective=hyp['perspective'], # 透視変換
            flipud=hyp['flipud'],           # 上下反転
            fliplr=hyp['fliplr'],           # 左右反転
            mosaic=hyp['mosaic'],           # モザイク拡張
            mixup=hyp['mixup'],             # ミックスアップ
            copy_paste=hyp['copy_paste'],   # コピーペースト
            
            # 損失関数重み
            box=hyp['box'],                 # バウンディングボックス損失重み
            cls=hyp['cls'],                 # クラス分類損失重み
            dfl=hyp['dfl'],                 # DFL損失重み
            
            # その他最適化
            cos_lr=hyp['cos_lr'],           # コサイン学習率スケジュール
            close_mosaic=hyp['close_mosaic'], # モザイク拡張終了タイミング
            
            # その他設定
            cache=False,                    # メモリ使用量制御
            rect=False,                     # 矩形学習無効（精度重視）
            amp=True,                       # Automatic Mixed Precision
            half=False,                     # 精度重視でfloat32維持
            dnn=False,                      # OpenCV DNN無効
            exist_ok=True,                  # 上書き許可
            resume=False,                   # 新規学習
        )
        
        print("\\n✅ 学習完了!")
        print(f"結果保存先: runs/detect/stable_optimized/")
        
        # 結果分析
        if hasattr(results, 'results_dict'):
            best_metrics = results.results_dict
            print(f"\\n📊 最終結果:")
            if 'metrics/mAP50(B)' in best_metrics:
                print(f"  mAP50: {best_metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in best_metrics:
                print(f"  mAP50-95: {best_metrics['metrics/mAP50-95(B)']:.4f}")
            if 'metrics/precision(B)' in best_metrics:
                print(f"  Precision: {best_metrics['metrics/precision(B)']:.4f}")
            if 'metrics/recall(B)' in best_metrics:
                print(f"  Recall: {best_metrics['metrics/recall(B)']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ エラー発生: {e}")
        print("エラー詳細:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧠 安定化最適YOLOv8学習システム")
    print("batch=8, epochs=30, ハイパーパラメータ整理版")
    print()
    
    success = train_stable_optimized()
    
    if success:
        print("\\n🎉 安定化学習が成功しました！")
        print("📁 結果: runs/detect/stable_optimized/weights/best.pt")
        print("📊 学習曲線: runs/detect/stable_optimized/results.csv")
        print("\\n💡 次のステップ:")
        print("  1. 学習結果の確認")
        print("  2. 推論テストの実行")
        print("  3. 必要に応じて epochs を増やして再学習")
    else:
        print("\\n⚠️ 学習に失敗しました。設定を再調整します。")