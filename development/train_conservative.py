#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
保守的設定での学習スクリプト
問題を回避するための最小構成
"""

from ultralytics import YOLO
import torch

def train_conservative():
    print("🛡️ 保守的設定での学習開始")
    print("="*50)
    
    # 設定の詳細表示
    settings = {
        "モデル": "yolov8s.pt (軽量版)",
        "バッチサイズ": "4 (大幅減少)",
        "画像サイズ": "416 (縮小)",
        "エポック": "20 (短縮)",
        "デバイス": "CPU (安定)",
        "Workers": "1 (最小)",
        "データ拡張": "無効"
    }
    
    for key, value in settings.items():
        print(f"{key}: {value}")
    
    print("\n🚀 学習開始...")
    
    try:
        # 軽量モデルを使用
        model = YOLO('yolov8s.pt')  # Medium → Small
        
        # 非常に保守的な設定
        results = model.train(
            data='yolo_dataset/dataset.yaml',
            epochs=20,          # 50 → 20
            imgsz=416,          # 640 → 416  
            batch=4,            # 16 → 4
            device='cpu',       # MPS問題を回避
            workers=1,          # 最小値
            patience=100,       # Early stopping無効化
            save=True,
            save_period=5,      # 5エポックごとに保存
            val=True,
            plots=True,
            verbose=True,
            project='runs/detect',
            name='conservative_training',
            # データ拡張を最小化
            hsv_h=0.0,
            hsv_s=0.0, 
            hsv_v=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.0,
            mixup=0.0,
            copy_paste=0.0
        )
        
        print("\n✅ 学習完了!")
        print(f"結果: {results}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        print("さらに設定を軽量化する必要があります")
        return False

if __name__ == "__main__":
    success = train_conservative()
    if success:
        print("\n🎉 保守的設定での学習が成功しました！")
    else:
        print("\n⚠️ 追加の調整が必要です")