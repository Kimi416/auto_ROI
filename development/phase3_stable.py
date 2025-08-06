#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 3 安定版学習スクリプト
"""

from ultralytics import YOLO
import torch
import gc
import time
from pathlib import Path

def train_phase3_stable():
    """Phase 3 安定版実行"""
    print("🔧 Phase 3 安定版開始")
    print("="*50)
    
    # メモリクリア
    gc.collect()
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    
    # Phase 2の最終モデルを使用
    model_path = 'runs/detect/optimal_stable_phase2/weights/best.pt'
    
    if not Path(model_path).exists():
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return None
    
    print(f"📁 使用モデル: {model_path}")
    
    # モデル読み込み
    model = YOLO(model_path)
    
    # デバイス設定
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🖥️ デバイス: {device}")
    
    # 超安定設定
    print("⚙️ 超安定設定:")
    print("- エポック数: 10")
    print("- バッチサイズ: 1")
    print("- 最小学習率")
    print("- データ拡張最小")
    
    try:
        results = model.train(
            data='lesion_detection.yaml',
            epochs=10,  # 最小エポック数
            imgsz=640,
            batch=1,  # 最小バッチサイズ
            device=device,
            optimizer='AdamW',
            lr0=0.0001,  # 安全な学習率
            lrf=0.1,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=0,  # ウォームアップなし
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=1.0,
            dfl=1.5,
            # データ拡張を最小限に
            hsv_h=0.005,
            hsv_s=0.3,
            hsv_v=0.2,
            translate=0.02,
            scale=0.1,
            mosaic=0.0,  # モザイク無効
            mixup=0.0,
            copy_paste=0.0,
            patience=5,  # 早期停止
            save=True,
            save_period=2,  # 頻繁保存
            val=True,
            plots=True,
            exist_ok=True,
            project='runs/detect',
            name='phase3_stable',
            workers=0,  # マルチプロセシング無効
            verbose=True
        )
        
        print("✅ Phase 3 安定版完了!")
        return 'runs/detect/phase3_stable/weights/best.pt'
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 Phase 3 安定版学習開始")
    
    # メモリ確認
    if torch.backends.mps.is_available():
        print("✅ MPS利用可能")
    
    result = train_phase3_stable()
    
    if result:
        print(f"🎉 学習完了: {result}")
    else:
        print("❌ 学習失敗")