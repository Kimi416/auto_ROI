#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
安定した学習実行スクリプト
"""

from ultralytics import YOLO
import time
import signal
import sys
from pathlib import Path

def signal_handler(sig, frame):
    print('\n学習を安全に終了しています...')
    sys.exit(0)

def train_stable():
    print("🚀 安定した学習を開始します...")
    
    # シグナルハンドラを設定
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # YOLOモデルを初期化
        model = YOLO('yolov8m.pt')
        
        # 学習実行
        results = model.train(
            data='yolo_dataset/dataset.yaml',
            epochs=50,
            imgsz=640,
            batch=16,
            patience=10,  # Early stopping
            save=True,
            save_period=10,  # 10エポックごとに保存
            val=True,
            plots=True,
            verbose=True,
            project='runs/detect',
            name='skin_lesion_stable'
        )
        
        print("✅ 学習が正常に完了しました！")
        print(f"結果: {results}")
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによって中断されました")
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("学習を再試行してください")

if __name__ == "__main__":
    train_stable()