#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
安定学習システム - 停止問題を根本解決
"""

import subprocess
import signal
import os
from ultralytics import YOLO
import torch
import gc
import time
from pathlib import Path

class StableTrainer:
    def __init__(self):
        self.caffeinate_process = None
        self.setup_system()
    
    def setup_system(self):
        """システム最適化設定"""
        print("🔧 システム最適化開始")
        
        # 1. スリープ防止 (caffeinate)
        try:
            self.caffeinate_process = subprocess.Popen([
                'caffeinate', '-i', '-d', '-s'
            ])
            print("✅ スリープ防止開始")
        except:
            print("⚠️ スリープ防止失敗")
        
        # 2. プロセス優先度上昇
        try:
            os.nice(-5)  # 優先度上昇
            print("✅ プロセス優先度上昇")
        except:
            print("⚠️ 優先度変更失敗")
        
        # 3. メモリ最適化
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("✅ メモリクリア完了")
    
    def cleanup(self):
        """クリーンアップ"""
        if self.caffeinate_process:
            self.caffeinate_process.terminate()
            print("🔄 スリープ防止終了")
    
    def train_robust(self):
        """超安定学習実行"""
        print("🚀 超安定学習開始")
        print("="*50)
        
        try:
            # Phase 2ベースモデル
            model_path = 'runs/detect/optimal_stable_phase2/weights/best.pt'
            model = YOLO(model_path)
            
            # 超保守的設定
            print("⚙️ 超安定設定:")
            print("- デバイス: CPU (MPS問題回避)")
            print("- エポック: 5 (短時間)")
            print("- バッチ: 4")
            print("- workers: 1")
            print("- 頻繁保存: 毎エポック")
            
            results = model.train(
                data='lesion_detection.yaml',
                epochs=5,  # 短時間で確実
                imgsz=640,
                batch=4,
                device='cpu',  # CPU使用でMPS問題回避
                optimizer='AdamW',
                lr0=0.001,  # やや高い学習率
                lrf=0.1,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=1,
                # データ拡張軽減
                hsv_h=0.01,
                hsv_s=0.5,
                hsv_v=0.3,
                translate=0.1,
                scale=0.5,
                mosaic=0.5,
                mixup=0.0,
                copy_paste=0.0,
                # 安定性設定
                patience=3,
                save=True,
                save_period=1,  # 毎エポック保存
                val=True,
                plots=True,
                exist_ok=True,
                project='runs/detect',
                name='stable_training',
                workers=1,  # シングルワーカー
                verbose=True,
                freeze=10  # 部分凍結
            )
            
            print("✅ 安定学習完了!")
            return 'runs/detect/stable_training/weights/best.pt'
            
        except Exception as e:
            print(f"❌ エラー: {e}")
            return None
        finally:
            self.cleanup()

def main():
    trainer = StableTrainer()
    
    # 信号ハンドラー設定
    def signal_handler(sig, frame):
        print("\\n🛑 学習中断 - クリーンアップ中...")
        trainer.cleanup()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    result = trainer.train_robust()
    
    if result:
        print(f"🎉 学習成功: {result}")
        
        # 結果比較
        try:
            import pandas as pd
            df = pd.read_csv('runs/detect/stable_training/results.csv')
            final = df.iloc[-1]
            print(f"最終mAP50: {final['metrics/mAP50(B)']:.4f}")
        except:
            print("結果確認失敗")
    else:
        print("❌ 学習失敗")

if __name__ == "__main__":
    main()