#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MPS短時間学習 - 早期停止検知付き
"""

import subprocess
import signal
import os
from ultralytics import YOLO
import torch
import gc
import time
from pathlib import Path
import threading

class MPSQuickTrainer:
    def __init__(self):
        self.caffeinate_process = None
        self.stop_flag = False
        self.setup_system()
    
    def setup_system(self):
        """システム最適化"""
        print("🔧 MPS学習システム準備")
        
        # スリープ防止
        try:
            self.caffeinate_process = subprocess.Popen([
                'caffeinate', '-i', '-d', '-s'
            ])
            print("✅ スリープ防止開始")
        except:
            print("⚠️ スリープ防止失敗")
        
        # メモリクリア
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("✅ MPS準備完了")
    
    def monitor_progress(self, results_file):
        """進捗監視スレッド"""
        last_update = time.time()
        
        while not self.stop_flag:
            if Path(results_file).exists():
                current_mtime = os.path.getmtime(results_file)
                time_diff = time.time() - current_mtime
                
                if time_diff > 300:  # 5分更新なし
                    print(f"⚠️ 警告: {int(time_diff//60)}分更新なし")
                    
                if time_diff > 600:  # 10分更新なし
                    print("🚨 停止疑い: 10分以上更新なし")
                    
                if time_diff > 900:  # 15分更新なし
                    print("🔴 停止確定: 学習停止")
                    break
                    
            time.sleep(60)  # 1分ごとにチェック
    
    def cleanup(self):
        """クリーンアップ"""
        self.stop_flag = True
        if self.caffeinate_process:
            self.caffeinate_process.terminate()
            print("🔄 システム設定復元")
    
    def train_mps_quick(self):
        """MPS超短時間学習"""
        print("🚀 MPS短時間学習開始")
        print("="*50)
        
        try:
            # Phase 2ベースモデル
            model_path = 'runs/detect/optimal_stable_phase2/weights/best.pt'
            model = YOLO(model_path)
            
            results_file = 'runs/detect/mps_quick/results.csv'
            
            # 監視スレッド開始
            monitor_thread = threading.Thread(
                target=self.monitor_progress, 
                args=(results_file,)
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            
            print("⚙️ MPS超高速設定:")
            print("- デバイス: MPS")
            print("- エポック: 3 (超短時間)")
            print("- バッチ: 6 (最適サイズ)")
            print("- 監視: リアルタイム")
            
            results = model.train(
                data='lesion_detection.yaml',
                epochs=3,  # 超短時間
                imgsz=640,
                batch=6,   # Phase 2と同じ安定サイズ
                device='mps',
                optimizer='AdamW',
                lr0=0.0008,  # やや高め
                lrf=0.1,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=0,  # ウォームアップなし
                # データ拡張最小
                hsv_h=0.01,
                hsv_s=0.4,
                hsv_v=0.3,
                translate=0.05,
                scale=0.3,
                mosaic=0.3,
                mixup=0.0,
                copy_paste=0.0,
                # 安定性重視
                patience=2,
                save=True,
                save_period=1,  # 毎エポック保存
                val=True,
                plots=True,
                exist_ok=True,
                project='runs/detect',
                name='mps_quick',
                workers=2,  # 軽減
                verbose=True,
                freeze=8  # 軽い凍結
            )
            
            print("✅ MPS学習完了!")
            return 'runs/detect/mps_quick/weights/best.pt'
            
        except Exception as e:
            print(f"❌ MPS学習エラー: {e}")
            return None
        finally:
            self.cleanup()

def main():
    trainer = MPSQuickTrainer()
    
    # 信号ハンドラー
    def signal_handler(sig, frame):
        print("\n🛑 学習中断")
        trainer.cleanup()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("⚡ MPS短時間学習システム")
    print("- 3エポック限定")
    print("- リアルタイム監視")
    print("- 早期停止検知")
    
    result = trainer.train_mps_quick()
    
    if result:
        print(f"🎉 MPS学習成功: {result}")
        
        # 結果表示
        try:
            import pandas as pd
            df = pd.read_csv('runs/detect/mps_quick/results.csv')
            final = df.iloc[-1]
            best_map50 = df['metrics/mAP50(B)'].max()
            
            print(f"\n📊 MPS結果:")
            print(f"最終mAP50: {final['metrics/mAP50(B)']:.4f}")
            print(f"最高mAP50: {best_map50:.4f}")
            print(f"Phase 2比較: {best_map50 - 0.621:+.4f}")
            
        except Exception as e:
            print(f"結果確認エラー: {e}")
    else:
        print("❌ MPS学習失敗")

if __name__ == "__main__":
    main()