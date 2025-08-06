#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv8学習進捗モニター
リアルタイムで学習状況を表示
"""

import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def display_progress():
    """学習進捗を表示"""
    results_dir = Path('runs/detect/skin_lesion_detection')
    csv_path = results_dir / 'results.csv'
    
    if not csv_path.exists():
        print("⏳ 学習結果ファイルを待機中...")
        return None
        
    # CSVファイルを読み込み
    df = pd.read_csv(csv_path)
    current_epoch = len(df)
    total_epochs = 50
    
    # プログレスバー表示
    progress = current_epoch / total_epochs
    bar_length = 50
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    # 画面をクリア（Unix/Mac）
    os.system('clear')
    
    print("="*70)
    print("🎯 YOLOv8 皮膚病変検出モデル学習モニター")
    print("="*70)
    print()
    print(f"進捗: [{bar}] {progress*100:.1f}%")
    print(f"エポック: {current_epoch}/{total_epochs}")
    print()
    
    if current_epoch > 0:
        # 最新の性能指標
        last_row = df.iloc[-1]
        
        print("📊 現在の性能指標")
        print("-"*50)
        
        # 主要な指標を表示
        metrics = {
            'mAP50': 'metrics/mAP50(B)',
            'mAP50-95': 'metrics/mAP50-95(B)', 
            'Precision': 'metrics/precision(B)',
            'Recall': 'metrics/recall(B)',
            'Box Loss': 'train/box_loss',
            'Cls Loss': 'train/cls_loss'
        }
        
        for name, col in metrics.items():
            if col in last_row:
                value = last_row[col]
                if 'Loss' in name:
                    print(f"{name:15}: {value:.4f} ↓")
                else:
                    print(f"{name:15}: {value:.3f} ↑")
        
        print()
        
        # 学習時間の推定
        if current_epoch > 1:
            epochs_remaining = total_epochs - current_epoch
            # 簡易的な時間推定（1エポック約2分と仮定）
            minutes_remaining = epochs_remaining * 2
            hours = minutes_remaining // 60
            minutes = minutes_remaining % 60
            
            print(f"⏱️  推定残り時間: {hours}時間{minutes}分")
    
    print()
    print("💡 ヒント: Ctrl+Cで終了")
    print("="*70)
    
    return current_epoch

def monitor_loop():
    """継続的に進捗を監視"""
    print("🚀 学習モニターを開始します...")
    
    last_epoch = 0
    while True:
        try:
            current_epoch = display_progress()
            
            if current_epoch and current_epoch >= 50:
                print("\n✅ 学習が完了しました！")
                break
                
            # 新しいエポックが完了したら通知
            if current_epoch and current_epoch > last_epoch:
                print(f"\n🔔 エポック {current_epoch} が完了しました")
                last_epoch = current_epoch
                
            time.sleep(10)  # 10秒ごとに更新
            
        except KeyboardInterrupt:
            print("\n\n⚠️ モニターを終了します")
            break
        except Exception as e:
            print(f"エラー: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_loop()