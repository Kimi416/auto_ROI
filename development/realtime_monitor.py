#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv8学習リアルタイムモニター
自動更新で進捗を表示
"""

import time
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

def clear_screen():
    """画面をクリア"""
    os.system('clear' if os.name == 'posix' else 'cls')

def display_progress_bar(current, total, bar_length=50):
    """プログレスバーを生成"""
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    return bar, progress

def format_time(seconds):
    """秒を時間形式に変換"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}時間{minutes}分{secs}秒"
    elif minutes > 0:
        return f"{minutes}分{secs}秒"
    else:
        return f"{secs}秒"

def monitor_training():
    """学習をリアルタイムでモニタリング"""
    results_dir = Path('runs/detect/skin_lesion_detection')
    csv_path = results_dir / 'results.csv'
    
    print("🚀 YOLOv8学習リアルタイムモニターを開始します...")
    print("   (Ctrl+Cで終了)")
    time.sleep(2)
    
    start_time = time.time()
    last_epoch = 0
    total_epochs = 50
    
    while True:
        try:
            if csv_path.exists():
                # CSVファイルを読み込み
                df = pd.read_csv(csv_path)
                current_epoch = len(df)
                
                # 画面をクリア
                clear_screen()
                
                # ヘッダー
                print("="*80)
                print("🎯 YOLOv8 皮膚病変検出モデル - リアルタイム学習モニター")
                print("="*80)
                print()
                
                # プログレスバー
                bar, progress = display_progress_bar(current_epoch, total_epochs)
                print(f"進捗: [{bar}] {progress*100:.1f}%")
                print(f"エポック: {current_epoch}/{total_epochs}")
                print()
                
                # 経過時間と推定残り時間
                elapsed_time = time.time() - start_time
                print(f"⏱️  経過時間: {format_time(elapsed_time)}")
                
                if current_epoch > 0:
                    avg_time_per_epoch = elapsed_time / current_epoch
                    remaining_epochs = total_epochs - current_epoch
                    estimated_time = remaining_epochs * avg_time_per_epoch
                    print(f"⏳ 推定残り時間: {format_time(estimated_time)}")
                print()
                
                # 性能指標
                if current_epoch > 0:
                    print("📊 現在の性能指標")
                    print("-"*60)
                    
                    last_row = df.iloc[-1]
                    
                    # 主要指標を表示
                    metrics = [
                        ('mAP50', 'metrics/mAP50(B)', '↑'),
                        ('mAP50-95', 'metrics/mAP50-95(B)', '↑'),
                        ('Precision', 'metrics/precision(B)', '↑'),
                        ('Recall', 'metrics/recall(B)', '↑'),
                        ('Box Loss', 'train/box_loss', '↓'),
                        ('Cls Loss', 'train/cls_loss', '↓')
                    ]
                    
                    for name, col, target in metrics:
                        if col in df.columns:
                            value = last_row[col]
                            # トレンド計算
                            if current_epoch > 1:
                                prev_value = df.iloc[-2][col]
                                if '↑' in target:  # 上昇が良い指標
                                    trend = '📈' if value > prev_value else '📉' if value < prev_value else '➡️'
                                else:  # 下降が良い指標
                                    trend = '📈' if value < prev_value else '📉' if value > prev_value else '➡️'
                            else:
                                trend = ''
                            
                            print(f"{name:15}: {value:8.4f} {trend}")
                    
                    print()
                    
                    # 簡易グラフ（最新10エポック）
                    if 'metrics/mAP50(B)' in df.columns and current_epoch > 1:
                        print("📈 mAP50推移（最新10エポック）")
                        print("-"*60)
                        
                        map_values = df['metrics/mAP50(B)'].values
                        start_idx = max(0, len(map_values) - 10)
                        recent_values = map_values[start_idx:]
                        
                        max_val = max(recent_values) if max(recent_values) > 0 else 1
                        
                        for i, val in enumerate(recent_values):
                            epoch_num = start_idx + i + 1
                            bar_len = int((val / max_val) * 40)
                            bar = '▓' * bar_len + '░' * (40 - bar_len)
                            print(f"Epoch {epoch_num:3d}: |{bar}| {val:.4f}")
                
                # 新しいエポックの通知
                if current_epoch > last_epoch:
                    print()
                    print(f"🔔 エポック {current_epoch} が完了しました！")
                    last_epoch = current_epoch
                
                # 完了チェック
                if current_epoch >= total_epochs:
                    print()
                    print("="*80)
                    print("✅ 学習が完了しました！")
                    print(f"📁 モデル保存先: {results_dir / 'weights/best.pt'}")
                    print("="*80)
                    break
                
                # フッター
                print()
                print("-"*80)
                print(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("Ctrl+Cで終了")
                
            else:
                clear_screen()
                print("⏳ 学習開始を待機中...")
                print(f"   結果ファイル: {csv_path}")
            
            # 5秒ごとに更新
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  モニターを終了します")
            break
        except Exception as e:
            print(f"\nエラー: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_training()