#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学習完了チェッカー
"""

import pandas as pd
from pathlib import Path
import time

def check_completion():
    csv_path = Path('runs/detect/skin_lesion_detection/results.csv')
    
    if not csv_path.exists():
        return False, 0
    
    df = pd.read_csv(csv_path)
    current_epoch = len(df)
    
    if current_epoch >= 50:
        return True, current_epoch
    else:
        return False, current_epoch

if __name__ == "__main__":
    is_complete, epochs = check_completion()
    
    if is_complete:
        print("✅ 学習が完了しました！")
        print(f"総エポック数: {epochs}")
        
        # 最終性能を表示
        df = pd.read_csv('runs/detect/skin_lesion_detection/results.csv')
        last_row = df.iloc[-1]
        
        if 'metrics/mAP50(B)' in df.columns:
            final_map50 = last_row['metrics/mAP50(B)']
            best_map50 = df['metrics/mAP50(B)'].max()
            best_epoch = df['metrics/mAP50(B)'].idxmax() + 1
            
            print(f"\n📊 最終性能:")
            print(f"最終mAP50: {final_map50:.3f}")
            print(f"最高mAP50: {best_map50:.3f} (エポック {best_epoch})")
            
        print(f"\n📁 モデルファイル:")
        print(f"最良モデル: runs/detect/skin_lesion_detection/weights/best.pt")
        print(f"最終モデル: runs/detect/skin_lesion_detection/weights/last.pt")
    else:
        print(f"⏳ 学習中... (エポック {epochs}/50)")