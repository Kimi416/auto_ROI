#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 2 継続学習スクリプト
Phase 1の優れた結果をさらに向上
"""

from ultralytics import YOLO
import torch
import gc

print("🚀 Phase 2: 中間学習開始")
print("目的: Phase 1の成果（mAP50: 0.456）をさらに向上")
print("="*60)

# メモリクリア
gc.collect()
torch.mps.empty_cache() if torch.backends.mps.is_available() else None

# Phase 1の最良モデルを読み込み
model = YOLO('runs/detect/optimal_stable_phase1/weights/best.pt')

print("⚙️ Phase 2 設定:")
print("  - Freezing: 5層に削減（より多くの層を学習）")
print("  - Batch Size: 6")
print("  - Epochs: 25")
print("  - 学習率: 0.0005（Phase 1より低め）")

# Phase 2実行
results = model.train(
    data='yolo_dataset/dataset.yaml',
    epochs=25,
    imgsz=640,
    batch=6,
    device='mps',
    workers=2,
    patience=10,
    save=True,
    save_period=5,
    val=True,
    plots=True,
    verbose=True,
    project='runs/detect',
    name='optimal_stable_phase2',
    
    # 学習率（Phase 2用に調整）
    lr0=0.0005,
    lrf=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    warmup_epochs=2,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # データ拡張（医療画像用）
    hsv_h=0.01,
    hsv_s=0.3,
    hsv_v=0.2,
    degrees=2.0,
    translate=0.03,
    scale=0.2,
    shear=1.0,
    perspective=0.0001,
    flipud=0.0,
    fliplr=0.5,
    mosaic=0.3,
    mixup=0.0,
    copy_paste=0.0,
    
    # 損失関数
    box=7.5,
    cls=1.0,
    dfl=1.5,
    
    # 最適化
    cos_lr=True,
    close_mosaic=10,
    
    # メモリ最適化
    cache=False,
    rect=False,
    amp=True,
    half=False,
    dnn=False,
    
    # その他
    exist_ok=True,
    resume=False,
    
    # Freezing（5層のみ）
    freeze=list(range(5)),
)

print("\n✅ Phase 2 完了!")
print("📁 結果: runs/detect/optimal_stable_phase2/weights/best.pt")