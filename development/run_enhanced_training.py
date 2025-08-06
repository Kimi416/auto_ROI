#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced PAD Lesion Trainer実行スクリプト
バッチサイズと画像数を調整して実行
"""

from enhanced_pad_lesion_trainer import EnhancedPADLesionTrainer

def main():
    print("🎯 Enhanced PAD-UFES-20 Lesion Trainer")
    print("=" * 50)
    
    # 既存のPAD-UFES-20モデルを使用
    base_model = "runs/detect/pad_ufes_20_realistic_20250805_174734/weights/best.pt"
    
    trainer = EnhancedPADLesionTrainer(base_model_path=base_model)
    
    # 強化学習実行（パラメータを調整）
    results = trainer.run_complete_enhancement(
        source_images_dir="organized_advanced_masked",
        confidence_threshold=0.5,  # より高い閾値で高品質な病変のみ使用
        epochs=20  # エポック数を調整
    )
    
    if results:
        print(f"\n✅ 強化学習成功!")
        print(f"📈 精度: {results['accuracy']:.3f}")
        print(f"💾 モデル保存: {results['model_path']}")
    else:
        print("❌ 強化学習に失敗しました")

if __name__ == "__main__":
    main()