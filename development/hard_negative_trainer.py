#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hard Negative Training実行器
2段階学習でFalse Positive削減を行う
"""

import subprocess
import time
from pathlib import Path
import json
from ultralytics import YOLO

class HardNegativeTrainer:
    def __init__(self):
        self.dataset_yaml = "lesion_detection.yaml"
        self.base_model = "yolov8s.pt"  # COCO事前学習モデル
        self.imgsz = 832  # 小病変対応で少し大きめ
        
        # 出力ディレクトリ
        self.output_base = Path("hard_negative_training")
        self.output_base.mkdir(exist_ok=True)
        
        print(f"🎯 Hard Negative Training セットアップ完了")
        print(f"📊 データセット: {self.dataset_yaml}")
        print(f"🏗️  ベースモデル: {self.base_model}")
        print(f"📐 画像サイズ: {self.imgsz}")
    
    def phase_a_frozen_training(self):
        """フェーズA: 凍結学習（backbone慣らし）"""
        print(f"\n🧊 フェーズA: 凍結学習開始")
        print("=" * 60)
        print("backboneの一部を凍結して誤検出パターンを安定学習")
        
        # YOLOコマンド構築
        cmd = [
            "yolo", "detect", "train",
            f"model={self.base_model}",
            f"data={self.dataset_yaml}",
            f"imgsz={self.imgsz}",
            "epochs=10",
            "batch=16",
            "freeze=10",        # backbone上位10層を凍結
            "mosaic=0.1",       # mosaicを弱める（背景バイアス防止）
            "mixup=0",          # mixupをOFF
            "lr0=0.001",        # 初期学習率
            "patience=50",      # 早期停止を緩める
            "save_period=5",    # 5エポックごとに保存
            f"project={self.output_base}",
            "name=phase_a_frozen",
            "exist_ok=True"
        ]
        
        print(f"🚀 実行コマンド:")
        print(f"  {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                elapsed = time.time() - start_time
                print(f"✅ フェーズA完了 ({elapsed/60:.1f}分)")
                
                # 最適重みのパスを返す
                phase_a_weights = self.output_base / "phase_a_frozen" / "weights" / "best.pt"
                return str(phase_a_weights)
            else:
                print(f"❌ フェーズA失敗:")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ フェーズA タイムアウト（60分）")
            return None
        except Exception as e:
            print(f"❌ フェーズA エラー: {e}")
            return None
    
    def phase_b_unfrozen_training(self, phase_a_weights):
        """フェーズB: 解凍学習（仕上げ）"""
        print(f"\n🔥 フェーズB: 解凍学習開始")
        print("=" * 60)
        print("全層解凍でFalse Positive削減を仕上げ")
        
        cmd = [
            "yolo", "detect", "train",
            f"model={phase_a_weights}",  # フェーズAの結果を使用
            f"data={self.dataset_yaml}",
            f"imgsz={self.imgsz}",
            "epochs=30",
            "batch=16", 
            "freeze=0",         # 全層解凍
            "mosaic=0.1",       # 引き続き弱いmosaic
            "mixup=0",
            "lr0=0.0005",       # より低い学習率
            "patience=50",
            "save_period=10",
            f"project={self.output_base}",
            "name=phase_b_unfrozen", 
            "exist_ok=True"
        ]
        
        print(f"🚀 実行コマンド:")
        print(f"  {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)  # 90分
            
            if result.returncode == 0:
                elapsed = time.time() - start_time
                print(f"✅ フェーズB完了 ({elapsed/60:.1f}分)")
                
                # 最終重みのパスを返す
                final_weights = self.output_base / "phase_b_unfrozen" / "weights" / "best.pt"
                return str(final_weights)
            else:
                print(f"❌ フェーズB失敗:")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ フェーズB タイムアウト（90分）")
            return None
        except Exception as e:
            print(f"❌ フェーズB エラー: {e}")
            return None
    
    def validate_model(self, model_weights, phase_name):
        """モデルの性能検証"""
        print(f"\n📊 {phase_name} 性能検証")
        print("-" * 40)
        
        cmd = [
            "yolo", "detect", "val",
            f"model={model_weights}",
            f"data={self.dataset_yaml}",
            f"imgsz={self.imgsz}",
            "plots=True",
            "save_json=True",
            f"project={self.output_base}",
            f"name={phase_name}_validation",
            "exist_ok=True"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                print(f"✅ {phase_name} 検証完了")
                
                # 結果ファイルのパスを返す
                val_results = self.output_base / f"{phase_name}_validation"
                return str(val_results)
            else:
                print(f"❌ {phase_name} 検証失敗:")
                print(f"stderr: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ {phase_name} 検証エラー: {e}")
            return None
    
    def test_on_problem_images(self, model_weights):
        """問題画像での改善テスト"""
        print(f"\n🎯 問題画像での改善テスト")
        print("-" * 40)
        
        test_images = [
            "test_preview.jpg",
            "test1.jpeg",
            "test2.jpeg"
        ]
        
        model = YOLO(model_weights)
        improvements = {}
        
        for img_path in test_images:
            if not Path(img_path).exists():
                continue
                
            print(f"🔍 {img_path} をテスト...")
            
            # 複数の信頼度で検出数をカウント
            detection_counts = {}
            for conf in [0.01, 0.05, 0.1, 0.2, 0.3]:
                results = model(img_path, conf=conf, verbose=False)
                count = 0
                for result in results:
                    if result.boxes is not None:
                        count = len(result.boxes)
                detection_counts[conf] = count
            
            improvements[img_path] = detection_counts
            print(f"  検出数（信頼度別）: {detection_counts}")
        
        return improvements
    
    def generate_training_report(self, final_weights, improvements):
        """訓練レポート生成"""
        report = {
            'timestamp': str(time.time()),
            'training_config': {
                'base_model': self.base_model,
                'dataset': self.dataset_yaml,
                'image_size': self.imgsz,
                'phase_a_epochs': 10,
                'phase_b_epochs': 30
            },
            'final_model': final_weights,
            'test_improvements': improvements,
            'recommendations': []
        }
        
        # 改善分析
        for img_path, counts in improvements.items():
            if counts.get(0.01, 0) < 2:  # 超低信頼度での検出が2個未満
                report['recommendations'].append(f"{img_path}: 誤検出が大幅改善")
            elif counts.get(0.1, 0) == 0:  # 0.1で検出なし
                report['recommendations'].append(f"{img_path}: 適切な検出なし、追加改善が必要")
        
        # レポート保存
        report_path = self.output_base / "hard_negative_training_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 訓練レポート保存: {report_path}")
        return report

def main():
    trainer = HardNegativeTrainer()
    
    print("🚀 Hard Negative Training 開始")
    print("=" * 80)
    
    # Phase A: 凍結学習
    phase_a_weights = trainer.phase_a_frozen_training()
    if not phase_a_weights or not Path(phase_a_weights).exists():
        print("❌ フェーズA失敗のため中断")
        return
    
    # Phase A検証
    trainer.validate_model(phase_a_weights, "phase_a")
    
    # Phase B: 解凍学習
    final_weights = trainer.phase_b_unfrozen_training(phase_a_weights)
    if not final_weights or not Path(final_weights).exists():
        print("❌ フェーズB失敗")
        print("🔄 フェーズAの結果を使用して継続")
        final_weights = phase_a_weights
    
    # 最終検証
    trainer.validate_model(final_weights, "final")
    
    # 改善テスト
    improvements = trainer.test_on_problem_images(final_weights)
    
    # レポート生成
    report = trainer.generate_training_report(final_weights, improvements)
    
    print(f"\n🎉 Hard Negative Training 完了!")
    print(f"📊 最終モデル: {final_weights}")
    print(f"💡 改善状況:")
    for img, counts in improvements.items():
        lowest_detection = counts.get(0.01, 0)
        print(f"  {img}: {lowest_detection}個検出（信頼度0.01）")
    
    print(f"\n🔄 次のステップ:")
    print(f"  1. test_preview.jpgで最終テスト実行")
    print(f"  2. python3 test_improved_model.py {final_weights}")

if __name__ == "__main__":
    main()