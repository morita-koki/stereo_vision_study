#!/usr/bin/env python3
"""
Configuration Validator

設定ファイルの妥当性を検証するユーティリティ
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the package to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from stereo_vision.feature_extractor import get_available_extractors
from stereo_vision.feature_extractor.config import load_config, FeatureExtractorConfig


class ConfigValidator:
    """設定ファイルの検証クラス"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate(self, config: FeatureExtractorConfig) -> bool:
        """
        設定の総合的な検証
        
        Args:
            config: 検証する設定
            
        Returns:
            bool: 検証結果（True = 正常）
        """
        self.errors.clear()
        self.warnings.clear()
        
        # 基本設定の検証
        self._validate_basic_config(config)
        
        # 抽出器固有設定の検証
        self._validate_extractor_config(config)
        
        # ディレクトリ・ファイルの検証
        self._validate_paths(config)
        
        # 可視化設定の検証
        self._validate_visualization_config(config)
        
        return len(self.errors) == 0
    
    def _validate_basic_config(self, config: FeatureExtractorConfig):
        """基本設定の検証"""
        # 抽出器タイプの確認
        available_extractors = get_available_extractors()
        if config.extractor_type not in available_extractors:
            self.errors.append(
                f"Unknown extractor type: '{config.extractor_type}'. "
                f"Available: {available_extractors}"
            )
        
        # max_keypointsの確認
        if config.max_keypoints is not None:
            if config.max_keypoints <= 0:
                self.errors.append("max_keypoints must be positive")
            elif config.max_keypoints > 10000:
                self.warnings.append(
                    f"max_keypoints ({config.max_keypoints}) is very large, "
                    "which may cause memory issues"
                )
        
        # min_responseの確認
        if config.min_response is not None:
            if config.min_response < 0:
                self.errors.append("min_response must be non-negative")
        
        # filter_regionの確認
        if config.filter_region is not None:
            if (not isinstance(config.filter_region, (list, tuple)) or 
                len(config.filter_region) != 4):
                self.errors.append(
                    "filter_region must be a list/tuple of 4 values [x, y, width, height]"
                )
            else:
                x, y, w, h = config.filter_region
                if any(v < 0 for v in [x, y, w, h]):
                    self.errors.append("filter_region values must be non-negative")
    
    def _validate_extractor_config(self, config: FeatureExtractorConfig):
        """抽出器固有設定の検証"""
        extractor_type = config.extractor_type
        
        if extractor_type == 'sift':
            self._validate_sift_config(config.sift)
        elif extractor_type == 'orb':
            self._validate_orb_config(config.orb)
        elif extractor_type == 'surf':
            self._validate_surf_config(config.surf)
        elif extractor_type == 'brief':
            self._validate_brief_config(config.brief)
        elif extractor_type == 'superpoint':
            self._validate_superpoint_config(config.superpoint)
        elif extractor_type == 'disk':
            self._validate_disk_config(config.disk)
        elif extractor_type in ['alike-t', 'alike-s', 'alike-n', 'alike-l']:
            self._validate_alike_config(config.alike, extractor_type)
        elif extractor_type == 'dummy_learned':
            self._validate_dummy_learned_config(config.dummy_learned)
    
    def _validate_sift_config(self, sift_config):
        """SIFT設定の検証"""
        if sift_config.n_features < 0:
            self.errors.append("SIFT n_features must be non-negative")
        
        if sift_config.n_octave_layers <= 0:
            self.errors.append("SIFT n_octave_layers must be positive")
        
        if not 0 < sift_config.contrast_threshold < 1:
            self.errors.append("SIFT contrast_threshold must be between 0 and 1")
        
        if sift_config.edge_threshold <= 0:
            self.errors.append("SIFT edge_threshold must be positive")
        
        if sift_config.sigma <= 0:
            self.errors.append("SIFT sigma must be positive")
    
    def _validate_orb_config(self, orb_config):
        """ORB設定の検証"""
        if orb_config.n_features <= 0:
            self.errors.append("ORB n_features must be positive")
        
        if orb_config.scale_factor <= 1.0:
            self.errors.append("ORB scale_factor must be greater than 1.0")
        
        if orb_config.n_levels <= 0:
            self.errors.append("ORB n_levels must be positive")
        
        if orb_config.edge_threshold <= 0:
            self.errors.append("ORB edge_threshold must be positive")
        
        if orb_config.wta_k not in [2, 3, 4]:
            self.errors.append("ORB wta_k must be 2, 3, or 4")
        
        if orb_config.patch_size <= 0:
            self.errors.append("ORB patch_size must be positive")
    
    def _validate_surf_config(self, surf_config):
        """SURF設定の検証"""
        if surf_config.hessian_threshold <= 0:
            self.errors.append("SURF hessian_threshold must be positive")
        
        if surf_config.n_octaves <= 0:
            self.errors.append("SURF n_octaves must be positive")
        
        if surf_config.n_octave_layers <= 0:
            self.errors.append("SURF n_octave_layers must be positive")
    
    def _validate_brief_config(self, brief_config):
        """BRIEF設定の検証"""
        valid_sizes = [16, 32, 64]
        if brief_config.descriptor_size not in valid_sizes:
            self.errors.append(
                f"BRIEF descriptor_size must be one of {valid_sizes}"
            )
    
    def _validate_superpoint_config(self, superpoint_config):
        """SuperPoint設定の検証"""
        if superpoint_config.nms_radius <= 0:
            self.errors.append("SuperPoint nms_radius must be positive")
        
        if not 0 < superpoint_config.keypoint_threshold < 1:
            self.errors.append("SuperPoint keypoint_threshold must be between 0 and 1")
        
        if superpoint_config.max_keypoints <= 0 and superpoint_config.max_keypoints != -1:
            self.errors.append("SuperPoint max_keypoints must be positive or -1 (unlimited)")
        
        if superpoint_config.remove_borders < 0:
            self.errors.append("SuperPoint remove_borders must be non-negative")
    
    def _validate_disk_config(self, disk_config):
        """DISK設定の検証"""
        valid_dims = [64, 128, 256]
        if disk_config.desc_dim not in valid_dims:
            self.errors.append(f"DISK desc_dim must be one of {valid_dims}")
        
        if disk_config.max_keypoints <= 0:
            self.errors.append("DISK max_keypoints must be positive")
        
        if disk_config.keypoint_threshold < 0:
            self.errors.append("DISK keypoint_threshold must be non-negative")
        
        if disk_config.nms_radius <= 0:
            self.errors.append("DISK nms_radius must be positive")
    
    def _validate_alike_config(self, alike_config, extractor_type: str):
        """ALIKE設定の検証"""
        valid_types = ['alike-t', 'alike-s', 'alike-n', 'alike-l']
        if alike_config.model_type not in valid_types:
            self.errors.append(f"ALIKE model_type must be one of {valid_types}")
        
        # 各モデルタイプに応じた記述子次元の確認
        expected_dims = {
            'alike-t': 64,
            'alike-s': 128,
            'alike-n': 128,
            'alike-l': 256
        }
        
        if extractor_type in expected_dims:
            expected_dim = expected_dims[extractor_type]
            if alike_config.desc_dim != expected_dim:
                self.warnings.append(
                    f"ALIKE {extractor_type} typically uses {expected_dim}D descriptors, "
                    f"but config specifies {alike_config.desc_dim}D"
                )
        
        if alike_config.max_keypoints <= 0:
            self.errors.append("ALIKE max_keypoints must be positive")
        
        if not 0 <= alike_config.keypoint_threshold <= 1:
            self.errors.append("ALIKE keypoint_threshold must be between 0 and 1")
        
        if alike_config.nms_radius <= 0:
            self.errors.append("ALIKE nms_radius must be positive")
    
    def _validate_dummy_learned_config(self, dummy_config):
        """Dummy学習ベース設定の検証"""
        if dummy_config.n_features <= 0:
            self.errors.append("Dummy learned n_features must be positive")
        
        valid_devices = ['cpu', 'cuda', 'mps']
        if dummy_config.device not in valid_devices:
            self.warnings.append(
                f"Dummy learned device '{dummy_config.device}' may not be supported. "
                f"Typical values: {valid_devices}"
            )
    
    def _validate_paths(self, config: FeatureExtractorConfig):
        """パス・ディレクトリの検証"""
        # 入力ディレクトリの確認
        input_path = Path(config.input_dir)
        if not input_path.exists():
            self.errors.append(f"Input directory not found: {input_path}")
        elif not input_path.is_dir():
            self.errors.append(f"Input path is not a directory: {input_path}")
        
        # 出力ディレクトリの確認（存在しない場合は警告）
        output_path = Path(config.output_dir)
        if not output_path.exists():
            self.warnings.append(f"Output directory does not exist: {output_path} (will be created)")
        
        # ファイルパターンの確認
        if not config.file_patterns:
            self.errors.append("file_patterns must be specified and non-empty")
        elif not isinstance(config.file_patterns, list):
            self.errors.append("file_patterns must be a list")
    
    def _validate_visualization_config(self, config: FeatureExtractorConfig):
        """可視化設定の検証"""
        if config.max_display_keypoints <= 0:
            self.errors.append("max_display_keypoints must be positive")
        elif config.max_display_keypoints > 1000:
            self.warnings.append(
                f"max_display_keypoints ({config.max_display_keypoints}) is very large, "
                "which may make visualization cluttered"
            )
    
    def get_report(self) -> str:
        """検証レポートを取得"""
        report = []
        
        if self.errors:
            report.append("ERRORS:")
            for error in self.errors:
                report.append(f"  - {error}")
        
        if self.warnings:
            report.append("WARNINGS:")
            for warning in self.warnings:
                report.append(f"  - {warning}")
        
        if not self.errors and not self.warnings:
            report.append("Configuration is valid!")
        
        return "\n".join(report)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='Validate feature extractor configuration files'
    )
    
    parser.add_argument('config', type=str,
                       help='Path to YAML configuration file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--warnings-as-errors', action='store_true',
                       help='Treat warnings as errors')
    
    args = parser.parse_args()
    
    try:
        # 設定ファイルを読み込み
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
            return 1
        
        print(f"Validating configuration: {config_path}")
        config = load_config(config_path)
        
        # 設定を検証
        validator = ConfigValidator(verbose=args.verbose)
        is_valid = validator.validate(config)
        
        # 結果を表示
        report = validator.get_report()
        print(report)
        
        # 戻り値を決定
        if not is_valid:
            return 1
        elif args.warnings_as_errors and validator.warnings:
            print("\nTreating warnings as errors due to --warnings-as-errors flag")
            return 1
        else:
            return 0
            
    except Exception as e:
        print(f"Error validating configuration: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())