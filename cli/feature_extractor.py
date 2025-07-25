#!/usr/bin/env python3
"""
Configuration-based Feature Extractor CLI

This script provides a configuration-driven interface for feature extraction.
All parameters are specified in YAML configuration files, ensuring
reproducibility and easy parameter management.

Usage:
    uv run cli/feature_extractor.py --config config.yaml
    uv run cli/feature_extractor.py --config config.yaml --output-dir custom_output
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from glob import glob

# Add the package to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from stereo_vision.feature_extractor import (
    FeatureExtractorFactory,
    get_available_extractors,
    print_feature_statistics
)
from stereo_vision.feature_extractor.config import (
    ConfigManager,
    FeatureExtractorConfig,
    load_config
)
from stereo_vision.feature_extractor.utils import (
    draw_keypoints,
    draw_keypoints_advanced,
    save_keypoints_image,
    save_keypoints_comparison
)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    ログ設定をセットアップ
    
    Args:
        level: ログレベル
        
    Returns:
        logging.Logger: 設定されたロガー
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def validate_config(config: FeatureExtractorConfig) -> bool:
    """
    設定の妥当性を検証
    
    Args:
        config: 設定オブジェクト
        
    Returns:
        bool: 妥当性
    """
    # 抽出器タイプの確認
    available_extractors = get_available_extractors()
    if config.extractor_type not in available_extractors:
        raise ValueError(f"Unknown extractor type: {config.extractor_type}. "
                        f"Available: {available_extractors}")
    
    # 入力ディレクトリの確認
    input_path = Path(config.input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    
    # ファイルパターンの確認
    if not config.file_patterns:
        raise ValueError("file_patterns must be specified")
    
    return True


def find_images(input_dir: str, file_patterns: List[str]) -> List[Path]:
    """
    指定されたディレクトリから画像ファイルを検索
    
    Args:
        input_dir: 入力ディレクトリ
        file_patterns: ファイルパターンのリスト
        
    Returns:
        List[Path]: 見つかった画像ファイルのリスト
    """
    input_path = Path(input_dir)
    image_files = []
    
    for pattern in file_patterns:
        pattern_path = input_path / pattern
        matches = glob(str(pattern_path), recursive=True)
        image_files.extend([Path(f) for f in matches])
    
    # 重複を除去してソート
    image_files = sorted(list(set(image_files)))
    
    return image_files


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    画像を読み込み
    
    Args:
        image_path: 画像ファイルパス
        
    Returns:
        Optional[np.ndarray]: 読み込まれた画像（グレースケール）
    """
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None


def extract_features_single(image: np.ndarray, 
                          config: FeatureExtractorConfig,
                          logger: logging.Logger) -> Dict[str, Any]:
    """
    単一画像から特徴量を抽出
    
    Args:
        image: 入力画像
        config: 設定
        logger: ロガー
        
    Returns:
        Dict[str, Any]: 抽出結果
    """
    try:
        # 設定マネージャーを作成
        config_manager = ConfigManager()
        config_manager.config = config
        
        # 抽出器固有の設定を取得
        extractor_config = config_manager.get_extractor_config()
        
        # max_keypointsの設定を抽出器に合わせて調整
        if config.max_keypoints is not None:
            if config.extractor_type in ['sift', 'orb', 'surf']:
                extractor_config['n_features'] = config.max_keypoints
            elif config.extractor_type in ['superpoint', 'disk', 'alike-t', 'alike-s', 'alike-n', 'alike-l']:
                extractor_config['max_keypoints'] = config.max_keypoints
            elif config.extractor_type == 'dummy_learned':
                extractor_config['n_features'] = config.max_keypoints
        
        # 特徴量抽出器を作成
        extractor = FeatureExtractorFactory.create(config.extractor_type, **extractor_config)
        
        # 特徴量抽出の実行
        start_time = time.time()
        features = extractor.extract(image)
        end_time = time.time()
        
        extraction_time = end_time - start_time
        
        logger.debug(f"Extracted {len(features)} features in {extraction_time:.4f}s")
        
        return {
            'features': features,
            'extraction_time': extraction_time,
            'extractor_config': extractor_config,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return {
            'features': None,
            'extraction_time': None,
            'extractor_config': None,
            'success': False,
            'error': str(e)
        }


def save_visualization(image: np.ndarray,
                      features,
                      output_path: Path,
                      config: FeatureExtractorConfig,
                      logger: logging.Logger):
    """
    可視化結果を保存
    
    Args:
        image: 入力画像
        features: 抽出された特徴
        output_path: 出力パス（拡張子なし）
        config: 設定
        logger: ロガー
    """
    try:
        if not config.visualization_enabled or len(features.keypoints) == 0:
            return
        
        # 基本的な可視化
        basic_vis = draw_keypoints(image, features, (0, 255, 0), True)
        basic_path = output_path.parent / f"{output_path.stem}_basic.jpg"
        cv2.imwrite(str(basic_path), basic_vis)
        logger.debug(f"Saved basic visualization: {basic_path}")
        
        # 高度な可視化
        if config.show_response or config.color_by_response:
            advanced_vis = draw_keypoints_advanced(
                image, features,
                show_response=config.show_response,
                color_by_response=config.color_by_response,
                max_keypoints_display=config.max_display_keypoints
            )
            advanced_path = output_path.parent / f"{output_path.stem}_advanced.jpg"
            cv2.imwrite(str(advanced_path), advanced_vis)
            logger.debug(f"Saved advanced visualization: {advanced_path}")
        
    except Exception as e:
        logger.error(f"Error saving visualization: {e}")


def process_images(config: FeatureExtractorConfig, logger: logging.Logger) -> Dict[str, Any]:
    """
    設定に基づいて画像を処理
    
    Args:
        config: 設定
        logger: ロガー
        
    Returns:
        Dict[str, Any]: 処理結果の統計
    """
    # 画像ファイルを検索
    image_files = find_images(config.input_dir, config.file_patterns)
    
    if not image_files:
        logger.warning(f"No images found in {config.input_dir} with patterns {config.file_patterns}")
        return {'processed': 0, 'failed': 0, 'total_time': 0}
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # 出力ディレクトリを作成
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 処理統計
    stats = {
        'processed': 0,
        'failed': 0,
        'total_time': 0,
        'results': []
    }
    
    # 各画像を処理
    for i, image_file in enumerate(image_files, 1):
        logger.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
        
        # 画像を読み込み
        image = load_image(image_file)
        if image is None:
            stats['failed'] += 1
            continue
        
        # 特徴量抽出
        result = extract_features_single(image, config, logger)
        
        if result['success']:
            stats['processed'] += 1
            stats['total_time'] += result['extraction_time']
            
            # 統計表示
            features = result['features']
            logger.info(f"  Extracted {len(features)} features in {result['extraction_time']:.4f}s")
            
            # 可視化保存
            output_file_path = output_path / image_file.stem
            save_visualization(image, features, output_file_path, config, logger)
            
            # 結果を記録
            stats['results'].append({
                'file': str(image_file),
                'num_features': len(features),
                'extraction_time': result['extraction_time']
            })
            
        else:
            stats['failed'] += 1
            logger.error(f"  Failed: {result['error']}")
    
    return stats


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='Configuration-based Feature Extractor CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    uv run cli/feature_extractor.py --config configs/sift_config.yaml
    uv run cli/feature_extractor.py --config configs/superpoint_config.yaml --log-level DEBUG
    uv run cli/feature_extractor.py --config config.yaml --output-dir custom_output
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--output-dir', type=str,
                       help='Override output directory from config')
    parser.add_argument('--input-dir', type=str,
                       help='Override input directory from config')
    parser.add_argument('--extractor', type=str,
                       help='Override extractor type from config')
    parser.add_argument('--max-keypoints', type=int,
                       help='Override max keypoints from config')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate config and show what would be processed without actual processing')
    
    args = parser.parse_args()
    
    # ログ設定
    logger = setup_logging(args.log_level)
    
    try:
        # 設定ファイルを読み込み
        logger.info(f"Loading configuration from: {args.config}")
        config_path = Path(args.config)
        
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1
        
        config = load_config(config_path)
        
        # コマンドライン引数で設定をオーバーライド
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.input_dir:
            config.input_dir = args.input_dir
        if args.extractor:
            config.extractor_type = args.extractor
        if args.max_keypoints:
            config.max_keypoints = args.max_keypoints
        
        logger.info(f"Configuration loaded:")
        logger.info(f"  Extractor: {config.extractor_type}")
        logger.info(f"  Input dir: {config.input_dir}")
        logger.info(f"  Output dir: {config.output_dir}")
        logger.info(f"  Max keypoints: {config.max_keypoints}")
        logger.info(f"  File patterns: {config.file_patterns}")
        
        # 設定の妥当性を検証
        validate_config(config)
        
        # Dry run の場合は検証のみ
        if args.dry_run:
            image_files = find_images(config.input_dir, config.file_patterns)
            logger.info(f"Dry run: Would process {len(image_files)} images")
            for img_file in image_files[:5]:  # 最初の5ファイルを表示
                logger.info(f"  {img_file}")
            if len(image_files) > 5:
                logger.info(f"  ... and {len(image_files) - 5} more files")
            return 0
        
        # 画像処理を実行
        logger.info("Starting feature extraction...")
        stats = process_images(config, logger)
        
        # 結果統計を表示
        logger.info("Processing completed!")
        logger.info(f"  Processed: {stats['processed']} images")
        logger.info(f"  Failed: {stats['failed']} images")
        if stats['processed'] > 0:
            avg_time = stats['total_time'] / stats['processed']
            logger.info(f"  Average extraction time: {avg_time:.4f}s")
            logger.info(f"  Total time: {stats['total_time']:.4f}s")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())