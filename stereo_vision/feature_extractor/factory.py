from typing import Dict, Any, Type, Optional
import warnings

from .base import BaseFeatureExtractor
from .traditional import SIFTExtractor, ORBExtractor, SURFExtractor, BRIEFExtractor
from .learned import DummyLearnedExtractor
from .superpoint import SuperPointExtractor
from .disk import DISKExtractor
from .alike import ALikeExtractor


class FeatureExtractorFactory:
    """特徴量抽出器を生成するファクトリクラス"""
    
    # 利用可能な特徴量抽出器の登録
    _extractors: Dict[str, Type[BaseFeatureExtractor]] = {
        'sift': SIFTExtractor,
        'orb': ORBExtractor,
        'surf': SURFExtractor,
        'brief': BRIEFExtractor,
        'dummy_learned': DummyLearnedExtractor,
        'superpoint': SuperPointExtractor,
        'disk': DISKExtractor,
        'alike-t': ALikeExtractor,
        'alike-s': ALikeExtractor,
        'alike-n': ALikeExtractor,
        'alike-l': ALikeExtractor,
    }
    
    # 各抽出器のデフォルト設定
    _default_configs: Dict[str, Dict[str, Any]] = {
        'sift': {
            'n_features': 0,
            'n_octave_layers': 3,
            'contrast_threshold': 0.04,
            'edge_threshold': 10,
            'sigma': 1.6
        },
        'orb': {
            'n_features': 500,
            'scale_factor': 1.2,
            'n_levels': 8,
            'edge_threshold': 31,
            'first_level': 0,
            'wta_k': 2,
            'patch_size': 31,
            'fast_threshold': 20
        },
        'surf': {
            'hessian_threshold': 400,
            'n_octaves': 4,
            'n_octave_layers': 3,
            'extended': False,
            'upright': False
        },
        'brief': {
            'descriptor_size': 32,
            'use_orientation': False
        },
        'dummy_learned': {
            'n_features': 100,
            'device': 'cpu'
        },
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1,
            'remove_borders': 4
        },
        'disk': {
            'desc_dim': 128,
            'max_keypoints': 2048,
            'keypoint_threshold': 0.0,
            'nms_radius': 2
        },
        'alike-t': {
            'model_type': 'alike-t',
            'desc_dim': 64,
            'max_keypoints': 1000,
            'keypoint_threshold': 0.5,
            'nms_radius': 2
        },
        'alike-s': {
            'model_type': 'alike-s',
            'desc_dim': 128,
            'max_keypoints': 1000,
            'keypoint_threshold': 0.5,
            'nms_radius': 2
        },
        'alike-n': {
            'model_type': 'alike-n',
            'desc_dim': 128,
            'max_keypoints': 1000,
            'keypoint_threshold': 0.5,
            'nms_radius': 2
        },
        'alike-l': {
            'model_type': 'alike-l',
            'desc_dim': 256,
            'max_keypoints': 1000,
            'keypoint_threshold': 0.5,
            'nms_radius': 2
        }
    }
    
    @classmethod
    def create(cls, 
               extractor_type: str, 
               config: Optional[Dict[str, Any]] = None,
               **kwargs) -> BaseFeatureExtractor:
        """
        指定された型の特徴量抽出器を生成
        
        Args:
            extractor_type: 抽出器の種類 ('sift', 'orb', 'surf', 'brief')
            config: 設定辞書 (省略可能)
            **kwargs: 個別の設定パラメータ
            
        Returns:
            BaseFeatureExtractor: 生成された特徴量抽出器
            
        Raises:
            ValueError: 未知の抽出器タイプが指定された場合
            ImportError: 必要な依存関係がインストールされていない場合
        """
        if extractor_type not in cls._extractors:
            available_types = list(cls._extractors.keys())
            raise ValueError(f"Unknown extractor type: {extractor_type}. "
                           f"Available types: {available_types}")
        
        # デフォルト設定を取得
        default_config = cls._default_configs.get(extractor_type, {}).copy()
        
        # 設定をマージ (優先順位: kwargs > config > default)
        if config:
            default_config.update(config)
        default_config.update(kwargs)
        
        # 抽出器を生成
        extractor_class = cls._extractors[extractor_type]
        
        try:
            return extractor_class(**default_config)
        except ImportError as e:
            raise ImportError(f"Failed to create {extractor_type} extractor: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to create {extractor_type} extractor: {e}")
    
    @classmethod
    def get_available_extractors(cls) -> list:
        """
        利用可能な特徴量抽出器の一覧を取得
        
        Returns:
            list: 利用可能な抽出器名のリスト
        """
        return list(cls._extractors.keys())
    
    @classmethod
    def get_default_config(cls, extractor_type: str) -> Dict[str, Any]:
        """
        指定された抽出器のデフォルト設定を取得
        
        Args:
            extractor_type: 抽出器の種類
            
        Returns:
            Dict[str, Any]: デフォルト設定
            
        Raises:
            ValueError: 未知の抽出器タイプが指定された場合
        """
        if extractor_type not in cls._extractors:
            available_types = list(cls._extractors.keys())
            raise ValueError(f"Unknown extractor type: {extractor_type}. "
                           f"Available types: {available_types}")
        
        return cls._default_configs.get(extractor_type, {}).copy()
    
    @classmethod
    def register_extractor(cls, 
                          name: str, 
                          extractor_class: Type[BaseFeatureExtractor],
                          default_config: Optional[Dict[str, Any]] = None):
        """
        新しい特徴量抽出器を登録
        
        Args:
            name: 抽出器の名前
            extractor_class: 抽出器のクラス
            default_config: デフォルト設定 (省略可能)
        """
        if not issubclass(extractor_class, BaseFeatureExtractor):
            raise TypeError(f"extractor_class must be a subclass of BaseFeatureExtractor")
        
        if name in cls._extractors:
            warnings.warn(f"Overriding existing extractor: {name}")
        
        cls._extractors[name] = extractor_class
        if default_config:
            cls._default_configs[name] = default_config
    
    @classmethod
    def unregister_extractor(cls, name: str):
        """
        特徴量抽出器の登録を解除
        
        Args:
            name: 抽出器の名前
        """
        if name in cls._extractors:
            del cls._extractors[name]
        if name in cls._default_configs:
            del cls._default_configs[name]
    
    @classmethod
    def is_available(cls, extractor_type: str) -> bool:
        """
        指定された抽出器が利用可能かどうかを確認
        
        Args:
            extractor_type: 抽出器の種類
            
        Returns:
            bool: 利用可能かどうか
        """
        if extractor_type not in cls._extractors:
            return False
        
        try:
            cls.create(extractor_type)
            return True
        except Exception:
            return False