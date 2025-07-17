import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class SIFTConfig:
    """SIFT特徴量抽出器の設定"""
    n_features: int = 0
    n_octave_layers: int = 3
    contrast_threshold: float = 0.04
    edge_threshold: float = 10
    sigma: float = 1.6


@dataclass
class ORBConfig:
    """ORB特徴量抽出器の設定"""
    n_features: int = 500
    scale_factor: float = 1.2
    n_levels: int = 8
    edge_threshold: int = 31
    first_level: int = 0
    wta_k: int = 2
    patch_size: int = 31
    fast_threshold: int = 20


@dataclass
class SURFConfig:
    """SURF特徴量抽出器の設定"""
    hessian_threshold: float = 400
    n_octaves: int = 4
    n_octave_layers: int = 3
    extended: bool = False
    upright: bool = False


@dataclass
class BRIEFConfig:
    """BRIEF特徴量抽出器の設定"""
    descriptor_size: int = 32
    use_orientation: bool = False


@dataclass
class FeatureExtractorConfig:
    """特徴量抽出器の全体設定"""
    extractor_type: str = "sift"
    max_keypoints: Optional[int] = None
    min_response: Optional[float] = None
    filter_region: Optional[tuple] = None
    sift: SIFTConfig = None
    orb: ORBConfig = None
    surf: SURFConfig = None
    brief: BRIEFConfig = None
    
    def __post_init__(self):
        """初期化後の処理"""
        if self.sift is None:
            self.sift = SIFTConfig()
        if self.orb is None:
            self.orb = ORBConfig()
        if self.surf is None:
            self.surf = SURFConfig()
        if self.brief is None:
            self.brief = BRIEFConfig()


class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self):
        self.config = FeatureExtractorConfig()
    
    def load_from_dict(self, config_dict: Dict[str, Any]):
        """
        辞書から設定を読み込み
        
        Args:
            config_dict: 設定辞書
        """
        # 基本設定
        self.config.extractor_type = config_dict.get('extractor_type', 'sift')
        self.config.max_keypoints = config_dict.get('max_keypoints')
        self.config.min_response = config_dict.get('min_response')
        self.config.filter_region = config_dict.get('filter_region')
        
        # SIFT設定
        if 'sift' in config_dict:
            sift_config = config_dict['sift']
            self.config.sift = SIFTConfig(**sift_config)
        
        # ORB設定
        if 'orb' in config_dict:
            orb_config = config_dict['orb']
            self.config.orb = ORBConfig(**orb_config)
        
        # SURF設定
        if 'surf' in config_dict:
            surf_config = config_dict['surf']
            self.config.surf = SURFConfig(**surf_config)
        
        # BRIEF設定
        if 'brief' in config_dict:
            brief_config = config_dict['brief']
            self.config.brief = BRIEFConfig(**brief_config)
    
    def load_from_json(self, json_path: Union[str, Path]):
        """
        JSONファイルから設定を読み込み
        
        Args:
            json_path: JSONファイルのパス
        """
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        self.load_from_dict(config_dict)
    
    def load_from_yaml(self, yaml_path: Union[str, Path]):
        """
        YAMLファイルから設定を読み込み
        
        Args:
            yaml_path: YAMLファイルのパス
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        self.load_from_dict(config_dict)
    
    def save_to_json(self, json_path: Union[str, Path]):
        """
        JSONファイルに設定を保存
        
        Args:
            json_path: JSONファイルのパス
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self.config)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def save_to_yaml(self, yaml_path: Union[str, Path]):
        """
        YAMLファイルに設定を保存
        
        Args:
            yaml_path: YAMLファイルのパス
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self.config)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def get_extractor_config(self, extractor_type: Optional[str] = None) -> Dict[str, Any]:
        """
        指定された抽出器の設定を取得
        
        Args:
            extractor_type: 抽出器の種類（省略時は現在の設定を使用）
            
        Returns:
            Dict[str, Any]: 抽出器の設定
        """
        if extractor_type is None:
            extractor_type = self.config.extractor_type
        
        if extractor_type == 'sift':
            return asdict(self.config.sift)
        elif extractor_type == 'orb':
            return asdict(self.config.orb)
        elif extractor_type == 'surf':
            return asdict(self.config.surf)
        elif extractor_type == 'brief':
            return asdict(self.config.brief)
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
    
    def update_extractor_config(self, extractor_type: str, **kwargs):
        """
        指定された抽出器の設定を更新
        
        Args:
            extractor_type: 抽出器の種類
            **kwargs: 更新する設定
        """
        if extractor_type == 'sift':
            for key, value in kwargs.items():
                if hasattr(self.config.sift, key):
                    setattr(self.config.sift, key, value)
        elif extractor_type == 'orb':
            for key, value in kwargs.items():
                if hasattr(self.config.orb, key):
                    setattr(self.config.orb, key, value)
        elif extractor_type == 'surf':
            for key, value in kwargs.items():
                if hasattr(self.config.surf, key):
                    setattr(self.config.surf, key, value)
        elif extractor_type == 'brief':
            for key, value in kwargs.items():
                if hasattr(self.config.brief, key):
                    setattr(self.config.brief, key, value)
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        設定を辞書として取得
        
        Returns:
            Dict[str, Any]: 設定辞書
        """
        return asdict(self.config)
    
    def __str__(self) -> str:
        """文字列表現"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def create_default_config() -> FeatureExtractorConfig:
    """
    デフォルト設定を作成
    
    Returns:
        FeatureExtractorConfig: デフォルト設定
    """
    return FeatureExtractorConfig()


def load_config(config_path: Union[str, Path]) -> FeatureExtractorConfig:
    """
    設定ファイルを読み込み
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        FeatureExtractorConfig: 設定
    """
    config_path = Path(config_path)
    manager = ConfigManager()
    
    if config_path.suffix.lower() == '.json':
        manager.load_from_json(config_path)
    elif config_path.suffix.lower() in ['.yaml', '.yml']:
        manager.load_from_yaml(config_path)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return manager.config