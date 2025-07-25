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
class SuperPointConfig:
    """SuperPoint特徴量抽出器の設定"""
    nms_radius: int = 4
    keypoint_threshold: float = 0.005
    max_keypoints: int = -1
    remove_borders: int = 4


@dataclass
class DISKConfig:
    """DISK特徴量抽出器の設定"""
    desc_dim: int = 128
    max_keypoints: int = 2048
    keypoint_threshold: float = 0.0
    nms_radius: int = 2


@dataclass
class ALikeConfig:
    """ALIKE特徴量抽出器の設定"""
    model_type: str = "alike-t"  # alike-t, alike-s, alike-n, alike-l
    desc_dim: int = 64  # 64 for alike-t, 128 for alike-s/n, 256 for alike-l
    max_keypoints: int = 1000
    keypoint_threshold: float = 0.5
    nms_radius: int = 2


@dataclass
class DummyLearnedConfig:
    """Dummy学習ベース特徴量抽出器の設定"""
    n_features: int = 100
    device: str = "cpu"


@dataclass
class FeatureExtractorConfig:
    """特徴量抽出器の全体設定"""
    # 基本設定
    extractor_type: str = "sift"
    input_dir: str = "data/"
    output_dir: str = "output/"
    max_keypoints: Optional[int] = None
    min_response: Optional[float] = None
    filter_region: Optional[tuple] = None
    
    # 可視化設定
    visualization_enabled: bool = True
    show_response: bool = False
    color_by_response: bool = False
    max_display_keypoints: int = 100
    
    # バッチ処理設定
    file_patterns: list = None
    parallel_processing: bool = False
    
    # 従来手法の設定（使わなくても全て含める）
    sift: SIFTConfig = None
    orb: ORBConfig = None
    surf: SURFConfig = None
    brief: BRIEFConfig = None
    
    # 学習ベース手法の設定（使わなくても全て含める）
    superpoint: SuperPointConfig = None
    disk: DISKConfig = None
    alike: ALikeConfig = None
    dummy_learned: DummyLearnedConfig = None
    
    def __post_init__(self):
        """初期化後の処理"""
        # 従来手法の設定初期化
        if self.sift is None:
            self.sift = SIFTConfig()
        if self.orb is None:
            self.orb = ORBConfig()
        if self.surf is None:
            self.surf = SURFConfig()
        if self.brief is None:
            self.brief = BRIEFConfig()
        
        # 学習ベース手法の設定初期化
        if self.superpoint is None:
            self.superpoint = SuperPointConfig()
        if self.disk is None:
            self.disk = DISKConfig()
        if self.alike is None:
            self.alike = ALikeConfig()
        if self.dummy_learned is None:
            self.dummy_learned = DummyLearnedConfig()
        
        # file_patternsのデフォルト値
        if self.file_patterns is None:
            self.file_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]


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
        self.config.input_dir = config_dict.get('input_dir', 'data/')
        self.config.output_dir = config_dict.get('output_dir', 'output/')
        self.config.max_keypoints = config_dict.get('max_keypoints')
        self.config.min_response = config_dict.get('min_response')
        self.config.filter_region = config_dict.get('filter_region')
        
        # 可視化設定
        self.config.visualization_enabled = config_dict.get('visualization_enabled', True)
        self.config.show_response = config_dict.get('show_response', False)
        self.config.color_by_response = config_dict.get('color_by_response', False)
        self.config.max_display_keypoints = config_dict.get('max_display_keypoints', 100)
        
        # バッチ処理設定
        self.config.file_patterns = config_dict.get('file_patterns')
        self.config.parallel_processing = config_dict.get('parallel_processing', False)
        
        # 従来手法の設定
        if 'sift' in config_dict:
            sift_config = config_dict['sift']
            self.config.sift = SIFTConfig(**sift_config)
        
        if 'orb' in config_dict:
            orb_config = config_dict['orb']
            self.config.orb = ORBConfig(**orb_config)
        
        if 'surf' in config_dict:
            surf_config = config_dict['surf']
            self.config.surf = SURFConfig(**surf_config)
        
        if 'brief' in config_dict:
            brief_config = config_dict['brief']
            self.config.brief = BRIEFConfig(**brief_config)
        
        # 学習ベース手法の設定
        if 'superpoint' in config_dict:
            superpoint_config = config_dict['superpoint']
            self.config.superpoint = SuperPointConfig(**superpoint_config)
        
        if 'disk' in config_dict:
            disk_config = config_dict['disk']
            self.config.disk = DISKConfig(**disk_config)
        
        if 'alike' in config_dict:
            alike_config = config_dict['alike']
            self.config.alike = ALikeConfig(**alike_config)
        
        if 'dummy_learned' in config_dict:
            dummy_learned_config = config_dict['dummy_learned']
            self.config.dummy_learned = DummyLearnedConfig(**dummy_learned_config)
    
    
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
        elif extractor_type == 'superpoint':
            return asdict(self.config.superpoint)
        elif extractor_type == 'disk':
            return asdict(self.config.disk)
        elif extractor_type in ['alike-t', 'alike-s', 'alike-n', 'alike-l']:
            config = asdict(self.config.alike)
            config['model_type'] = extractor_type
            # ALikeの各モデルに応じたdesc_dimを設定
            if extractor_type == 'alike-t':
                config['desc_dim'] = 64
            elif extractor_type in ['alike-s', 'alike-n']:
                config['desc_dim'] = 128
            elif extractor_type == 'alike-l':
                config['desc_dim'] = 256
            return config
        elif extractor_type == 'dummy_learned':
            return asdict(self.config.dummy_learned)
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
        elif extractor_type == 'superpoint':
            for key, value in kwargs.items():
                if hasattr(self.config.superpoint, key):
                    setattr(self.config.superpoint, key, value)
        elif extractor_type == 'disk':
            for key, value in kwargs.items():
                if hasattr(self.config.disk, key):
                    setattr(self.config.disk, key, value)
        elif extractor_type in ['alike-t', 'alike-s', 'alike-n', 'alike-l']:
            for key, value in kwargs.items():
                if hasattr(self.config.alike, key):
                    setattr(self.config.alike, key, value)
        elif extractor_type == 'dummy_learned':
            for key, value in kwargs.items():
                if hasattr(self.config.dummy_learned, key):
                    setattr(self.config.dummy_learned, key, value)
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
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)


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
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        manager.load_from_yaml(config_path)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}. Only YAML files are supported.")
    
    return manager.config