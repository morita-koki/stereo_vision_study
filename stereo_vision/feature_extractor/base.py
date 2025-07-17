from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class KeyPoint:
    """特徴点の情報を格納するデータクラス"""
    x: float
    y: float
    scale: float
    angle: float
    response: float
    octave: int = 0
    class_id: int = -1


@dataclass
class FeatureSet:
    """特徴点と記述子をまとめたデータ構造"""
    keypoints: List[KeyPoint]
    descriptors: np.ndarray
    image_shape: Tuple[int, int]
    
    def __len__(self) -> int:
        """特徴点の数を返す"""
        return len(self.keypoints)
    
    def is_empty(self) -> bool:
        """特徴点が空かどうかを判定"""
        return len(self.keypoints) == 0


class BaseFeatureExtractor(ABC):
    """特徴量抽出器の抽象基底クラス"""
    
    def __init__(self, **kwargs):
        """初期化メソッド"""
        self.config = kwargs
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> FeatureSet:
        """
        画像から特徴量を抽出する
        
        Args:
            image: 入力画像 (グレースケールまたはカラー)
            
        Returns:
            FeatureSet: 抽出された特徴点と記述子
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        現在の設定を取得する
        
        Returns:
            Dict[str, Any]: 設定辞書
        """
        pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        画像の前処理を行う
        
        Args:
            image: 入力画像
            
        Returns:
            np.ndarray: 前処理された画像
        """
        if len(image.shape) == 3:
            import cv2
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def __str__(self) -> str:
        """オブジェクトの文字列表現"""
        return f"{self.__class__.__name__}({self.get_config()})"
    
    def __repr__(self) -> str:
        """オブジェクトの詳細表現"""
        return self.__str__()