from abc import abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import warnings

from .base import BaseFeatureExtractor, KeyPoint, FeatureSet


class LearnedFeatureExtractor(BaseFeatureExtractor):
    """学習ベース特徴量抽出器の基底クラス"""
    
    def __init__(self, 
                 device: Optional[str] = None,
                 model_path: Optional[Union[str, Path]] = None,
                 **kwargs):
        """
        学習ベース特徴量抽出器の初期化
        
        Args:
            device: 計算デバイス ('cpu', 'cuda', 'cuda:0', etc.)
            model_path: モデルファイルのパス
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        # デバイスの自動選択
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model_path = model_path
        self.model: Optional[nn.Module] = None
        self.is_loaded = False
        
        # デフォルト設定
        self.default_config = {
            'device': str(self.device),
            'model_path': str(model_path) if model_path else None,
        }
        self.default_config.update(kwargs)
    
    @abstractmethod
    def load_model(self, model_path: Optional[Union[str, Path]] = None):
        """
        モデルをロードする
        
        Args:
            model_path: モデルファイルのパス（省略時は初期化時のパスを使用）
        """
        pass
    
    @abstractmethod
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        推論用の画像前処理
        
        Args:
            image: 入力画像
            
        Returns:
            torch.Tensor: 前処理された画像テンソル
        """
        pass
    
    @abstractmethod
    def _postprocess_output(self, output: torch.Tensor, image_shape: tuple) -> FeatureSet:
        """
        推論結果の後処理
        
        Args:
            output: モデルの出力
            image_shape: 元画像のサイズ
            
        Returns:
            FeatureSet: 変換された特徴量セット
        """
        pass
    
    def extract(self, image: np.ndarray) -> FeatureSet:
        """
        画像から特徴量を抽出
        
        Args:
            image: 入力画像
            
        Returns:
            FeatureSet: 抽出された特徴量
        """
        if not self.is_loaded:
            self.load_model()
        
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        
        # 前処理
        input_tensor = self._preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        # 推論
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # 後処理
        return self._postprocess_output(output, image.shape[:2])
    
    def get_config(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        config = self.default_config.copy()
        config.update({
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'model_path': str(self.model_path) if self.model_path else None,
        })
        return config
    
    def to(self, device: Union[str, torch.device]) -> 'LearnedFeatureExtractor':
        """
        デバイスを変更
        
        Args:
            device: 新しいデバイス
            
        Returns:
            self: チェーンのため
        """
        self.device = torch.device(device)
        if self.model is not None:
            self.model = self.model.to(self.device)
        return self
    
    def cuda(self) -> 'LearnedFeatureExtractor':
        """GPUに移動"""
        return self.to('cuda')
    
    def cpu(self) -> 'LearnedFeatureExtractor':
        """CPUに移動"""
        return self.to('cpu')
    
    def eval(self):
        """評価モードに設定"""
        if self.model is not None:
            self.model.eval()
    
    def train(self):
        """学習モードに設定（通常は使用しない）"""
        if self.model is not None:
            self.model.train()
            warnings.warn("Setting model to training mode. This is unusual for feature extraction.")


def tensor_to_keypoints(keypoints_tensor: torch.Tensor, 
                       scores_tensor: torch.Tensor) -> list:
    """
    テンソルからKeyPointのリストに変換
    
    Args:
        keypoints_tensor: キーポイントテンソル [N, 2] (x, y)
        scores_tensor: スコアテンソル [N]
        
    Returns:
        list: KeyPointのリスト
    """
    keypoints = []
    
    # CPUに移動してnumpy配列に変換
    if keypoints_tensor.device.type != 'cpu':
        keypoints_tensor = keypoints_tensor.cpu()
    if scores_tensor.device.type != 'cpu':
        scores_tensor = scores_tensor.cpu()
    
    keypoints_np = keypoints_tensor.numpy()
    scores_np = scores_tensor.numpy()
    
    for i in range(len(keypoints_np)):
        kp = KeyPoint(
            x=float(keypoints_np[i, 0]),
            y=float(keypoints_np[i, 1]),
            scale=1.0,  # 学習ベース手法では通常スケールは固定
            angle=0.0,  # 学習ベース手法では通常角度は0
            response=float(scores_np[i]),
            octave=0,
            class_id=-1
        )
        keypoints.append(kp)
    
    return keypoints


def normalize_image(image: np.ndarray, 
                   mean: tuple = (0.485, 0.456, 0.406),
                   std: tuple = (0.229, 0.224, 0.225)) -> torch.Tensor:
    """
    画像を正規化してテンソルに変換
    
    Args:
        image: 入力画像 [H, W, C] または [H, W]
        mean: 平均値
        std: 標準偏差
        
    Returns:
        torch.Tensor: 正規化された画像テンソル [1, C, H, W]
    """
    # グレースケール変換
    if len(image.shape) == 3:
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # [H, W] -> [1, H, W]
    image = image[None, ...]
    
    # numpy -> tensor
    image_tensor = torch.from_numpy(image).float()
    
    # 正規化 [0, 255] -> [0, 1]
    image_tensor = image_tensor / 255.0
    
    # バッチ次元追加 [1, H, W] -> [1, 1, H, W]
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def nms_keypoints(keypoints: list, 
                 scores: list,
                 radius: float = 4.0,
                 max_keypoints: Optional[int] = None) -> tuple:
    """
    Non-Maximum Suppression for keypoints
    
    Args:
        keypoints: キーポイントのリスト [(x, y), ...]
        scores: スコアのリスト
        radius: NMS半径
        max_keypoints: 最大キーポイント数
        
    Returns:
        tuple: フィルタリングされた(keypoints, scores)
    """
    if len(keypoints) == 0:
        return [], []
    
    # スコアで降順ソート
    indices = np.argsort(scores)[::-1]
    
    if max_keypoints is not None:
        indices = indices[:max_keypoints]
    
    keypoints = np.array(keypoints)[indices]
    scores = np.array(scores)[indices]
    
    # NMS
    keep = []
    for i in range(len(keypoints)):
        if i in keep:
            continue
        
        keep.append(i)
        current_kp = keypoints[i]
        
        # 残りのキーポイントとの距離を計算
        for j in range(i + 1, len(keypoints)):
            if j in keep:
                continue
            
            dist = np.linalg.norm(keypoints[j] - current_kp)
            if dist < radius:
                # 近すぎるキーポイントを除外
                pass
            else:
                keep.append(j)
    
    # より効率的なNMS実装
    keep = []
    for i in range(len(keypoints)):
        current_kp = keypoints[i]
        
        # 既に選択されたキーポイントと距離チェック
        too_close = False
        for j in keep:
            if np.linalg.norm(keypoints[j] - current_kp) < radius:
                too_close = True
                break
        
        if not too_close:
            keep.append(i)
    
    return keypoints[keep].tolist(), scores[keep].tolist()


class DummyLearnedExtractor(LearnedFeatureExtractor):
    """
    学習ベース特徴量抽出器のダミー実装（テスト・デモ用）
    """
    
    def __init__(self, n_features: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.is_loaded = True  # ダミーなので常にロード済み
    
    def load_model(self, model_path: Optional[Union[str, Path]] = None):
        """ダミーモデルロード"""
        self.is_loaded = True
        print(f"Dummy model loaded on {self.device}")
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """ダミー前処理"""
        return normalize_image(image)
    
    def _postprocess_output(self, output: torch.Tensor, image_shape: tuple) -> FeatureSet:
        """ダミー後処理"""
        h, w = image_shape
        
        # ランダムな特徴点を生成
        np.random.seed(42)  # 再現性のため
        keypoints_np = np.random.rand(self.n_features, 2)
        keypoints_np[:, 0] *= w  # x座標
        keypoints_np[:, 1] *= h  # y座標
        
        scores_np = np.random.rand(self.n_features)
        
        # KeyPointに変換
        keypoints = []
        for i in range(self.n_features):
            kp = KeyPoint(
                x=float(keypoints_np[i, 0]),
                y=float(keypoints_np[i, 1]),
                scale=1.0,
                angle=0.0,
                response=float(scores_np[i]),
                octave=0,
                class_id=-1
            )
            keypoints.append(kp)
        
        # ダミー記述子
        descriptors = np.random.rand(self.n_features, 256).astype(np.float32)
        
        return FeatureSet(
            keypoints=keypoints,
            descriptors=descriptors,
            image_shape=image_shape
        )
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'n_features': self.n_features,
            'extractor_type': 'dummy_learned'
        })
        return config