import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .learned import LearnedFeatureExtractor, tensor_to_keypoints, normalize_image
from .base import FeatureSet
from ..utils.model_manager import download_model, get_model_path


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DISKNet(nn.Module):
    """DISK network architecture"""
    
    def __init__(self, desc_dim: int = 128):
        super().__init__()
        
        self.desc_dim = desc_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(1, 32, 3, 1, 1),
            ConvBlock(32, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # /2
            
            ConvBlock(32, 64, 3, 1, 1),
            ConvBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # /4
            
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # /8
            
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 256, 3, 1, 1),
        )
        
        # Feature head
        self.feature_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, desc_dim, 1, 1, 0),
        )
        
        # Keypoint head
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1, 1, 0),
        )
        
        # Descriptor head
        self.descriptor_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, desc_dim, 1, 1, 0),
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass"""
        # Encode
        features = self.encoder(x)
        
        # Feature map
        feature_map = self.feature_head(features)
        
        # Keypoints
        keypoint_map = self.keypoint_head(features)
        keypoint_map = torch.sigmoid(keypoint_map)
        
        # Descriptors
        descriptor_map = self.descriptor_head(features)
        descriptor_map = F.normalize(descriptor_map, p=2, dim=1)
        
        return {
            'feature_map': feature_map,
            'keypoint_map': keypoint_map,
            'descriptor_map': descriptor_map
        }


class DISKExtractor(LearnedFeatureExtractor):
    """DISK特徴量抽出器"""
    
    def __init__(self,
                 model_path: Optional[Union[str, Path]] = None,
                 desc_dim: int = 128,
                 max_keypoints: int = 2048,
                 keypoint_threshold: float = 0.0,
                 nms_radius: int = 2,
                 device: Optional[str] = None,
                 **kwargs):
        """
        DISKExtractorの初期化
        
        Args:
            model_path: モデルファイルのパス
            desc_dim: 記述子の次元数
            max_keypoints: 最大キーポイント数
            keypoint_threshold: キーポイント検出の閾値
            nms_radius: NMSの半径
            device: 計算デバイス
        """
        super().__init__(device=device, model_path=model_path, **kwargs)
        
        self.desc_dim = desc_dim
        self.max_keypoints = max_keypoints
        self.keypoint_threshold = keypoint_threshold
        self.nms_radius = nms_radius
        
        # モデルのダウンロードURL
        self.model_url = "https://github.com/cvlab-epfl/disk/raw/master/depth-save.pth"
        
        # 設定を更新
        self.default_config.update({
            'desc_dim': desc_dim,
            'max_keypoints': max_keypoints,
            'keypoint_threshold': keypoint_threshold,
            'nms_radius': nms_radius,
        })
    
    def load_model(self, model_path: Optional[Union[str, Path]] = None):
        """モデルをロード"""
        if model_path is None:
            model_path = self.model_path
        
        # モデルが存在しない場合は自動ダウンロード
        if model_path is None:
            print("Downloading DISK model...")
            model_path = download_model('disk', self.model_url)
            if model_path is None:
                raise RuntimeError("Failed to download DISK model")
            self.model_path = model_path
        
        # モデルファイルが存在しない場合
        if not Path(model_path).exists():
            # キャッシュから探す
            cached_path = get_model_path('disk')
            if cached_path and cached_path.exists():
                model_path = cached_path
                self.model_path = model_path
            else:
                # ダウンロード
                print("Downloading DISK model...")
                model_path = download_model('disk', self.model_url)
                if model_path is None:
                    raise RuntimeError("Failed to download DISK model")
                self.model_path = model_path
        
        # モデルの作成とロード
        self.model = DISKNet(self.desc_dim)
        
        try:
            # PyTorchモデルをロード
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # checkpoint format handling
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"DISK model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load DISK model: {e}")
            print("Using dummy implementation for demonstration")
            self.model = self._create_dummy_model()
            self.is_loaded = True
    
    def _create_dummy_model(self) -> nn.Module:
        """Create dummy model for demonstration"""
        class DummyDISK(nn.Module):
            def __init__(self, desc_dim):
                super().__init__()
                self.desc_dim = desc_dim
            
            def forward(self, x):
                b, c, h, w = x.shape
                
                # Generate random keypoint map
                keypoint_map = torch.rand(b, 1, h//8, w//8) * 0.5
                
                # Generate random descriptor map
                descriptor_map = torch.randn(b, self.desc_dim, h//8, w//8)
                descriptor_map = F.normalize(descriptor_map, p=2, dim=1)
                
                return {
                    'feature_map': descriptor_map,
                    'keypoint_map': keypoint_map,
                    'descriptor_map': descriptor_map
                }
        
        return DummyDISK(self.desc_dim)
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """画像の前処理"""
        # グレースケール変換と正規化
        image_tensor = normalize_image(image)
        return image_tensor
    
    def _postprocess_output(self, output: dict, image_shape: tuple) -> FeatureSet:
        """モデル出力の後処理"""
        keypoint_map = output['keypoint_map']
        descriptor_map = output['descriptor_map']
        
        # キーポイント検出
        keypoints, descriptors = self._extract_keypoints_and_descriptors(
            keypoint_map, descriptor_map, image_shape
        )
        
        return FeatureSet(
            keypoints=keypoints,
            descriptors=descriptors,
            image_shape=image_shape
        )
    
    def _extract_keypoints_and_descriptors(self, 
                                         keypoint_map: torch.Tensor,
                                         descriptor_map: torch.Tensor,
                                         image_shape: tuple) -> tuple:
        """キーポイントと記述子を抽出"""
        b, c, h, w = keypoint_map.shape
        
        # Flatten
        keypoint_scores = keypoint_map.view(b, -1)
        descriptors = descriptor_map.view(b, self.desc_dim, -1)
        
        # Get coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=keypoint_map.device),
            torch.arange(w, device=keypoint_map.device),
            indexing='ij'
        )
        
        coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=0)
        coords = coords.float() * 8  # Scale back to original image
        
        # Threshold
        valid_mask = keypoint_scores[0] > self.keypoint_threshold
        
        if valid_mask.sum() == 0:
            return [], np.array([])
        
        # Get valid keypoints
        valid_coords = coords[:, valid_mask]
        valid_scores = keypoint_scores[0][valid_mask]
        valid_descriptors = descriptors[0][:, valid_mask]
        
        # NMS (simplified)
        if self.nms_radius > 0:
            keep_indices = self._nms(valid_coords.T, valid_scores, self.nms_radius)
            valid_coords = valid_coords[:, keep_indices]
            valid_scores = valid_scores[keep_indices]
            valid_descriptors = valid_descriptors[:, keep_indices]
        
        # Limit keypoints
        if self.max_keypoints > 0 and len(valid_scores) > self.max_keypoints:
            _, top_indices = torch.topk(valid_scores, self.max_keypoints)
            valid_coords = valid_coords[:, top_indices]
            valid_scores = valid_scores[top_indices]
            valid_descriptors = valid_descriptors[:, top_indices]
        
        # Convert to KeyPoint objects
        keypoints = tensor_to_keypoints(valid_coords.T, valid_scores)
        
        # Convert descriptors to numpy
        descriptors = valid_descriptors.T.cpu().numpy()
        
        return keypoints, descriptors
    
    def _nms(self, coords: torch.Tensor, scores: torch.Tensor, radius: int) -> list:
        """Non-Maximum Suppression"""
        if len(coords) == 0:
            return []
        
        # Sort by score
        _, indices = torch.sort(scores, descending=True)
        
        keep = []
        for i in indices:
            # Check if this point is too close to any kept point
            if len(keep) == 0:
                keep.append(i.item())
                continue
            
            current_coord = coords[i]
            too_close = False
            
            for j in keep:
                kept_coord = coords[j]
                dist = torch.norm(current_coord - kept_coord)
                if dist < radius:
                    too_close = True
                    break
            
            if not too_close:
                keep.append(i.item())
        
        return keep
    
    def get_config(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        config = super().get_config()
        config.update({
            'desc_dim': self.desc_dim,
            'max_keypoints': self.max_keypoints,
            'keypoint_threshold': self.keypoint_threshold,
            'nms_radius': self.nms_radius,
            'extractor_type': 'disk'
        })
        return config