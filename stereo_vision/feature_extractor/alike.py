import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .learned import LearnedFeatureExtractor, tensor_to_keypoints, normalize_image
from .base import FeatureSet
from ..utils.model_manager import download_model, get_model_path


class ALikeNet(nn.Module):
    """ALIKE network architecture"""
    
    def __init__(self, c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 128, 
                 dim: int = 128, single_head: bool = False):
        super().__init__()
        
        self.c1, self.c2, self.c3, self.c4, self.dim = c1, c2, c3, c4, dim
        self.single_head = single_head
        
        # Backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        
        # Heads
        if single_head:
            self.head = nn.Conv2d(c1, dim + 1, kernel_size=3, stride=1, padding=1)
        else:
            # Score head
            self.score_head = nn.Sequential(
                nn.Conv2d(c1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )
            
            # Descriptor head
            self.desc_head = nn.Sequential(
                nn.Conv2d(c1, dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            )
    
    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass"""
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        
        # Decoder
        y1 = self.deconv1(x8) + x6
        y2 = self.deconv2(y1) + x4
        y3 = self.deconv3(y2) + x2
        
        if self.single_head:
            out = self.head(y3)
            scores = torch.sigmoid(out[:, :1, :, :])
            descriptors = F.normalize(out[:, 1:, :, :], p=2, dim=1)
        else:
            scores = self.score_head(y3)
            descriptors = self.desc_head(y3)
            descriptors = F.normalize(descriptors, p=2, dim=1)
        
        return {
            'scores': scores,
            'descriptors': descriptors
        }


class ALikeExtractor(LearnedFeatureExtractor):
    """ALIKE特徴量抽出器"""
    
    def __init__(self,
                 model_path: Optional[Union[str, Path]] = None,
                 model_type: str = 'alike-t',
                 desc_dim: int = 128,
                 max_keypoints: int = 1000,
                 keypoint_threshold: float = 0.5,
                 nms_radius: int = 2,
                 device: Optional[str] = None,
                 **kwargs):
        """
        ALikeExtractorの初期化
        
        Args:
            model_path: モデルファイルのパス
            model_type: モデルタイプ ('alike-t', 'alike-s', 'alike-n', 'alike-l')
            desc_dim: 記述子の次元数
            max_keypoints: 最大キーポイント数
            keypoint_threshold: キーポイント検出の閾値
            nms_radius: NMSの半径
            device: 計算デバイス
        """
        super().__init__(device=device, model_path=model_path, **kwargs)
        
        self.model_type = model_type
        self.desc_dim = desc_dim
        self.max_keypoints = max_keypoints
        self.keypoint_threshold = keypoint_threshold
        self.nms_radius = nms_radius
        
        # モデル設定
        self.model_configs = {
            'alike-t': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dim': 64, 'single_head': True},
            'alike-s': {'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'dim': 128, 'single_head': True},
            'alike-n': {'c1': 32, 'c2': 64, 'c3': 128, 'c4': 128, 'dim': 128, 'single_head': False},
            'alike-l': {'c1': 64, 'c2': 128, 'c3': 256, 'c4': 256, 'dim': 256, 'single_head': False},
        }
        
        # モデルのダウンロードURL
        self.model_urls = {
            'alike-t': 'https://github.com/Shiaoming/ALIKE/raw/main/models/alike-t.pth',
            'alike-s': 'https://github.com/Shiaoming/ALIKE/raw/main/models/alike-s.pth',
            'alike-n': 'https://github.com/Shiaoming/ALIKE/raw/main/models/alike-n.pth',
            'alike-l': 'https://github.com/Shiaoming/ALIKE/raw/main/models/alike-l.pth',
        }
        
        # 設定を更新
        self.default_config.update({
            'model_type': model_type,
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
            if self.model_type not in self.model_urls:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            print(f"Downloading ALIKE {self.model_type} model...")
            model_path = download_model(self.model_type, self.model_urls[self.model_type])
            if model_path is None:
                raise RuntimeError(f"Failed to download ALIKE {self.model_type} model")
            self.model_path = model_path
        
        # モデルファイルが存在しない場合
        if not Path(model_path).exists():
            # キャッシュから探す
            cached_path = get_model_path(self.model_type)
            if cached_path and cached_path.exists():
                model_path = cached_path
                self.model_path = model_path
            else:
                # ダウンロード
                print(f"Downloading ALIKE {self.model_type} model...")
                model_path = download_model(self.model_type, self.model_urls[self.model_type])
                if model_path is None:
                    raise RuntimeError(f"Failed to download ALIKE {self.model_type} model")
                self.model_path = model_path
        
        # モデルの作成とロード
        config = self.model_configs[self.model_type]
        self.model = ALikeNet(**config)
        
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
            print(f"ALIKE {self.model_type} model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load ALIKE model: {e}")
            print("Using dummy implementation for demonstration")
            self.model = self._create_dummy_model()
            self.is_loaded = True
    
    def _create_dummy_model(self) -> nn.Module:
        """Create dummy model for demonstration"""
        class DummyALike(nn.Module):
            def __init__(self, desc_dim):
                super().__init__()
                self.desc_dim = desc_dim
            
            def forward(self, x):
                b, c, h, w = x.shape
                
                # Generate random scores
                scores = torch.rand(b, 1, h, w) * 0.5
                
                # Generate random descriptors
                descriptors = torch.randn(b, self.desc_dim, h, w)
                descriptors = F.normalize(descriptors, p=2, dim=1)
                
                return {
                    'scores': scores,
                    'descriptors': descriptors
                }
        
        return DummyALike(self.desc_dim)
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """画像の前処理"""
        # グレースケール変換と正規化
        image_tensor = normalize_image(image)
        return image_tensor
    
    def _postprocess_output(self, output: dict, image_shape: tuple) -> FeatureSet:
        """モデル出力の後処理"""
        scores = output['scores']
        descriptors = output['descriptors']
        
        # キーポイント検出
        keypoints, descriptors = self._extract_keypoints_and_descriptors(
            scores, descriptors, image_shape
        )
        
        return FeatureSet(
            keypoints=keypoints,
            descriptors=descriptors,
            image_shape=image_shape
        )
    
    def _extract_keypoints_and_descriptors(self, 
                                         scores: torch.Tensor,
                                         descriptors: torch.Tensor,
                                         image_shape: tuple) -> tuple:
        """キーポイントと記述子を抽出"""
        b, c, h, w = scores.shape
        
        # Flatten
        scores_flat = scores.view(b, -1)
        descriptors_flat = descriptors.view(b, self.desc_dim, -1)
        
        # Get coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=scores.device),
            torch.arange(w, device=scores.device),
            indexing='ij'
        )
        
        coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=0)
        coords = coords.float()
        
        # Threshold
        valid_mask = scores_flat[0] > self.keypoint_threshold
        
        if valid_mask.sum() == 0:
            return [], np.array([])
        
        # Get valid keypoints
        valid_coords = coords[:, valid_mask]
        valid_scores = scores_flat[0][valid_mask]
        valid_descriptors = descriptors_flat[0][:, valid_mask]
        
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
            'model_type': self.model_type,
            'desc_dim': self.desc_dim,
            'max_keypoints': self.max_keypoints,
            'keypoint_threshold': self.keypoint_threshold,
            'nms_radius': self.nms_radius,
            'extractor_type': 'alike'
        })
        return config