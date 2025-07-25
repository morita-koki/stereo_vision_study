import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .learned import LearnedFeatureExtractor, tensor_to_keypoints, normalize_image, nms_keypoints
from .base import FeatureSet
from ..utils.model_manager import download_model, get_model_path


class SuperPointNet(nn.Module):
    """SuperPointネットワークの実装"""
    
    def __init__(self):
        super().__init__()
        
        # Shared Encoder
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        
        # Encoder
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        
        # Detector Head
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        
        # Descriptor Head
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, 256, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        """Forward pass"""
        # Shared Encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)  # /2
        
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)  # /4
        
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)  # /8
        
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        
        # Detector Head
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)  # [B, 65, H/8, W/8]
        
        # Descriptor Head
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)  # [B, 256, H/8, W/8]
        
        # Normalize descriptors
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        
        return semi, desc


class SuperPointExtractor(LearnedFeatureExtractor):
    """SuperPoint特徴量抽出器"""
    
    def __init__(self,
                 model_path: Optional[Union[str, Path]] = None,
                 nms_radius: int = 4,
                 keypoint_threshold: float = 0.005,
                 max_keypoints: int = -1,
                 remove_borders: int = 4,
                 device: Optional[str] = None,
                 **kwargs):
        """
        SuperPointExtractorの初期化
        
        Args:
            model_path: モデルファイルのパス
            nms_radius: NMSの半径
            keypoint_threshold: キーポイント検出の閾値
            max_keypoints: 最大キーポイント数（-1で制限なし）
            remove_borders: 境界から除去する画素数
            device: 計算デバイス
        """
        super().__init__(device=device, model_path=model_path, **kwargs)
        
        self.nms_radius = nms_radius
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self.remove_borders = remove_borders
        
        # モデルのダウンロードURL
        # Official SuperPoint model URL from Magic Leap
        self.model_url = "https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth"
        
        # 設定を更新
        self.default_config.update({
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints,
            'remove_borders': remove_borders,
        })
    
    def load_model(self, model_path: Optional[Union[str, Path]] = None):
        """モデルをロード"""
        if model_path is None:
            model_path = self.model_path
        
        # モデルが存在しない場合は自動ダウンロード
        if model_path is None:
            print("Downloading SuperPoint model...")
            model_path = download_model('superpoint_v1', self.model_url)
            
            if model_path is None:
                # Try alternative URLs
                alternative_urls = [
                    "https://raw.githubusercontent.com/magicleap/SuperPointPretrainedNetwork/master/superpoint_v1.pth",
                    "https://github.com/eric-yyjau/pytorch-superpoint/raw/master/pretrained/superpoint_v1.pth"
                ]
                
                for alt_url in alternative_urls:
                    print(f"Trying alternative URL: {alt_url}")
                    model_path = download_model('superpoint_v1', alt_url)
                    if model_path is not None:
                        break
                
                if model_path is None:
                    print("SuperPoint model download failed. Please check your internet connection.")
                    print("Alternative: Use traditional extractors like 'sift' or 'orb' which work without downloading models.")
                    raise RuntimeError("Failed to download SuperPoint model from all sources")
            
            self.model_path = model_path
        
        # モデルファイルが存在しない場合
        if not Path(model_path).exists():
            # キャッシュから探す
            cached_path = get_model_path('superpoint_v1')
            if cached_path and cached_path.exists():
                model_path = cached_path
                self.model_path = model_path
            else:
                # ダウンロード
                print("Downloading SuperPoint model...")
                model_path = download_model('superpoint_v1', self.model_url)
                
                if model_path is None:
                    print("Primary URL failed, trying alternatives...")
                    alternative_urls = [
                        "https://raw.githubusercontent.com/magicleap/SuperPointPretrainedNetwork/master/superpoint_v1.pth",
                        "https://github.com/eric-yyjau/pytorch-superpoint/raw/master/pretrained/superpoint_v1.pth"
                    ]
                    
                    for alt_url in alternative_urls:
                        print(f"Trying: {alt_url}")
                        model_path = download_model('superpoint_v1', alt_url)
                        if model_path is not None:
                            break
                    
                    if model_path is None:
                        print("All download attempts failed. Please use traditional extractors like 'sift' or 'orb'.")
                        raise RuntimeError("Failed to download SuperPoint model from all sources")
                
                self.model_path = model_path
        
        # モデルの作成とロード
        self.model = SuperPointNet()
        
        try:
            # PyTorchモデルをロード
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"SuperPoint model loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SuperPoint model: {e}")
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """画像の前処理"""
        # グレースケール変換と正規化
        image_tensor = normalize_image(image)
        return image_tensor
    
    def _postprocess_output(self, output: tuple, image_shape: tuple) -> FeatureSet:
        """モデル出力の後処理"""
        semi, desc = output
        
        # キーポイント検出
        keypoints, scores = self._extract_keypoints(semi, image_shape)
        
        # 記述子の抽出
        descriptors = self._extract_descriptors(desc, keypoints, image_shape)
        
        # NMS適用
        if self.nms_radius > 0:
            keypoints_list = [(kp.x, kp.y) for kp in keypoints]
            scores_list = [kp.response for kp in keypoints]
            
            filtered_keypoints, filtered_scores = nms_keypoints(
                keypoints_list, scores_list, self.nms_radius, self.max_keypoints
            )
            
            # KeyPointオブジェクトを再構築
            filtered_kps = []
            for i, ((x, y), score) in enumerate(zip(filtered_keypoints, filtered_scores)):
                # 新しいKeyPointオブジェクトを作成
                from .base import KeyPoint
                kp = KeyPoint(
                    x=float(x),
                    y=float(y),
                    response=float(score),
                    angle=-1.0,  # SuperPointは方向情報なし
                    scale=1.0,   # デフォルトスケール
                    octave=0,
                    class_id=0
                )
                filtered_kps.append(kp)
            keypoints = filtered_kps
            
            # 記述子も対応する分だけ抽出
            if len(descriptors) > 0:
                descriptors = descriptors[:len(keypoints)]
        
        # 最大キーポイント数制限
        if self.max_keypoints > 0 and len(keypoints) > self.max_keypoints:
            # スコアでソート
            sorted_indices = sorted(range(len(keypoints)), 
                                  key=lambda i: keypoints[i].response, reverse=True)
            keypoints = [keypoints[i] for i in sorted_indices[:self.max_keypoints]]
            if len(descriptors) > 0:
                descriptors = descriptors[sorted_indices[:self.max_keypoints]]
        
        print(f"SuperPoint: Final FeatureSet with {len(keypoints)} keypoints")
        if len(keypoints) > 0:
            print(f"SuperPoint: First few keypoints: {[(kp.x, kp.y, kp.response) for kp in keypoints[:3]]}")
        
        return FeatureSet(
            keypoints=keypoints,
            descriptors=descriptors,
            image_shape=image_shape
        )
    
    def _extract_keypoints(self, semi: torch.Tensor, image_shape: tuple) -> tuple:
        """キーポイントを抽出"""
        # semi: [1, 65, H/8, W/8]
        batch_size, _, hc, wc = semi.shape
        
        # Softmax to get probabilities
        # 65 = 8*8 + 1 (no keypoint)
        semi = semi.softmax(dim=1)
        
        # Remove no-keypoint channel
        nodust = semi[:, :-1, :, :]  # [1, 64, H/8, W/8]
        
        # Reshape to [1, 8, 8, H/8, W/8]
        nodust = nodust.view(batch_size, 8, 8, hc, wc)
        
        # Permute to [1, H/8, W/8, 8, 8]
        nodust = nodust.permute(0, 3, 4, 1, 2)
        
        # Reshape to [1, H/8*8, W/8*8] = [1, H, W]
        heatmap = nodust.reshape(batch_size, hc*8, wc*8)
        
        # Extract keypoints
        keypoints = []
        scores = []
        
        heatmap_np = heatmap[0].cpu().numpy()
        h, w = image_shape
        
        # Find peaks above threshold
        valid_mask = heatmap_np > self.keypoint_threshold
        
        if self.remove_borders > 0:
            valid_mask[:self.remove_borders, :] = False
            valid_mask[-self.remove_borders:, :] = False
            valid_mask[:, :self.remove_borders] = False
            valid_mask[:, -self.remove_borders:] = False
        
        # 有効な点を抽出
        ys, xs = np.where(valid_mask)
        scores_np = heatmap_np[ys, xs]
        
        # デバッグ情報を追加
        print(f"SuperPoint: Found {len(xs)} candidate keypoints above threshold {self.keypoint_threshold}")
        
        if len(scores_np) > 0:
            print(f"SuperPoint: Score range: {scores_np.min():.4f} - {scores_np.max():.4f}")
        else:
            print(f"SuperPoint: No keypoints found! Max score in heatmap: {heatmap_np.max():.4f}")
            
            # 閾値を下げて再試行
            if heatmap_np.max() > 0:
                adaptive_threshold = heatmap_np.max() * 0.1  # 最大値の10%
                print(f"SuperPoint: Trying adaptive threshold: {adaptive_threshold:.4f}")
                valid_mask = heatmap_np > adaptive_threshold
                
                if self.remove_borders > 0:
                    valid_mask[:self.remove_borders, :] = False
                    valid_mask[-self.remove_borders:, :] = False
                    valid_mask[:, :self.remove_borders] = False
                    valid_mask[:, -self.remove_borders:] = False
                
                ys, xs = np.where(valid_mask)
                scores_np = heatmap_np[ys, xs]
                print(f"SuperPoint: With adaptive threshold, found {len(xs)} keypoints")
        
        # KeyPointオブジェクトに変換
        if len(xs) > 0:
            keypoints = tensor_to_keypoints(
                torch.from_numpy(np.stack([xs, ys], axis=1).astype(np.float32)),
                torch.from_numpy(scores_np.astype(np.float32))
            )
        else:
            keypoints = []
        
        print(f"SuperPoint: Created {len(keypoints)} KeyPoint objects")
        return keypoints, scores_np
    
    def _extract_descriptors(self, desc: torch.Tensor, keypoints: list, image_shape: tuple) -> np.ndarray:
        """記述子を抽出"""
        if len(keypoints) == 0:
            return np.array([])
        
        # desc: [1, 256, H/8, W/8]
        desc = desc[0]  # [256, H/8, W/8]
        
        # キーポイント座標を8で割る（ダウンサンプリング係数）
        keypoint_coords = []
        for kp in keypoints:
            x_scaled = kp.x / 8.0
            y_scaled = kp.y / 8.0
            keypoint_coords.append([x_scaled, y_scaled])
        
        keypoint_coords = np.array(keypoint_coords)
        
        # バイリニア補間で記述子を抽出
        descriptors = []
        for coord in keypoint_coords:
            x, y = coord
            
            # 境界チェック
            if x < 0 or y < 0 or x >= desc.shape[2] or y >= desc.shape[1]:
                # 境界外の場合はゼロ記述子
                descriptors.append(np.zeros(256))
                continue
            
            # バイリニア補間
            x0, y0 = int(x), int(y)
            x1, y1 = min(x0 + 1, desc.shape[2] - 1), min(y0 + 1, desc.shape[1] - 1)
            
            dx, dy = x - x0, y - y0
            
            # 4つの近傍点の記述子を取得
            desc_tl = desc[:, y0, x0].cpu().numpy()  # top-left
            desc_tr = desc[:, y0, x1].cpu().numpy()  # top-right
            desc_bl = desc[:, y1, x0].cpu().numpy()  # bottom-left
            desc_br = desc[:, y1, x1].cpu().numpy()  # bottom-right
            
            # バイリニア補間
            desc_interpolated = (
                desc_tl * (1 - dx) * (1 - dy) +
                desc_tr * dx * (1 - dy) +
                desc_bl * (1 - dx) * dy +
                desc_br * dx * dy
            )
            
            descriptors.append(desc_interpolated)
        
        return np.array(descriptors, dtype=np.float32)
    
    def extract(self, image: np.ndarray) -> FeatureSet:
        """画像から特徴量を抽出"""
        if not self.is_loaded:
            self.load_model()
        
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        
        print(f"SuperPoint: Processing image of size {image.shape}")
        
        # 大きい画像の場合はリサイズ
        original_shape = image.shape[:2]
        max_dimension = 1024  # 最大サイズを制限
        
        if max(original_shape) > max_dimension:
            scale_factor = max_dimension / max(original_shape)
            new_height = int(original_shape[0] * scale_factor)
            new_width = int(original_shape[1] * scale_factor)
            
            print(f"SuperPoint: Resizing image from {original_shape} to ({new_height}, {new_width})")
            resized_image = cv2.resize(image, (new_width, new_height))
        else:
            resized_image = image
            scale_factor = 1.0
        
        print(f"SuperPoint: Starting preprocessing...")
        # 前処理
        input_tensor = self._preprocess_image(resized_image)
        input_tensor = input_tensor.to(self.device)
        
        print(f"SuperPoint: Input tensor shape: {input_tensor.shape}")
        print(f"SuperPoint: Starting inference...")
        
        # 推論
        self.model.eval()
        with torch.no_grad():
            semi, desc = self.model(input_tensor)
        
        print(f"SuperPoint: Inference completed. Output shapes: semi={semi.shape}, desc={desc.shape}")
        
        # 後処理
        feature_set = self._postprocess_output((semi, desc), resized_image.shape[:2])
        
        # 座標を元の画像サイズにスケールバック
        if scale_factor != 1.0:
            print(f"SuperPoint: Scaling coordinates back by factor {1/scale_factor}")
            for kp in feature_set.keypoints:
                kp.x = kp.x / scale_factor
                kp.y = kp.y / scale_factor
            
            # 元の画像サイズに更新
            feature_set.image_shape = original_shape
        
        return feature_set
    
    def get_config(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        config = super().get_config()
        config.update({
            'nms_radius': self.nms_radius,
            'keypoint_threshold': self.keypoint_threshold,
            'max_keypoints': self.max_keypoints,
            'remove_borders': self.remove_borders,
            'extractor_type': 'superpoint'
        })
        return config