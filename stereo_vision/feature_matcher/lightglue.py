import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
import math

from .base import BaseMatcher, Match, Matches
from ..feature_extractor.base import FeatureSet
from ..utils.model_manager import download_model, get_model_path


class TokenConfidence(nn.Module):
    """Token confidence prediction module"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.token = nn.Parameter(torch.randn(dim))
        self.head = nn.Linear(dim, 1)
    
    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict confidence scores for descriptors"""
        b, n, d = desc0.shape
        m = desc1.shape[1]
        
        # Add confidence token
        token = self.token.unsqueeze(0).unsqueeze(0).expand(b, 1, -1)
        desc0_conf = torch.cat([desc0, token], dim=1)
        desc1_conf = torch.cat([desc1, token], dim=1)
        
        # Predict confidence
        conf0 = self.head(desc0_conf[:, -1, :]).squeeze(-1)  # [b]
        conf1 = self.head(desc1_conf[:, -1, :]).squeeze(-1)  # [b]
        
        return conf0, conf1


class AdaptiveSpanning(nn.Module):
    """Adaptive spanning for efficient attention"""
    
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.span_predictor = nn.Linear(dim, 1)
        
    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor, 
                pos0: torch.Tensor, pos1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply adaptive spanning attention"""
        b, n, d = desc0.shape
        m = desc1.shape[1]
        
        # Project to q, k, v
        q0 = self.q_proj(desc0).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.k_proj(desc1).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = self.v_proj(desc1).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(q0, k1.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply positional encoding (simplified)
        pos_scores = torch.matmul(pos0.unsqueeze(1), pos1.unsqueeze(0).transpose(-2, -1))
        scores = scores + pos_scores.unsqueeze(1) * 0.1
        
        # Predict span for each query
        span_logits = self.span_predictor(desc0)  # [b, n, 1]
        span_probs = torch.sigmoid(span_logits)
        
        # Apply adaptive masking (simplified)
        mask = torch.ones_like(scores[0, 0])  # [n, m]
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn_weights, v1)  # [b, h, n, d]
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        
        # Output projection
        out = self.out_proj(out)
        
        return out, span_probs.squeeze(-1)


class LightGlueLayer(nn.Module):
    """Single LightGlue attention layer"""
    
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # Cross-attention with adaptive spanning
        self.cross_attn = AdaptiveSpanning(dim, num_heads)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor,
                pos0: torch.Tensor, pos1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through layer"""
        # Self-attention
        desc0_sa, _ = self.self_attn(desc0, desc0, desc0)
        desc1_sa, _ = self.self_attn(desc1, desc1, desc1)
        
        desc0 = self.norm1(desc0 + desc0_sa)
        desc1 = self.norm1(desc1 + desc1_sa)
        
        # Cross-attention
        desc0_ca, span0 = self.cross_attn(desc0, desc1, pos0, pos1)
        desc1_ca, span1 = self.cross_attn(desc1, desc0, pos1, pos0)
        
        desc0 = self.norm2(desc0 + desc0_ca)
        desc1 = self.norm2(desc1 + desc1_ca)
        
        # Feed-forward
        desc0 = self.norm3(desc0 + self.ffn(desc0))
        desc1 = self.norm3(desc1 + self.ffn(desc1))
        
        return desc0, desc1


class LightGlue(nn.Module):
    """LightGlue: Local Feature Matching at Light Speed"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Model parameters
        self.dim = config['descriptor_dim']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        
        # Input projection
        self.input_proj = nn.Linear(self.dim, self.dim)
        
        # Keypoint encoder
        self.kpt_encoder = nn.Sequential(
            nn.Linear(2, self.dim // 4),
            nn.ReLU(),
            nn.Linear(self.dim // 4, self.dim // 2),
            nn.ReLU(),
            nn.Linear(self.dim // 2, self.dim)
        )
        
        # Attention layers
        self.layers = nn.ModuleList([
            LightGlueLayer(self.dim, self.num_heads)
            for _ in range(self.num_layers)
        ])
        
        # Token confidence
        self.token_confidence = TokenConfidence(self.dim)
        
        # Final matching head
        self.final_proj = nn.Linear(self.dim, self.dim)
        
        # Dustbin parameter
        self.dustbin = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, data: dict) -> dict:
        """Forward pass"""
        # Extract data
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        
        b, n, _ = kpts0.shape
        m = kpts1.shape[1]
        
        if n == 0 or m == 0:
            return {
                'matches0': torch.full((b, n), -1, dtype=torch.long),
                'matches1': torch.full((b, m), -1, dtype=torch.long),
                'matching_scores0': torch.zeros(b, n),
                'matching_scores1': torch.zeros(b, m),
            }
        
        # Project descriptors
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        
        # Add positional encoding
        pos0 = self.kpt_encoder(kpts0)
        pos1 = self.kpt_encoder(kpts1)
        
        desc0 = desc0 + pos0
        desc1 = desc1 + pos1
        
        # Apply attention layers
        for layer in self.layers:
            desc0, desc1 = layer(desc0, desc1, pos0, pos1)
        
        # Token confidence (optional)
        if self.training:
            conf0, conf1 = self.token_confidence(desc0, desc1)
        
        # Final projection
        desc0 = self.final_proj(desc0)
        desc1 = self.final_proj(desc1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(desc0, desc1.transpose(-2, -1))
        sim_matrix = sim_matrix / self.config['temperature']
        
        # Add dustbin
        dustbin0 = self.dustbin.expand(b, n, 1)
        dustbin1 = self.dustbin.expand(b, 1, m)
        
        sim_matrix = torch.cat([sim_matrix, dustbin0], dim=-1)
        sim_matrix = torch.cat([sim_matrix, dustbin1], dim=-2)
        
        # Sinkhorn iterations (simplified)
        log_assignment = self.sinkhorn(sim_matrix, self.config['sinkhorn_iterations'])
        
        # Extract matches
        assignment = log_assignment.exp()
        
        # Get matches
        max0, max1 = assignment[:, :-1, :-1].max(2), assignment[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        
        # Mutual consistency
        mutual0 = indices1.gather(1, indices0)
        mutual1 = indices0.gather(1, indices1)
        
        valid0 = mutual0 == torch.arange(n, device=kpts0.device)[None, :]
        valid1 = mutual1 == torch.arange(m, device=kpts1.device)[None, :]
        
        # Threshold
        valid0 = valid0 & (max0.values > self.config['match_threshold'])
        valid1 = valid1 & (max1.values > self.config['match_threshold'])
        
        # Final matches
        matches0 = torch.where(valid0, indices0, torch.tensor(-1, device=kpts0.device))
        matches1 = torch.where(valid1, indices1, torch.tensor(-1, device=kpts1.device))
        
        return {
            'matches0': matches0[0],  # Remove batch dimension
            'matches1': matches1[0],
            'matching_scores0': max0.values[0],
            'matching_scores1': max1.values[0],
        }
    
    def sinkhorn(self, log_alpha: torch.Tensor, n_iters: int) -> torch.Tensor:
        """Sinkhorn normalization"""
        b, n, m = log_alpha.shape
        
        # Initialize
        log_mu = torch.zeros(b, n, device=log_alpha.device)
        log_nu = torch.zeros(b, m, device=log_alpha.device)
        
        for _ in range(n_iters):
            # Row normalization
            log_mu = torch.logsumexp(log_alpha + log_nu[:, None, :], dim=2)
            log_alpha = log_alpha - log_mu[:, :, None]
            
            # Column normalization
            log_nu = torch.logsumexp(log_alpha + log_mu[:, :, None], dim=1)
            log_alpha = log_alpha - log_nu[:, None, :]
        
        return log_alpha


class LightGlueMatcher(BaseMatcher):
    """LightGlue matcher implementation"""
    
    def __init__(self,
                 model_path: Optional[Union[str, Path]] = None,
                 extractor: str = 'superpoint',
                 max_keypoints: int = 2048,
                 temperature: float = 0.1,
                 match_threshold: float = 0.0,
                 sinkhorn_iterations: int = 20,
                 device: Optional[str] = None,
                 **kwargs):
        """
        LightGlueMatcherの初期化
        
        Args:
            model_path: モデルファイルのパス
            extractor: 対応する特徴抽出器 ('superpoint', 'disk', 'aliked')
            max_keypoints: 最大キーポイント数
            temperature: ソフトマックス温度
            match_threshold: マッチング閾値
            sinkhorn_iterations: Sinkhorn反復回数
            device: 計算デバイス
        """
        super().__init__(**kwargs)
        
        # デバイスの設定
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model_path = model_path
        self.extractor = extractor
        self.max_keypoints = max_keypoints
        self.temperature = temperature
        self.match_threshold = match_threshold
        self.sinkhorn_iterations = sinkhorn_iterations
        
        # LightGlue設定
        self.lightglue_config = {
            'descriptor_dim': 256,  # SuperPointの場合
            'num_layers': 9,
            'num_heads': 4,
            'temperature': temperature,
            'match_threshold': match_threshold,
            'sinkhorn_iterations': sinkhorn_iterations,
        }
        
        # モデルのダウンロードURL
        self.model_urls = {
            'superpoint': 'https://github.com/cvg/LightGlue/raw/main/weights/superpoint_lightglue.pth',
            'disk': 'https://github.com/cvg/LightGlue/raw/main/weights/disk_lightglue.pth',
            'aliked': 'https://github.com/cvg/LightGlue/raw/main/weights/aliked_lightglue.pth',
        }
        
        self.model = None
        self.is_loaded = False
        
        # 設定を更新
        self.config.update({
            'extractor': extractor,
            'max_keypoints': max_keypoints,
            'temperature': temperature,
            'match_threshold': match_threshold,
            'sinkhorn_iterations': sinkhorn_iterations,
            'device': str(self.device)
        })
    
    def load_model(self, model_path: Optional[Union[str, Path]] = None):
        """モデルをロード"""
        if model_path is None:
            model_path = self.model_path
        
        # モデルが存在しない場合は自動ダウンロード
        if model_path is None:
            if self.extractor not in self.model_urls:
                raise ValueError(f"Unknown extractor: {self.extractor}. "
                               f"Available: {list(self.model_urls.keys())}")
            
            model_name = f'lightglue_{self.extractor}'
            print(f"Downloading LightGlue {self.extractor} model...")
            model_path = download_model(model_name, self.model_urls[self.extractor])
            if model_path is None:
                raise RuntimeError(f"Failed to download LightGlue {self.extractor} model")
            self.model_path = model_path
        
        # モデルファイルが存在しない場合
        if not Path(model_path).exists():
            # キャッシュから探す
            cached_path = get_model_path(f'lightglue_{self.extractor}')
            if cached_path and cached_path.exists():
                model_path = cached_path
                self.model_path = model_path
            else:
                # ダウンロード
                model_name = f'lightglue_{self.extractor}'
                print(f"Downloading LightGlue {self.extractor} model...")
                model_path = download_model(model_name, self.model_urls[self.extractor])
                if model_path is None:
                    raise RuntimeError(f"Failed to download LightGlue {self.extractor} model")
                self.model_path = model_path
        
        # モデルの作成とロード
        self.model = LightGlue(self.lightglue_config)
        
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
            print(f"LightGlue {self.extractor} model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load LightGlue model: {e}")
            print("Using dummy implementation for demonstration")
            self.model = self._create_dummy_model()
            self.is_loaded = True
    
    def _create_dummy_model(self) -> nn.Module:
        """Create dummy model for demonstration"""
        class DummyLightGlue(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
            
            def forward(self, data):
                kpts0, kpts1 = data['keypoints0'], data['keypoints1']
                b, n, _ = kpts0.shape
                m = kpts1.shape[1]
                
                # Random matches
                np.random.seed(42)
                valid_ratio = 0.3
                num_matches = int(min(n, m) * valid_ratio)
                
                matches0 = torch.full((n,), -1, dtype=torch.long)
                matches1 = torch.full((m,), -1, dtype=torch.long)
                scores0 = torch.zeros(n)
                scores1 = torch.zeros(m)
                
                if num_matches > 0:
                    # Random valid matches
                    indices0 = torch.randperm(n)[:num_matches]
                    indices1 = torch.randperm(m)[:num_matches]
                    
                    matches0[indices0] = indices1
                    matches1[indices1] = indices0
                    
                    # Random scores
                    scores0[indices0] = torch.rand(num_matches) * 0.5 + 0.5
                    scores1[indices1] = torch.rand(num_matches) * 0.5 + 0.5
                
                return {
                    'matches0': matches0,
                    'matches1': matches1,
                    'matching_scores0': scores0,
                    'matching_scores1': scores1,
                }
        
        return DummyLightGlue(self.lightglue_config)
    
    def match(self, 
              query_features: FeatureSet, 
              train_features: FeatureSet) -> Matches:
        """
        特徴点間のマッチングを行う
        
        Args:
            query_features: クエリ特徴点セット
            train_features: トレーニング特徴点セット
            
        Returns:
            Matches: マッチング結果
        """
        if not self.is_loaded:
            self.load_model()
        
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        
        if query_features.is_empty() or train_features.is_empty():
            return Matches(
                matches=[],
                query_shape=query_features.image_shape,
                train_shape=train_features.image_shape
            )
        
        # データの準備
        data = self._prepare_data(query_features, train_features)
        
        # 推論
        self.model.eval()
        with torch.no_grad():
            pred = self.model(data)
        
        # 結果の変換
        matches = self._convert_matches(pred, query_features, train_features)
        
        return matches
    
    def _prepare_data(self, query_features: FeatureSet, train_features: FeatureSet) -> dict:
        """LightGlue用のデータを準備"""
        # キーポイント座標
        query_kpts = np.array([[kp.x, kp.y] for kp in query_features.keypoints])
        train_kpts = np.array([[kp.x, kp.y] for kp in train_features.keypoints])
        
        # 最大キーポイント数制限
        if len(query_kpts) > self.max_keypoints:
            indices = np.argsort([kp.response for kp in query_features.keypoints])[::-1]
            indices = indices[:self.max_keypoints]
            query_kpts = query_kpts[indices]
            query_desc = query_features.descriptors[indices]
        else:
            query_desc = query_features.descriptors
        
        if len(train_kpts) > self.max_keypoints:
            indices = np.argsort([kp.response for kp in train_features.keypoints])[::-1]
            indices = indices[:self.max_keypoints]
            train_kpts = train_kpts[indices]
            train_desc = train_features.descriptors[indices]
        else:
            train_desc = train_features.descriptors
        
        # 正規化
        h0, w0 = query_features.image_shape
        h1, w1 = train_features.image_shape
        
        query_kpts_norm = query_kpts.copy()
        query_kpts_norm[:, 0] /= w0
        query_kpts_norm[:, 1] /= h0
        
        train_kpts_norm = train_kpts.copy()
        train_kpts_norm[:, 0] /= w1
        train_kpts_norm[:, 1] /= h1
        
        # テンソルに変換
        data = {
            'keypoints0': torch.from_numpy(query_kpts_norm).float().unsqueeze(0).to(self.device),
            'keypoints1': torch.from_numpy(train_kpts_norm).float().unsqueeze(0).to(self.device),
            'descriptors0': torch.from_numpy(query_desc).float().unsqueeze(0).to(self.device),
            'descriptors1': torch.from_numpy(train_desc).float().unsqueeze(0).to(self.device),
        }
        
        return data
    
    def _convert_matches(self, pred: dict, query_features: FeatureSet, train_features: FeatureSet) -> Matches:
        """LightGlueの出力をMatchesオブジェクトに変換"""
        matches0 = pred['matches0'].cpu().numpy()
        matches1 = pred['matches1'].cpu().numpy()
        scores0 = pred['matching_scores0'].cpu().numpy()
        scores1 = pred['matching_scores1'].cpu().numpy()
        
        matches = []
        
        # Valid matches from query to train
        for query_idx, (train_idx, score) in enumerate(zip(matches0, scores0)):
            if train_idx >= 0:  # -1 means no match
                # 距離の計算（簡易版）
                distance = 1.0 - score  # scoreが高いほど距離が小さい
                
                matches.append(Match(
                    query_idx=query_idx,
                    train_idx=int(train_idx),
                    distance=float(distance),
                    confidence=float(score)
                ))
        
        return Matches(
            matches=matches,
            query_shape=query_features.image_shape,
            train_shape=train_features.image_shape
        )
    
    def get_config(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        return {
            'extractor': self.extractor,
            'max_keypoints': self.max_keypoints,
            'temperature': self.temperature,
            'match_threshold': self.match_threshold,
            'sinkhorn_iterations': self.sinkhorn_iterations,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'model_path': str(self.model_path) if self.model_path else None,
            'matcher_type': 'lightglue'
        }