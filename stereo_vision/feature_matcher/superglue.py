import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path

from .base import BaseMatcher, Match, Matches
from ..feature_extractor.base import FeatureSet
from ..utils.model_manager import download_model, get_model_path


def MLP(channels: list, do_bn: bool = True) -> nn.Sequential:
    """Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts: torch.Tensor, image_shape: Tuple[int, int]) -> torch.Tensor:
    """Normalize keypoints coordinates to [-1, 1]"""
    height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""
    
    def __init__(self, feature_dim: int, layers: list):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)
    
    def forward(self, kpts: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


class AttentionalPropagation(nn.Module):
    """Attention-based message passing"""
    
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.dim = feature_dim
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        
        self.merge = nn.Linear(feature_dim, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.merge.weight)
    
    def forward(self, query: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query_dim = query.size(1)
        source_dim = source.size(1)
        
        # Multi-head attention
        query = query.view(batch_dim, query_dim, self.num_heads, self.dim // self.num_heads)
        source = source.view(batch_dim, source_dim, self.num_heads, self.dim // self.num_heads)
        
        query = query.transpose(1, 2)
        source = source.transpose(1, 2)
        
        # Compute attention
        q = self.q_proj(query)
        k = self.k_proj(source)
        v = self.v_proj(source)
        
        # Attention weights
        attention = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        message = torch.matmul(attention, v)
        
        # Merge heads
        message = message.transpose(1, 2).contiguous()
        message = message.view(batch_dim, query_dim, self.dim)
        message = self.merge(message)
        
        # Update
        update = query.transpose(1, 2).contiguous().view(batch_dim, query_dim, self.dim)
        return self.mlp(torch.cat([update, message], dim=2))


class AttentionalGNN(nn.Module):
    """Attentional Graph Neural Network"""
    
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4) for _ in layer_names
        ])
        self.names = layer_names
    
    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'self':
                desc0 = layer(desc0, desc0)
                desc1 = layer(desc1, desc1)
            elif name == 'cross':
                desc0 = layer(desc0, desc1)
                desc1 = layer(desc1, desc0)
            else:
                raise ValueError(f"Unknown layer name: {name}")
        
        return desc0, desc1


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, 
                           iters: int) -> torch.Tensor:
    """Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: float, iters: int) -> torch.Tensor:
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)
    
    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)
    
    couplings = torch.cat([torch.cat([scores, bins0], -1),
                          torch.cat([bins1, alpha], -1)], 1)
    
    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)
    
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm[None, None, None]
    
    return Z


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Point embedding
        self.kenc = KeypointEncoder(config['descriptor_dim'], config['keypoint_encoder'])
        
        # GNN layers
        self.gnn = AttentionalGNN(config['descriptor_dim'], config['GNN_layers'])
        
        # Final projection
        self.final_proj = nn.Conv1d(config['descriptor_dim'], config['descriptor_dim'],
                                  kernel_size=1, bias=True)
        
        # Bin score
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
    
    def forward(self, data: dict) -> dict:
        """Forward pass"""
        # Extract features
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        
        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts0.new_zeros(shape0)[0],
                'matching_scores1': kpts1.new_zeros(shape1)[0],
            }
        
        # Keypoint normalization
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        
        # Keypoint MLP encoder
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])
        
        # Multi-layer Transformer network
        desc0, desc1 = self.gnn(desc0, desc1)
        
        # Final MLP projection
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        
        # Compute matching descriptor distance
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5
        
        # Run the optimal transport
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])
        
        # Get the matches with score above "match_threshold"
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = indices1.gather(1, indices0)
        mutual1 = indices0.gather(1, indices1)
        zero = indices0.new_tensor(0)
        mscores0 = max0.values
        mscores1 = max1.values
        valid0 = mutual0 == torch.arange(indices0.shape[1], device=indices0.device)[None]
        valid1 = mutual1 == torch.arange(indices1.shape[1], device=indices1.device)[None]
        valid0 = valid0 & (mscores0 > self.config['match_threshold'])
        valid1 = valid1 & (mscores1 > self.config['match_threshold'])
        
        # Discard invalid matches
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        
        return {
            'matches0': indices0[0],  # use the first element of the batch
            'matches1': indices1[0],
            'matching_scores0': mscores0[0],
            'matching_scores1': mscores1[0],
        }


class SuperGlueMatcher(BaseMatcher):
    """SuperGlue matcher implementation"""
    
    def __init__(self,
                 model_path: Optional[Union[str, Path]] = None,
                 weights: str = 'indoor',
                 sinkhorn_iterations: int = 100,
                 match_threshold: float = 0.2,
                 device: Optional[str] = None,
                 **kwargs):
        """
        SuperGlueMatcherの初期化
        
        Args:
            model_path: モデルファイルのパス
            weights: 事前学習重み ('indoor' or 'outdoor')
            sinkhorn_iterations: Sinkhorn反復回数
            match_threshold: マッチング閾値
            device: 計算デバイス
        """
        super().__init__(**kwargs)
        
        # デバイスの設定
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model_path = model_path
        self.weights = weights
        self.sinkhorn_iterations = sinkhorn_iterations
        self.match_threshold = match_threshold
        
        # SuperGlue設定
        self.superglue_config = {
            'descriptor_dim': 256,
            'keypoint_encoder': [32, 64, 128, 256],
            'GNN_layers': ['self', 'cross'] * 9,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
        
        # モデルのダウンロードURL
        self.model_urls = {
            'indoor': 'https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_indoor.pth',
            'outdoor': 'https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_outdoor.pth'
        }
        
        self.model = None
        self.is_loaded = False
        
        # 設定を更新
        self.config.update({
            'weights': weights,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
            'device': str(self.device)
        })
    
    def load_model(self, model_path: Optional[Union[str, Path]] = None):
        """モデルをロード"""
        if model_path is None:
            model_path = self.model_path
        
        # モデルが存在しない場合は自動ダウンロード
        if model_path is None:
            model_name = f'superglue_{self.weights}'
            print(f"Downloading SuperGlue {self.weights} model...")
            model_path = download_model(model_name, self.model_urls[self.weights])
            if model_path is None:
                raise RuntimeError(f"Failed to download SuperGlue {self.weights} model")
            self.model_path = model_path
        
        # モデルファイルが存在しない場合
        if not Path(model_path).exists():
            # キャッシュから探す
            cached_path = get_model_path(f'superglue_{self.weights}')
            if cached_path and cached_path.exists():
                model_path = cached_path
                self.model_path = model_path
            else:
                # ダウンロード
                model_name = f'superglue_{self.weights}'
                print(f"Downloading SuperGlue {self.weights} model...")
                model_path = download_model(model_name, self.model_urls[self.weights])
                if model_path is None:
                    raise RuntimeError(f"Failed to download SuperGlue {self.weights} model")
                self.model_path = model_path
        
        # モデルの作成とロード
        self.model = SuperGlue(self.superglue_config)
        
        try:
            # PyTorchモデルをロード
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"SuperGlue {self.weights} model loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SuperGlue model: {e}")
    
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
        """SuperGlue用のデータを準備"""
        # キーポイント座標
        query_kpts = np.array([[kp.x, kp.y] for kp in query_features.keypoints])
        train_kpts = np.array([[kp.x, kp.y] for kp in train_features.keypoints])
        
        # スコア
        query_scores = np.array([kp.response for kp in query_features.keypoints])
        train_scores = np.array([kp.response for kp in train_features.keypoints])
        
        # テンソルに変換
        data = {
            'keypoints0': torch.from_numpy(query_kpts).float().unsqueeze(0).to(self.device),
            'keypoints1': torch.from_numpy(train_kpts).float().unsqueeze(0).to(self.device),
            'descriptors0': torch.from_numpy(query_features.descriptors).float().unsqueeze(0).transpose(1, 2).to(self.device),
            'descriptors1': torch.from_numpy(train_features.descriptors).float().unsqueeze(0).transpose(1, 2).to(self.device),
            'scores0': torch.from_numpy(query_scores).float().unsqueeze(0).to(self.device),
            'scores1': torch.from_numpy(train_scores).float().unsqueeze(0).to(self.device),
            'image0': torch.zeros(1, 1, *query_features.image_shape).to(self.device),
            'image1': torch.zeros(1, 1, *train_features.image_shape).to(self.device),
        }
        
        return data
    
    def _convert_matches(self, pred: dict, query_features: FeatureSet, train_features: FeatureSet) -> Matches:
        """SuperGlueの出力をMatchesオブジェクトに変換"""
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
            'weights': self.weights,
            'sinkhorn_iterations': self.sinkhorn_iterations,
            'match_threshold': self.match_threshold,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'model_path': str(self.model_path) if self.model_path else None,
            'matcher_type': 'superglue'
        }