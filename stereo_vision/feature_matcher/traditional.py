from typing import Dict, Any, List, Optional
import numpy as np
import cv2

from .base import BaseMatcher, Match, Matches, lowe_ratio_test
from ..feature_extractor.base import FeatureSet


class BruteForceMatcher(BaseMatcher):
    """Brute Force マッチャー"""
    
    def __init__(self, 
                 norm_type: int = cv2.NORM_L2,
                 cross_check: bool = False,
                 k: int = 2,
                 ratio_threshold: float = 0.75,
                 **kwargs):
        """
        BruteForceMatcherの初期化
        
        Args:
            norm_type: 距離の計算方法 (cv2.NORM_L2, cv2.NORM_HAMMING, etc.)
            cross_check: クロスチェックを行うかどうか
            k: k-NN マッチング (1 または 2)
            ratio_threshold: Lowe's ratio testの閾値
        """
        super().__init__(**kwargs)
        self.norm_type = norm_type
        self.cross_check = cross_check
        self.k = k
        self.ratio_threshold = ratio_threshold
        
        self.matcher = cv2.BFMatcher(normType=norm_type, crossCheck=cross_check)
    
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
        if query_features.is_empty() or train_features.is_empty():
            return Matches(
                matches=[],
                query_shape=query_features.image_shape,
                train_shape=train_features.image_shape
            )
        
        if len(query_features.descriptors) == 0 or len(train_features.descriptors) == 0:
            return Matches(
                matches=[],
                query_shape=query_features.image_shape,
                train_shape=train_features.image_shape
            )
        
        # OpenCV マッチング
        if self.k == 1:
            cv_matches = self.matcher.match(
                query_features.descriptors, 
                train_features.descriptors
            )
            matches = []
            for m in cv_matches:
                matches.append(Match(
                    query_idx=m.queryIdx,
                    train_idx=m.trainIdx,
                    distance=m.distance,
                    confidence=1.0 / (1.0 + m.distance)  # 距離から信頼度を計算
                ))
        else:
            # k-NN マッチング
            cv_matches = self.matcher.knnMatch(
                query_features.descriptors, 
                train_features.descriptors, 
                k=self.k
            )
            
            # Lowe's ratio test
            good_matches = []
            for match_pair in cv_matches:
                if len(match_pair) == 2:
                    m1, m2 = match_pair
                    if m1.distance < self.ratio_threshold * m2.distance:
                        good_matches.append(Match(
                            query_idx=m1.queryIdx,
                            train_idx=m1.trainIdx,
                            distance=m1.distance,
                            confidence=1.0 / (1.0 + m1.distance)
                        ))
                elif len(match_pair) == 1:
                    m = match_pair[0]
                    good_matches.append(Match(
                        query_idx=m.queryIdx,
                        train_idx=m.trainIdx,
                        distance=m.distance,
                        confidence=1.0 / (1.0 + m.distance)
                    ))
            
            matches = good_matches
        
        return Matches(
            matches=matches,
            query_shape=query_features.image_shape,
            train_shape=train_features.image_shape
        )
    
    def get_config(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        return {
            'norm_type': self.norm_type,
            'cross_check': self.cross_check,
            'k': self.k,
            'ratio_threshold': self.ratio_threshold
        }


class FLANNMatcher(BaseMatcher):
    """FLANN (Fast Library for Approximate Nearest Neighbors) マッチャー"""
    
    def __init__(self, 
                 algorithm: str = 'kdtree',
                 trees: int = 5,
                 checks: int = 50,
                 k: int = 2,
                 ratio_threshold: float = 0.75,
                 **kwargs):
        """
        FLANNMatcherの初期化
        
        Args:
            algorithm: アルゴリズム ('kdtree' or 'lsh')
            trees: KDTreeの数 (kdtreeの場合)
            checks: チェック回数
            k: k-NN マッチング
            ratio_threshold: Lowe's ratio testの閾値
        """
        super().__init__(**kwargs)
        self.algorithm = algorithm
        self.trees = trees
        self.checks = checks
        self.k = k
        self.ratio_threshold = ratio_threshold
        
        # FLANN パラメータ設定
        if algorithm == 'kdtree':
            # SIFT, SURF などの浮動小数点記述子用
            index_params = dict(algorithm=1, trees=trees)
        elif algorithm == 'lsh':
            # ORB などのバイナリ記述子用
            index_params = dict(algorithm=6,
                               table_number=6,
                               key_size=12,
                               multi_probe_level=1)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        search_params = dict(checks=checks)
        
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
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
        if query_features.is_empty() or train_features.is_empty():
            return Matches(
                matches=[],
                query_shape=query_features.image_shape,
                train_shape=train_features.image_shape
            )
        
        if len(query_features.descriptors) == 0 or len(train_features.descriptors) == 0:
            return Matches(
                matches=[],
                query_shape=query_features.image_shape,
                train_shape=train_features.image_shape
            )
        
        # データ型確認（FLANNは float32 が必要）
        query_desc = query_features.descriptors
        train_desc = train_features.descriptors
        
        if query_desc.dtype != np.float32:
            query_desc = query_desc.astype(np.float32)
        if train_desc.dtype != np.float32:
            train_desc = train_desc.astype(np.float32)
        
        try:
            # k-NN マッチング
            cv_matches = self.matcher.knnMatch(query_desc, train_desc, k=self.k)
            
            # Lowe's ratio test
            good_matches = []
            for match_pair in cv_matches:
                if len(match_pair) == 2:
                    m1, m2 = match_pair
                    if m1.distance < self.ratio_threshold * m2.distance:
                        good_matches.append(Match(
                            query_idx=m1.queryIdx,
                            train_idx=m1.trainIdx,
                            distance=m1.distance,
                            confidence=1.0 / (1.0 + m1.distance)
                        ))
                elif len(match_pair) == 1:
                    m = match_pair[0]
                    good_matches.append(Match(
                        query_idx=m.queryIdx,
                        train_idx=m.trainIdx,
                        distance=m.distance,
                        confidence=1.0 / (1.0 + m.distance)
                    ))
            
        except cv2.error as e:
            print(f"FLANN matching failed: {e}")
            good_matches = []
        
        return Matches(
            matches=good_matches,
            query_shape=query_features.image_shape,
            train_shape=train_features.image_shape
        )
    
    def get_config(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        return {
            'algorithm': self.algorithm,
            'trees': self.trees,
            'checks': self.checks,
            'k': self.k,
            'ratio_threshold': self.ratio_threshold
        }


class TemplateMatcher(BaseMatcher):
    """テンプレートマッチング（単純な相関ベースマッチング）"""
    
    def __init__(self, 
                 method: int = cv2.TM_CCOEFF_NORMED,
                 threshold: float = 0.8,
                 **kwargs):
        """
        TemplateMatcherの初期化
        
        Args:
            method: テンプレートマッチング手法
            threshold: マッチング閾値
        """
        super().__init__(**kwargs)
        self.method = method
        self.threshold = threshold
    
    def match(self, 
              query_features: FeatureSet, 
              train_features: FeatureSet) -> Matches:
        """
        テンプレートマッチングを行う
        
        注意: この実装は簡素化されており、実際の特徴点マッチングには適していません
        """
        # 簡易実装（実際にはパッチベースマッチングを実装する必要がある）
        print("Warning: TemplateMatcher is a simplified implementation")
        
        return Matches(
            matches=[],
            query_shape=query_features.image_shape,
            train_shape=train_features.image_shape
        )
    
    def get_config(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        return {
            'method': self.method,
            'threshold': self.threshold
        }


class RatioTestMatcher(BaseMatcher):
    """Lowe's ratio testに特化したマッチャー"""
    
    def __init__(self, 
                 base_matcher: BaseMatcher,
                 ratio_threshold: float = 0.75,
                 **kwargs):
        """
        RatioTestMatcherの初期化
        
        Args:
            base_matcher: ベースとなるマッチャー
            ratio_threshold: ratio test の閾値
        """
        super().__init__(**kwargs)
        self.base_matcher = base_matcher
        self.ratio_threshold = ratio_threshold
    
    def match(self, 
              query_features: FeatureSet, 
              train_features: FeatureSet) -> Matches:
        """
        Ratio testを適用したマッチング
        
        Args:
            query_features: クエリ特徴点セット
            train_features: トレーニング特徴点セット
            
        Returns:
            Matches: マッチング結果
        """
        # まずベースマッチャーでマッチング
        base_matches = self.base_matcher.match(query_features, train_features)
        
        # ratio test は既にベースマッチャーで実装されている場合が多いので、
        # ここでは追加のフィルタリングを行う
        filtered_matches = []
        
        # 同じクエリ点に対する複数のマッチを探す
        query_to_matches = {}
        for match in base_matches.matches:
            query_idx = match.query_idx
            if query_idx not in query_to_matches:
                query_to_matches[query_idx] = []
            query_to_matches[query_idx].append(match)
        
        # 各クエリ点に対してratio testを実行
        for query_idx, matches in query_to_matches.items():
            if len(matches) >= 2:
                # 距離でソート
                matches.sort(key=lambda m: m.distance)
                m1, m2 = matches[0], matches[1]
                
                if m1.distance < self.ratio_threshold * m2.distance:
                    filtered_matches.append(m1)
            elif len(matches) == 1:
                # 1つしかマッチがない場合はそのまま採用
                filtered_matches.append(matches[0])
        
        return Matches(
            matches=filtered_matches,
            query_shape=base_matches.query_shape,
            train_shape=base_matches.train_shape
        )
    
    def get_config(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        config = self.base_matcher.get_config()
        config.update({
            'ratio_threshold': self.ratio_threshold,
            'base_matcher': str(self.base_matcher)
        })
        return config


def create_matcher_for_extractor(extractor_type: str, **kwargs) -> BaseMatcher:
    """
    特徴抽出器に適したマッチャーを作成
    
    Args:
        extractor_type: 特徴抽出器の種類
        **kwargs: マッチャーのパラメータ
        
    Returns:
        BaseMatcher: 適切なマッチャー
    """
    if extractor_type.lower() in ['sift', 'surf']:
        # 浮動小数点記述子用
        return FLANNMatcher(algorithm='kdtree', **kwargs)
    elif extractor_type.lower() in ['orb', 'brief']:
        # バイナリ記述子用
        return BruteForceMatcher(norm_type=cv2.NORM_HAMMING, **kwargs)
    else:
        # デフォルト
        return BruteForceMatcher(norm_type=cv2.NORM_L2, **kwargs)