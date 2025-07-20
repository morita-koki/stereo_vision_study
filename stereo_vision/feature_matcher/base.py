from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Union
from pathlib import Path
import numpy as np
import cv2

from ..feature_extractor.base import FeatureSet


@dataclass
class Match:
    """単一マッチの情報"""
    query_idx: int      # クエリ特徴点のインデックス
    train_idx: int      # トレーニング特徴点のインデックス
    distance: float     # マッチング距離
    confidence: float = 1.0  # 信頼度 (0.0 - 1.0)


@dataclass
class Matches:
    """マッチング結果を格納するデータ構造"""
    matches: List[Match]
    query_shape: Tuple[int, int]  # クエリ画像のサイズ
    train_shape: Tuple[int, int]  # トレーニング画像のサイズ
    
    def __len__(self) -> int:
        """マッチ数を返す"""
        return len(self.matches)
    
    def is_empty(self) -> bool:
        """マッチが空かどうかを判定"""
        return len(self.matches) == 0
    
    def get_matched_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        マッチした点の座標を取得
        
        Returns:
            tuple: (query_points, train_points) それぞれ [N, 2] の配列
        """
        if self.is_empty():
            return np.array([]), np.array([])
        
        query_points = []
        train_points = []
        
        for match in self.matches:
            query_points.append([match.query_idx, 0])  # インデックスのみ
            train_points.append([match.train_idx, 0])
        
        return np.array(query_points), np.array(train_points)
    
    def filter_by_distance(self, max_distance: float) -> 'Matches':
        """
        距離による フィルタリング
        
        Args:
            max_distance: 最大距離
            
        Returns:
            Matches: フィルタリングされたマッチ
        """
        filtered_matches = [
            match for match in self.matches 
            if match.distance <= max_distance
        ]
        
        return Matches(
            matches=filtered_matches,
            query_shape=self.query_shape,
            train_shape=self.train_shape
        )
    
    def filter_by_confidence(self, min_confidence: float) -> 'Matches':
        """
        信頼度によるフィルタリング
        
        Args:
            min_confidence: 最小信頼度
            
        Returns:
            Matches: フィルタリングされたマッチ
        """
        filtered_matches = [
            match for match in self.matches 
            if match.confidence >= min_confidence
        ]
        
        return Matches(
            matches=filtered_matches,
            query_shape=self.query_shape,
            train_shape=self.train_shape
        )
    
    def get_top_k(self, k: int) -> 'Matches':
        """
        上位k個のマッチを取得（距離の昇順）
        
        Args:
            k: 取得する数
            
        Returns:
            Matches: 上位k個のマッチ
        """
        sorted_matches = sorted(self.matches, key=lambda m: m.distance)
        top_k_matches = sorted_matches[:k]
        
        return Matches(
            matches=top_k_matches,
            query_shape=self.query_shape,
            train_shape=self.train_shape
        )


@dataclass
class DenseMatch:
    """密なマッチングの単一点"""
    query_point: Tuple[float, float]  # クエリ画像上の座標
    train_point: Tuple[float, float]  # トレーニング画像上の座標
    confidence: float = 1.0           # 信頼度


@dataclass
class DenseMatches:
    """密なマッチング結果を格納するデータ構造"""
    matches: List[DenseMatch]
    query_shape: Tuple[int, int]
    train_shape: Tuple[int, int]
    
    def __len__(self) -> int:
        """マッチ数を返す"""
        return len(self.matches)
    
    def is_empty(self) -> bool:
        """マッチが空かどうかを判定"""
        return len(self.matches) == 0
    
    def get_matched_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        マッチした点の座標を取得
        
        Returns:
            tuple: (query_points, train_points) それぞれ [N, 2] の配列
        """
        if self.is_empty():
            return np.array([]), np.array([])
        
        query_points = np.array([
            [match.query_point[0], match.query_point[1]] 
            for match in self.matches
        ])
        
        train_points = np.array([
            [match.train_point[0], match.train_point[1]] 
            for match in self.matches
        ])
        
        return query_points, train_points
    
    def filter_by_confidence(self, min_confidence: float) -> 'DenseMatches':
        """
        信頼度によるフィルタリング
        
        Args:
            min_confidence: 最小信頼度
            
        Returns:
            DenseMatches: フィルタリングされたマッチ
        """
        filtered_matches = [
            match for match in self.matches 
            if match.confidence >= min_confidence
        ]
        
        return DenseMatches(
            matches=filtered_matches,
            query_shape=self.query_shape,
            train_shape=self.train_shape
        )


class BaseMatcher(ABC):
    """特徴マッチングの抽象基底クラス"""
    
    def __init__(self, **kwargs):
        """初期化メソッド"""
        self.config = kwargs
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        現在の設定を取得
        
        Returns:
            Dict[str, Any]: 設定辞書
        """
        pass
    
    def __str__(self) -> str:
        """オブジェクトの文字列表現"""
        return f"{self.__class__.__name__}({self.get_config()})"
    
    def __repr__(self) -> str:
        """オブジェクトの詳細表現"""
        return self.__str__()


class BaseDenseMatcher(ABC):
    """密なマッチングの抽象基底クラス"""
    
    def __init__(self, **kwargs):
        """初期化メソッド"""
        self.config = kwargs
    
    @abstractmethod
    def match(self, 
              query_image: np.ndarray, 
              train_image: np.ndarray) -> DenseMatches:
        """
        画像間の密なマッチングを行う
        
        Args:
            query_image: クエリ画像
            train_image: トレーニング画像
            
        Returns:
            DenseMatches: 密なマッチング結果
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        現在の設定を取得
        
        Returns:
            Dict[str, Any]: 設定辞書
        """
        pass
    
    def __str__(self) -> str:
        """オブジェクトの文字列表現"""
        return f"{self.__class__.__name__}({self.get_config()})"
    
    def __repr__(self) -> str:
        """オブジェクトの詳細表現"""
        return self.__str__()


class BaseHybridMatcher(ABC):
    """ハイブリッドマッチャー（特徴抽出+マッチングを同時に行う）"""
    
    def __init__(self, **kwargs):
        """初期化メソッド"""
        self.config = kwargs
    
    @abstractmethod
    def extract_and_match(self, 
                         query_image: np.ndarray, 
                         train_image: np.ndarray) -> Union[Matches, DenseMatches]:
        """
        特徴抽出とマッチングを同時に行う
        
        Args:
            query_image: クエリ画像
            train_image: トレーニング画像
            
        Returns:
            Union[Matches, DenseMatches]: マッチング結果
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        現在の設定を取得
        
        Returns:
            Dict[str, Any]: 設定辞書
        """
        pass
    
    def __str__(self) -> str:
        """オブジェクトの文字列表現"""
        return f"{self.__class__.__name__}({self.get_config()})"
    
    def __repr__(self) -> str:
        """オブジェクトの詳細表現"""
        return self.__str__()


def lowe_ratio_test(matches: List[List[Match]], 
                   ratio: float = 0.75) -> List[Match]:
    """
    Lowe's ratio testを適用
    
    Args:
        matches: 各クエリ点に対するマッチのリスト（通常はk=2）
        ratio: ratio threshold
        
    Returns:
        List[Match]: フィルタリングされたマッチ
    """
    good_matches = []
    
    for match_pair in matches:
        if len(match_pair) == 2:
            m1, m2 = match_pair
            if m1.distance < ratio * m2.distance:
                good_matches.append(m1)
        elif len(match_pair) == 1:
            # 1つしかマッチがない場合はそのまま採用
            good_matches.append(match_pair[0])
    
    return good_matches


def compute_matching_statistics(matches: Union[Matches, DenseMatches]) -> Dict[str, Any]:
    """
    マッチング統計情報を計算
    
    Args:
        matches: マッチング結果
        
    Returns:
        Dict[str, Any]: 統計情報
    """
    if matches.is_empty():
        return {
            'num_matches': 0,
            'mean_distance': 0.0,
            'std_distance': 0.0,
            'mean_confidence': 0.0,
            'std_confidence': 0.0
        }
    
    if isinstance(matches, Matches):
        distances = [m.distance for m in matches.matches]
        confidences = [m.confidence for m in matches.matches]
        
        return {
            'num_matches': len(matches),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
    
    elif isinstance(matches, DenseMatches):
        confidences = [m.confidence for m in matches.matches]
        
        return {
            'num_matches': len(matches),
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
    
    else:
        raise ValueError(f"Unknown matches type: {type(matches)}")


def print_matching_statistics(matches: Union[Matches, DenseMatches], 
                            title: str = "Matching Statistics"):
    """
    マッチング統計情報を表示
    
    Args:
        matches: マッチング結果
        title: 表示タイトル
    """
    stats = compute_matching_statistics(matches)
    
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Number of matches: {stats['num_matches']}")
    print(f"Mean confidence: {stats['mean_confidence']:.4f} ± {stats['std_confidence']:.4f}")
    
    if 'mean_distance' in stats:
        print(f"Mean distance: {stats['mean_distance']:.4f} ± {stats['std_distance']:.4f}")
        print(f"Distance range: [{stats['min_distance']:.4f}, {stats['max_distance']:.4f}]")
    
    if stats['num_matches'] > 0:
        print(f"Confidence range: [{stats['min_confidence']:.4f}, {stats['max_confidence']:.4f}]")


def draw_matches(query_image: np.ndarray,
                train_image: np.ndarray, 
                matches: Matches,
                query_features: FeatureSet = None,
                train_features: FeatureSet = None,
                max_matches: int = 50,
                line_thickness: int = 1,
                point_size: int = 3) -> np.ndarray:
    """
    マッチング結果を画像に描画
    
    Args:
        query_image: クエリ画像
        train_image: トレーニング画像
        matches: マッチング結果
        query_features: クエリ特徴点（描画用）
        train_features: トレーニング特徴点（描画用）
        max_matches: 表示する最大マッチ数
        line_thickness: 線の太さ
        point_size: 点のサイズ
    
    Returns:
        np.ndarray: マッチング結果が描画された画像
    """
    # 画像をカラーに変換
    if len(query_image.shape) == 2:
        query_img = cv2.cvtColor(query_image, cv2.COLOR_GRAY2BGR)
    else:
        query_img = query_image.copy()
    
    if len(train_image.shape) == 2:
        train_img = cv2.cvtColor(train_image, cv2.COLOR_GRAY2BGR)
    else:
        train_img = train_image.copy()
    
    # 画像を横に結合
    h1, w1 = query_img.shape[:2]
    h2, w2 = train_img.shape[:2]
    
    # 高さを統一
    if h1 != h2:
        if h1 > h2:
            train_img = cv2.resize(train_img, (w2, h1))
            h2 = h1
        else:
            query_img = cv2.resize(query_img, (w1, h2))
            h1 = h2
    
    # 画像を結合
    combined = np.hstack([query_img, train_img])
    
    # マッチを描画（最大数まで）
    match_list = matches.matches[:max_matches] if len(matches.matches) > max_matches else matches.matches
    
    for i, match in enumerate(match_list):
        # 色を信頼度に基づいて決定
        if hasattr(match, 'confidence'):
            # 信頼度に基づく色 (赤=高, 青=低)
            confidence = max(0.0, min(1.0, match.confidence))
            color = (
                int(255 * (1 - confidence)),  # Blue
                int(255 * confidence * 0.5),  # Green
                int(255 * confidence)         # Red
            )
        else:
            # デフォルト色（赤）
            color = (0, 0, 255)
        
        # 座標を取得
        query_pt = (int(match.query_x), int(match.query_y))
        train_pt = (int(match.train_x + w1), int(match.train_y))  # x座標をオフセット
        
        # 点を描画
        cv2.circle(combined, query_pt, point_size, color, -1)
        cv2.circle(combined, train_pt, point_size, color, -1)
        
        # 線を描画
        cv2.line(combined, query_pt, train_pt, color, line_thickness)
    
    return combined


def draw_matches_advanced(query_image: np.ndarray,
                         train_image: np.ndarray,
                         matches: Matches,
                         show_confidence: bool = False,
                         show_indices: bool = False,
                         filter_by_confidence: float = None,
                         max_matches: int = 50) -> np.ndarray:
    """
    高度なマッチング描画（信頼度表示、フィルタリングなど）
    
    Args:
        query_image: クエリ画像
        train_image: トレーニング画像
        matches: マッチング結果
        show_confidence: 信頼度を表示するかどうか
        show_indices: インデックスを表示するかどうか
        filter_by_confidence: 信頼度による最小閾値
        max_matches: 表示する最大マッチ数
    
    Returns:
        np.ndarray: 描画された画像
    """
    # 信頼度によるフィルタリング
    filtered_matches = matches.matches
    if filter_by_confidence is not None:
        filtered_matches = [m for m in matches.matches 
                          if hasattr(m, 'confidence') and m.confidence >= filter_by_confidence]
    
    # 信頼度でソート
    if filtered_matches and hasattr(filtered_matches[0], 'confidence'):
        filtered_matches = sorted(filtered_matches, key=lambda m: m.confidence, reverse=True)
    
    # 最大数まで制限
    filtered_matches = filtered_matches[:max_matches]
    
    # 基本描画
    combined = draw_matches(query_image, train_image, 
                          Matches(filtered_matches, matches.query_shape, matches.train_shape))
    
    # 追加情報を描画
    w1 = query_image.shape[1]
    
    for i, match in enumerate(filtered_matches):
        query_pt = (int(match.query_x), int(match.query_y))
        train_pt = (int(match.train_x + w1), int(match.train_y))
        
        # 信頼度を表示
        if show_confidence and hasattr(match, 'confidence'):
            text = f"{match.confidence:.3f}"
            cv2.putText(combined, text, (query_pt[0] + 5, query_pt[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # インデックスを表示
        if show_indices:
            text = str(i)
            cv2.putText(combined, text, (query_pt[0] - 5, query_pt[1] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return combined


def save_matches_image(query_image: np.ndarray,
                      train_image: np.ndarray,
                      matches: Matches,
                      output_path: Union[str, Path],
                      max_matches: int = 50,
                      show_confidence: bool = False) -> bool:
    """
    マッチング結果画像を保存
    
    Args:
        query_image: クエリ画像
        train_image: トレーニング画像  
        matches: マッチング結果
        output_path: 出力ファイルパス
        max_matches: 表示する最大マッチ数
        show_confidence: 信頼度を表示するかどうか
    
    Returns:
        bool: 保存成功かどうか
    """
    try:
        if show_confidence:
            result_image = draw_matches_advanced(
                query_image, train_image, matches,
                show_confidence=True, max_matches=max_matches
            )
        else:
            result_image = draw_matches(
                query_image, train_image, matches, max_matches=max_matches
            )
        
        cv2.imwrite(str(output_path), result_image)
        return True
    except Exception as e:
        print(f"Failed to save matches image: {e}")
        return False