import numpy as np
import cv2
from typing import List, Tuple, Optional, Union
from pathlib import Path

from .base import FeatureSet, KeyPoint


def load_image(image_path: Union[str, Path], 
               color_mode: str = 'grayscale') -> np.ndarray:
    """
    画像ファイルを読み込む
    
    Args:
        image_path: 画像ファイルのパス
        color_mode: 色モード ('grayscale', 'color', 'unchanged')
        
    Returns:
        np.ndarray: 読み込まれた画像
        
    Raises:
        FileNotFoundError: 画像ファイルが見つからない場合
        ValueError: 画像の読み込みに失敗した場合
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    color_flags = {
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'color': cv2.IMREAD_COLOR,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    
    if color_mode not in color_flags:
        raise ValueError(f"Invalid color_mode: {color_mode}. "
                        f"Available modes: {list(color_flags.keys())}")
    
    image = cv2.imread(str(image_path), color_flags[color_mode])
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def draw_keypoints(image: np.ndarray, 
                   feature_set: FeatureSet,
                   color: Tuple[int, int, int] = (0, 255, 0),
                   draw_rich_keypoints: bool = False) -> np.ndarray:
    """
    特徴点を画像に描画
    
    Args:
        image: 入力画像
        feature_set: 特徴点セット
        color: 描画色 (B, G, R)
        draw_rich_keypoints: 方向と大きさを描画するかどうか
        
    Returns:
        np.ndarray: 特徴点が描画された画像
    """
    if feature_set.is_empty():
        return image.copy()
    
    # KeyPointをOpenCV形式に変換
    cv_keypoints = []
    for kp in feature_set.keypoints:
        cv_kp = cv2.KeyPoint(
            x=kp.x,
            y=kp.y,
            size=kp.scale,
            angle=kp.angle,
            response=kp.response,
            octave=kp.octave,
            class_id=kp.class_id
        )
        cv_keypoints.append(cv_kp)
    
    # 画像をカラーに変換（必要に応じて）
    if len(image.shape) == 2:
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output_image = image.copy()
    
    # 特徴点を描画
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if draw_rich_keypoints else cv2.DRAW_MATCHES_FLAGS_DEFAULT
    
    return cv2.drawKeypoints(output_image, cv_keypoints, None, color, flags)


def save_keypoints_image(image: np.ndarray,
                        feature_set: FeatureSet,
                        output_path: Union[str, Path],
                        color: Tuple[int, int, int] = (0, 255, 0),
                        draw_rich_keypoints: bool = False):
    """
    特徴点が描画された画像を保存
    
    Args:
        image: 入力画像
        feature_set: 特徴点セット
        output_path: 出力ファイルのパス
        color: 描画色 (B, G, R)
        draw_rich_keypoints: 方向と大きさを描画するかどうか
    """
    output_image = draw_keypoints(image, feature_set, color, draw_rich_keypoints)
    cv2.imwrite(str(output_path), output_image)


def filter_keypoints_by_response(feature_set: FeatureSet,
                                min_response: float) -> FeatureSet:
    """
    応答値による特徴点のフィルタリング
    
    Args:
        feature_set: 入力特徴点セット
        min_response: 最小応答値
        
    Returns:
        FeatureSet: フィルタリングされた特徴点セット
    """
    filtered_indices = []
    filtered_keypoints = []
    
    for i, kp in enumerate(feature_set.keypoints):
        if kp.response >= min_response:
            filtered_indices.append(i)
            filtered_keypoints.append(kp)
    
    # 記述子もフィルタリング
    if len(feature_set.descriptors) > 0:
        filtered_descriptors = feature_set.descriptors[filtered_indices]
    else:
        filtered_descriptors = np.array([])
    
    return FeatureSet(
        keypoints=filtered_keypoints,
        descriptors=filtered_descriptors,
        image_shape=feature_set.image_shape
    )


def filter_keypoints_by_region(feature_set: FeatureSet,
                              region: Tuple[int, int, int, int]) -> FeatureSet:
    """
    領域による特徴点のフィルタリング
    
    Args:
        feature_set: 入力特徴点セット
        region: 領域 (x, y, width, height)
        
    Returns:
        FeatureSet: フィルタリングされた特徴点セット
    """
    x, y, w, h = region
    filtered_indices = []
    filtered_keypoints = []
    
    for i, kp in enumerate(feature_set.keypoints):
        if x <= kp.x < x + w and y <= kp.y < y + h:
            filtered_indices.append(i)
            filtered_keypoints.append(kp)
    
    # 記述子もフィルタリング
    if len(feature_set.descriptors) > 0:
        filtered_descriptors = feature_set.descriptors[filtered_indices]
    else:
        filtered_descriptors = np.array([])
    
    return FeatureSet(
        keypoints=filtered_keypoints,
        descriptors=filtered_descriptors,
        image_shape=feature_set.image_shape
    )


def limit_keypoints(feature_set: FeatureSet,
                   max_keypoints: int) -> FeatureSet:
    """
    特徴点数の制限（応答値でソート）
    
    Args:
        feature_set: 入力特徴点セット
        max_keypoints: 最大特徴点数
        
    Returns:
        FeatureSet: 制限された特徴点セット
    """
    if len(feature_set.keypoints) <= max_keypoints:
        return feature_set
    
    # 応答値でソート
    keypoints_with_indices = [(i, kp) for i, kp in enumerate(feature_set.keypoints)]
    keypoints_with_indices.sort(key=lambda x: x[1].response, reverse=True)
    
    # 上位の特徴点を選択
    selected_indices = [i for i, _ in keypoints_with_indices[:max_keypoints]]
    selected_keypoints = [kp for _, kp in keypoints_with_indices[:max_keypoints]]
    
    # 記述子も制限
    if len(feature_set.descriptors) > 0:
        limited_descriptors = feature_set.descriptors[selected_indices]
    else:
        limited_descriptors = np.array([])
    
    return FeatureSet(
        keypoints=selected_keypoints,
        descriptors=limited_descriptors,
        image_shape=feature_set.image_shape
    )


def compute_feature_statistics(feature_set: FeatureSet) -> dict:
    """
    特徴点の統計情報を計算
    
    Args:
        feature_set: 特徴点セット
        
    Returns:
        dict: 統計情報
    """
    if feature_set.is_empty():
        return {
            'num_keypoints': 0,
            'descriptor_size': 0,
            'mean_response': 0,
            'std_response': 0,
            'mean_scale': 0,
            'std_scale': 0
        }
    
    responses = [kp.response for kp in feature_set.keypoints]
    scales = [kp.scale for kp in feature_set.keypoints]
    
    stats = {
        'num_keypoints': len(feature_set.keypoints),
        'descriptor_size': feature_set.descriptors.shape[1] if len(feature_set.descriptors) > 0 else 0,
        'mean_response': np.mean(responses),
        'std_response': np.std(responses),
        'mean_scale': np.mean(scales),
        'std_scale': np.std(scales),
        'min_response': np.min(responses),
        'max_response': np.max(responses),
        'min_scale': np.min(scales),
        'max_scale': np.max(scales)
    }
    
    return stats


def print_feature_statistics(feature_set: FeatureSet, title: str = "Feature Statistics"):
    """
    特徴点の統計情報を表示
    
    Args:
        feature_set: 特徴点セット
        title: 表示タイトル
    """
    stats = compute_feature_statistics(feature_set)
    
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Number of keypoints: {stats['num_keypoints']}")
    print(f"Descriptor size: {stats['descriptor_size']}")
    print(f"Response: {stats['mean_response']:.4f} ± {stats['std_response']:.4f}")
    print(f"Scale: {stats['mean_scale']:.4f} ± {stats['std_scale']:.4f}")
    
    if stats['num_keypoints'] > 0:
        print(f"Response range: [{stats['min_response']:.4f}, {stats['max_response']:.4f}]")
        print(f"Scale range: [{stats['min_scale']:.4f}, {stats['max_scale']:.4f}]")