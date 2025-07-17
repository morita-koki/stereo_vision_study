from typing import Dict, Any, Optional
import numpy as np
import cv2

from .base import BaseFeatureExtractor, KeyPoint, FeatureSet


def _convert_cv_keypoints_to_keypoints(cv_keypoints: list) -> list:
    """OpenCVのKeyPointオブジェクトを独自のKeyPointに変換"""
    keypoints = []
    for kp in cv_keypoints:
        keypoints.append(KeyPoint(
            x=kp.pt[0],
            y=kp.pt[1],
            scale=kp.size,
            angle=kp.angle,
            response=kp.response,
            octave=kp.octave,
            class_id=kp.class_id
        ))
    return keypoints


class SIFTExtractor(BaseFeatureExtractor):
    """SIFT特徴量抽出器"""
    
    def __init__(self, 
                 n_features: int = 0,
                 n_octave_layers: int = 3,
                 contrast_threshold: float = 0.04,
                 edge_threshold: float = 10,
                 sigma: float = 1.6,
                 **kwargs):
        """
        SIFTExtractorの初期化
        
        Args:
            n_features: 保持する特徴点の最大数 (0 = 制限なし)
            n_octave_layers: 各オクターブのレイヤー数
            contrast_threshold: 弱い特徴を除去するための閾値
            edge_threshold: エッジ応答を除去するための閾値
            sigma: ガウシアンフィルタのシグマ値
        """
        super().__init__(**kwargs)
        self.n_features = n_features
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        
        self.detector = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
    
    def extract(self, image: np.ndarray) -> FeatureSet:
        """画像からSIFT特徴量を抽出"""
        gray = self.preprocess_image(image)
        
        cv_keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None:
            descriptors = np.array([])
        
        keypoints = _convert_cv_keypoints_to_keypoints(cv_keypoints)
        
        return FeatureSet(
            keypoints=keypoints,
            descriptors=descriptors,
            image_shape=gray.shape
        )
    
    def get_config(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        return {
            'n_features': self.n_features,
            'n_octave_layers': self.n_octave_layers,
            'contrast_threshold': self.contrast_threshold,
            'edge_threshold': self.edge_threshold,
            'sigma': self.sigma
        }


class ORBExtractor(BaseFeatureExtractor):
    """ORB特徴量抽出器"""
    
    def __init__(self,
                 n_features: int = 500,
                 scale_factor: float = 1.2,
                 n_levels: int = 8,
                 edge_threshold: int = 31,
                 first_level: int = 0,
                 wta_k: int = 2,
                 score_type: int = cv2.ORB_HARRIS_SCORE,
                 patch_size: int = 31,
                 fast_threshold: int = 20,
                 **kwargs):
        """
        ORBExtractorの初期化
        
        Args:
            n_features: 保持する特徴点の最大数
            scale_factor: 画像ピラミッドの拡大率
            n_levels: 画像ピラミッドのレベル数
            edge_threshold: エッジ閾値
            first_level: 最初のレベル
            wta_k: WTA_K値
            score_type: スコア計算方法
            patch_size: パッチサイズ
            fast_threshold: FAST閾値
        """
        super().__init__(**kwargs)
        self.n_features = n_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        self.edge_threshold = edge_threshold
        self.first_level = first_level
        self.wta_k = wta_k
        self.score_type = score_type
        self.patch_size = patch_size
        self.fast_threshold = fast_threshold
        
        self.detector = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            firstLevel=first_level,
            WTA_K=wta_k,
            scoreType=score_type,
            patchSize=patch_size,
            fastThreshold=fast_threshold
        )
    
    def extract(self, image: np.ndarray) -> FeatureSet:
        """画像からORB特徴量を抽出"""
        gray = self.preprocess_image(image)
        
        cv_keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None:
            descriptors = np.array([])
        
        keypoints = _convert_cv_keypoints_to_keypoints(cv_keypoints)
        
        return FeatureSet(
            keypoints=keypoints,
            descriptors=descriptors,
            image_shape=gray.shape
        )
    
    def get_config(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        return {
            'n_features': self.n_features,
            'scale_factor': self.scale_factor,
            'n_levels': self.n_levels,
            'edge_threshold': self.edge_threshold,
            'first_level': self.first_level,
            'wta_k': self.wta_k,
            'score_type': self.score_type,
            'patch_size': self.patch_size,
            'fast_threshold': self.fast_threshold
        }


class SURFExtractor(BaseFeatureExtractor):
    """SURF特徴量抽出器 (opencv-contrib-pythonが必要)"""
    
    def __init__(self,
                 hessian_threshold: float = 400,
                 n_octaves: int = 4,
                 n_octave_layers: int = 3,
                 extended: bool = False,
                 upright: bool = False,
                 **kwargs):
        """
        SURFExtractorの初期化
        
        Args:
            hessian_threshold: Hessian閾値
            n_octaves: オクターブ数
            n_octave_layers: オクターブレイヤー数
            extended: 拡張記述子を使用するかどうか
            upright: 回転不変性を無効にするかどうか
        """
        super().__init__(**kwargs)
        self.hessian_threshold = hessian_threshold
        self.n_octaves = n_octaves
        self.n_octave_layers = n_octave_layers
        self.extended = extended
        self.upright = upright
        
        try:
            self.detector = cv2.xfeatures2d.SURF_create(
                hessianThreshold=hessian_threshold,
                nOctaves=n_octaves,
                nOctaveLayers=n_octave_layers,
                extended=extended,
                upright=upright
            )
        except AttributeError:
            raise ImportError("SURF requires opencv-contrib-python. Install with: pip install opencv-contrib-python")
    
    def extract(self, image: np.ndarray) -> FeatureSet:
        """画像からSURF特徴量を抽出"""
        gray = self.preprocess_image(image)
        
        cv_keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None:
            descriptors = np.array([])
        
        keypoints = _convert_cv_keypoints_to_keypoints(cv_keypoints)
        
        return FeatureSet(
            keypoints=keypoints,
            descriptors=descriptors,
            image_shape=gray.shape
        )
    
    def get_config(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        return {
            'hessian_threshold': self.hessian_threshold,
            'n_octaves': self.n_octaves,
            'n_octave_layers': self.n_octave_layers,
            'extended': self.extended,
            'upright': self.upright
        }


class BRIEFExtractor(BaseFeatureExtractor):
    """BRIEF記述子抽出器 (KeyPointの検出は別途必要)"""
    
    def __init__(self,
                 descriptor_size: int = 32,
                 use_orientation: bool = False,
                 **kwargs):
        """
        BRIEFExtractorの初期化
        
        Args:
            descriptor_size: 記述子のサイズ
            use_orientation: 方向を使用するかどうか
        """
        super().__init__(**kwargs)
        self.descriptor_size = descriptor_size
        self.use_orientation = use_orientation
        
        # FASTで特徴点を検出
        self.detector = cv2.FastFeatureDetector_create()
        self.descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
            bytes=descriptor_size,
            use_orientation=use_orientation
        )
    
    def extract(self, image: np.ndarray) -> FeatureSet:
        """画像からBRIEF特徴量を抽出"""
        gray = self.preprocess_image(image)
        
        cv_keypoints = self.detector.detect(gray, None)
        cv_keypoints, descriptors = self.descriptor.compute(gray, cv_keypoints)
        
        if descriptors is None:
            descriptors = np.array([])
        
        keypoints = _convert_cv_keypoints_to_keypoints(cv_keypoints)
        
        return FeatureSet(
            keypoints=keypoints,
            descriptors=descriptors,
            image_shape=gray.shape
        )
    
    def get_config(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        return {
            'descriptor_size': self.descriptor_size,
            'use_orientation': self.use_orientation
        }