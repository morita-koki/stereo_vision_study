"""
Feature Extractor Module

This module provides a unified interface for various feature extraction algorithms
including traditional methods (SIFT, ORB, SURF, BRIEF) and support for future
learning-based methods.

Basic Usage:
    from stereo_vision.feature_extractor import FeatureExtractorFactory
    
    # Create a SIFT extractor
    extractor = FeatureExtractorFactory.create('sift', n_features=1000)
    
    # Extract features from an image
    features = extractor.extract(image)
    
    # Print statistics
    print(f"Detected {len(features)} features")

Advanced Usage:
    from stereo_vision.feature_extractor import (
        FeatureExtractorFactory, 
        ConfigManager,
        load_image,
        draw_keypoints
    )
    
    # Load configuration
    config_manager = ConfigManager()
    config_manager.load_from_json('config.json')
    
    # Create extractor with configuration
    extractor = FeatureExtractorFactory.create(
        config_manager.config.extractor_type,
        **config_manager.get_extractor_config()
    )
    
    # Process image
    image = load_image('image.jpg')
    features = extractor.extract(image)
    
    # Visualize results
    result_image = draw_keypoints(image, features)
"""

from .base import (
    BaseFeatureExtractor,
    KeyPoint,
    FeatureSet
)

from .traditional import (
    SIFTExtractor,
    ORBExtractor,
    SURFExtractor,
    BRIEFExtractor
)

from .factory import FeatureExtractorFactory

from .config import (
    FeatureExtractorConfig,
    SIFTConfig,
    ORBConfig,
    SURFConfig,
    BRIEFConfig,
    ConfigManager,
    create_default_config,
    load_config
)

from .utils import (
    load_image,
    draw_keypoints,
    save_keypoints_image,
    filter_keypoints_by_response,
    filter_keypoints_by_region,
    limit_keypoints,
    compute_feature_statistics,
    print_feature_statistics
)

# Version information
__version__ = "0.1.0"
__author__ = "stereo-vision"

# Export list
__all__ = [
    # Base classes
    'BaseFeatureExtractor',
    'KeyPoint',
    'FeatureSet',
    
    # Traditional extractors
    'SIFTExtractor',
    'ORBExtractor',
    'SURFExtractor',
    'BRIEFExtractor',
    
    # Factory
    'FeatureExtractorFactory',
    
    # Configuration
    'FeatureExtractorConfig',
    'SIFTConfig',
    'ORBConfig',
    'SURFConfig',
    'BRIEFConfig',
    'ConfigManager',
    'create_default_config',
    'load_config',
    
    # Utilities
    'load_image',
    'draw_keypoints',
    'save_keypoints_image',
    'filter_keypoints_by_response',
    'filter_keypoints_by_region',
    'limit_keypoints',
    'compute_feature_statistics',
    'print_feature_statistics',
    
    # Version
    '__version__',
    '__author__'
]

# Convenience functions
def create_extractor(extractor_type: str, **kwargs) -> BaseFeatureExtractor:
    """
    Convenience function to create a feature extractor
    
    Args:
        extractor_type: Type of extractor ('sift', 'orb', 'surf', 'brief')
        **kwargs: Configuration parameters
        
    Returns:
        BaseFeatureExtractor: Created extractor instance
    """
    return FeatureExtractorFactory.create(extractor_type, **kwargs)


def get_available_extractors() -> list:
    """
    Get list of available feature extractors
    
    Returns:
        list: List of available extractor names
    """
    return FeatureExtractorFactory.get_available_extractors()


def quick_extract(image, extractor_type: str = 'sift', **kwargs) -> FeatureSet:
    """
    Quick feature extraction with default parameters
    
    Args:
        image: Input image (numpy array)
        extractor_type: Type of extractor to use
        **kwargs: Additional extractor parameters
        
    Returns:
        FeatureSet: Extracted features
    """
    extractor = create_extractor(extractor_type, **kwargs)
    return extractor.extract(image)


# Module-level convenience aliases
create = create_extractor
extract = quick_extract
available_extractors = get_available_extractors