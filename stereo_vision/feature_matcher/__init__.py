"""
Feature Matcher Module

This module provides various feature matching algorithms including traditional
methods (BruteForce, FLANN) and learning-based methods (SuperGlue, LoFTR).

Basic Usage:
    from stereo_vision.feature_matcher import BruteForceMatcher
    from stereo_vision.feature_extractor import FeatureExtractorFactory
    
    # Extract features
    extractor = FeatureExtractorFactory.create('sift')
    features1 = extractor.extract(image1)
    features2 = extractor.extract(image2)
    
    # Match features
    matcher = BruteForceMatcher()
    matches = matcher.match(features1, features2)
    
    print(f"Found {len(matches)} matches")

Advanced Usage:
    from stereo_vision.feature_matcher import SuperGlueMatcher
    from stereo_vision.feature_extractor import SuperPointExtractor
    
    # SuperPoint + SuperGlue pipeline
    extractor = SuperPointExtractor()
    matcher = SuperGlueMatcher(weights='indoor')
    
    features1 = extractor.extract(image1)
    features2 = extractor.extract(image2)
    matches = matcher.match(features1, features2)
    
    # Dense matching with LoFTR
    from stereo_vision.feature_matcher import LoFTRMatcher
    dense_matcher = LoFTRMatcher(weights='outdoor')
    dense_matches = dense_matcher.match(image1, image2)
"""

from .base import (
    BaseMatcher,
    BaseDenseMatcher,
    BaseHybridMatcher,
    Match,
    Matches,
    DenseMatch,
    DenseMatches,
    lowe_ratio_test,
    compute_matching_statistics,
    print_matching_statistics
)

from .traditional import (
    BruteForceMatcher,
    FLANNMatcher,
    TemplateMatcher,
    RatioTestMatcher,
    create_matcher_for_extractor
)

from .superglue import SuperGlueMatcher
from .loftr import LoFTRMatcher
from .lightglue import LightGlueMatcher

# Version information
__version__ = "0.1.0"
__author__ = "stereo-vision"

# Export list
__all__ = [
    # Base classes
    'BaseMatcher',
    'BaseDenseMatcher',
    'BaseHybridMatcher',
    'Match',
    'Matches',
    'DenseMatch',
    'DenseMatches',
    
    # Traditional matchers
    'BruteForceMatcher',
    'FLANNMatcher',
    'TemplateMatcher',
    'RatioTestMatcher',
    
    # Learning-based matchers
    'SuperGlueMatcher',
    'LoFTRMatcher',
    'LightGlueMatcher',
    
    # Utility functions
    'lowe_ratio_test',
    'compute_matching_statistics',
    'print_matching_statistics',
    'create_matcher_for_extractor',
    
    # Version
    '__version__',
    '__author__'
]

# Convenience functions
def create_matcher(matcher_type: str, **kwargs) -> BaseMatcher:
    """
    Convenience function to create a matcher
    
    Args:
        matcher_type: Type of matcher ('bruteforce', 'flann', 'superglue', etc.)
        **kwargs: Configuration parameters
        
    Returns:
        BaseMatcher: Created matcher instance
    """
    matcher_map = {
        'bruteforce': BruteForceMatcher,
        'bf': BruteForceMatcher,
        'flann': FLANNMatcher,
        'superglue': SuperGlueMatcher,
        'loftr': LoFTRMatcher,
        'lightglue': LightGlueMatcher,
    }
    
    if matcher_type.lower() not in matcher_map:
        available_types = list(matcher_map.keys())
        raise ValueError(f"Unknown matcher type: {matcher_type}. "
                        f"Available types: {available_types}")
    
    return matcher_map[matcher_type.lower()](**kwargs)


def get_available_matchers() -> list:
    """
    Get list of available matchers
    
    Returns:
        list: List of available matcher names
    """
    return ['bruteforce', 'flann', 'superglue', 'loftr', 'lightglue']


def quick_match(features1, features2, matcher_type: str = 'bruteforce', **kwargs):
    """
    Quick feature matching with default parameters
    
    Args:
        features1: First feature set
        features2: Second feature set
        matcher_type: Type of matcher to use
        **kwargs: Additional matcher parameters
        
    Returns:
        Matches: Matching results
    """
    matcher = create_matcher(matcher_type, **kwargs)
    return matcher.match(features1, features2)


# Module-level convenience aliases
create = create_matcher
match = quick_match
available_matchers = get_available_matchers