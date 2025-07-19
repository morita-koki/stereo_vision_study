#!/usr/bin/env python3
"""
Feature Extractor Demo Script

This script demonstrates various feature extraction methods including
traditional methods (SIFT, ORB, SURF, BRIEF) and learning-based methods
(SuperPoint, DISK, ALIKE).

Usage:
    uv run scripts/demo_feature_extractor.py [--extractor TYPE] [--image PATH]
"""

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

# Add the package to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from stereo_vision.feature_extractor import (
    FeatureExtractorFactory,
    get_available_extractors,
    print_feature_statistics
)
from stereo_vision.feature_extractor.utils import (
    draw_keypoints,
    draw_keypoints_advanced,
    save_keypoints_image,
    save_keypoints_comparison
)


def create_test_image(size=(480, 640), pattern='noise'):
    """
    Create a test image for demonstration
    
    Args:
        size: Image size as (height, width)
        pattern: Pattern type ('noise', 'checkerboard', 'circles')
    
    Returns:
        np.ndarray: Test image
    """
    height, width = size
    
    if pattern == 'noise':
        # Random noise pattern
        np.random.seed(42)
        image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    elif pattern == 'checkerboard':
        # Checkerboard pattern
        image = np.zeros((height, width), dtype=np.uint8)
        square_size = 40
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    image[i:i+square_size, j:j+square_size] = 255
    elif pattern == 'circles':
        # Circle pattern
        image = np.zeros((height, width), dtype=np.uint8)
        centers = [(120, 160), (120, 480), (360, 160), (360, 480)]
        for center in centers:
            cv2.circle(image, center, 60, 255, -1)
            cv2.circle(image, center, 30, 0, -1)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return image


def demo_single_extractor(extractor_type, image, max_keypoints=1000, verbose=True, 
                          visualize=False, output_dir=None, vis_options=None):
    """
    Demonstrate a single feature extractor
    
    Args:
        extractor_type: Type of extractor to test
        image: Input image
        max_keypoints: Maximum number of keypoints to extract
        verbose: Whether to print detailed information
        visualize: Whether to save visualization
        output_dir: Output directory for visualizations
        vis_options: Visualization options dict
    
    Returns:
        dict: Results containing features, timing, and config
    """
    if verbose:
        print(f"\n--- Testing {extractor_type.upper()} ---")
    
    try:
        # Create extractor with keypoint limit
        extractor_config = {}
        
        # Set keypoint limit based on extractor type
        if extractor_type in ['sift', 'orb', 'surf']:
            extractor_config['n_features'] = max_keypoints
        elif extractor_type in ['superpoint', 'disk', 'alike-t', 'alike-s', 'alike-n', 'alike-l']:
            extractor_config['max_keypoints'] = max_keypoints
        elif extractor_type == 'dummy_learned':
            extractor_config['n_features'] = max_keypoints
        
        extractor = FeatureExtractorFactory.create(extractor_type, **extractor_config)
        if verbose:
            print(f"Created: {extractor}")
            print(f"Config: {extractor.get_config()}")
            print(f"Max keypoints: {max_keypoints}")
        
        # Extract features with timing
        import time
        start_time = time.time()
        features = extractor.extract(image)
        end_time = time.time()
        
        extraction_time = end_time - start_time
        
        # Print statistics
        if verbose:
            print_feature_statistics(features, f"{extractor_type.upper()} Features")
            print(f"Extraction time: {extraction_time:.4f}s")
        
        # Visualization
        if visualize and output_dir and len(features.keypoints) > 0:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Basic visualization
            basic_vis = draw_keypoints(image, features, (0, 0, 255), True)
            basic_path = output_path / f"{extractor_type}_basic.jpg"
            cv2.imwrite(str(basic_path), basic_vis)
            
            # Advanced visualization
            if vis_options:
                advanced_vis = draw_keypoints_advanced(
                    image, features,
                    show_response=vis_options.get('show_response', False),
                    color_by_response=vis_options.get('color_by_response', False),
                    max_keypoints_display=vis_options.get('max_display', 100)
                )
                advanced_path = output_path / f"{extractor_type}_advanced.jpg"
                cv2.imwrite(str(advanced_path), advanced_vis)
                
                if verbose:
                    print(f"Saved visualizations: {basic_path}, {advanced_path}")
            else:
                if verbose:
                    print(f"Saved visualization: {basic_path}")
        
        return {
            'features': features,
            'timing': extraction_time,
            'config': extractor.get_config(),
            'success': True,
            'error': None
        }
        
    except Exception as e:
        if verbose:
            print(f"Error testing {extractor_type}: {e}")
        return {
            'features': None,
            'timing': None,
            'config': None,
            'success': False,
            'error': str(e)
        }


def demo_traditional_extractors(image, max_keypoints=1000):
    """Demo traditional feature extractors"""
    print("=== Traditional Feature Extractors Demo ===")
    
    traditional_extractors = ['sift', 'orb', 'surf', 'brief']
    results = {}
    
    for extractor_type in traditional_extractors:
        result = demo_single_extractor(extractor_type, image, max_keypoints)
        results[extractor_type] = result
    
    return results


def demo_learned_extractors(image, max_keypoints=1000):
    """Demo learning-based feature extractors"""
    print("\n=== Learning-based Feature Extractors Demo ===")
    
    # First test dummy learned extractor (always available)
    print("\n--- Testing Dummy Learned Extractor ---")
    try:
        extractor = FeatureExtractorFactory.create('dummy_learned', n_features=max_keypoints)
        features = extractor.extract(image)
        print_feature_statistics(features, "Dummy Learned Features")
        print(f"Max keypoints: {max_keypoints}")
    except Exception as e:
        print(f"Error testing dummy learned extractor: {e}")
    
    # Test other learned extractors (may require model downloads)
    learned_extractors = ['superpoint', 'disk', 'alike-t', 'alike-s', 'alike-n', 'alike-l']
    results = {}
    
    for extractor_type in learned_extractors:
        print(f"\n--- Testing {extractor_type.upper()} (may download model) ---")
        try:
            extractor = FeatureExtractorFactory.create(extractor_type, max_keypoints=max_keypoints)
            print(f"Created: {extractor}")
            print(f"Config: {extractor.get_config()}")
            print(f"Max keypoints: {max_keypoints}")
            
            # For demo purposes, just show that the extractor was created
            # Actual feature extraction may be slow due to model loading
            print(f"{extractor_type.upper()} extractor ready")
            results[extractor_type] = {'success': True, 'created': True}
            
        except Exception as e:
            print(f"{extractor_type.upper()} test skipped: {e}")
            results[extractor_type] = {'success': False, 'error': str(e)}
    
    return results


def demo_extractor_comparison(image, max_keypoints=1000, visualize=False, output_dir=None):
    """Compare different extractors on the same image"""
    print("\n=== Extractor Comparison ===")
    
    # Test a subset of extractors for comparison
    extractors_to_compare = ['sift', 'orb', 'dummy_learned']
    
    print(f"Comparing extractors on {image.shape} image...")
    print(f"Max keypoints: {max_keypoints}")
    
    results = {}
    comparison_images = []
    comparison_features = []
    comparison_titles = []
    
    for extractor_type in extractors_to_compare:
        result = demo_single_extractor(extractor_type, image, max_keypoints, verbose=False)
        results[extractor_type] = result
        
        if result['success']:
            num_features = len(result['features'])
            timing = result['timing']
            print(f"{extractor_type.upper():12}: {num_features:4d} features, {timing:.4f}s")
            
            # Collect for comparison visualization
            if visualize:
                comparison_images.append(image)
                comparison_features.append(result['features'])
                comparison_titles.append(f"{extractor_type.upper()}")
        else:
            print(f"{extractor_type.upper():12}: Failed - {result['error']}")
    
    # Save comparison visualization
    if visualize and output_dir and comparison_images:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        comparison_path = output_path / "extractor_comparison.jpg"
        
        save_keypoints_comparison(
            comparison_images, comparison_features, comparison_path, 
            comparison_titles, cols=3
        )
        print(f"Saved extractor comparison: {comparison_path}")
    
    return results


def demo_extractor_parameters(image):
    """Demonstrate different parameter settings"""
    print("\n=== Parameter Variation Demo ===")
    
    # SIFT with different parameters
    print("\n--- SIFT Parameter Variations ---")
    sift_configs = [
        {'n_features': 500, 'name': 'SIFT-500'},
        {'n_features': 1000, 'name': 'SIFT-1000'},
        {'n_features': 2000, 'name': 'SIFT-2000'},
        {'contrast_threshold': 0.03, 'name': 'SIFT-low-contrast'},
        {'contrast_threshold': 0.08, 'name': 'SIFT-high-contrast'},
    ]
    
    for config in sift_configs:
        name = config.pop('name')
        try:
            extractor = FeatureExtractorFactory.create('sift', **config)
            features = extractor.extract(image)
            print(f"{name:20}: {len(features):4d} features")
        except Exception as e:
            print(f"{name:20}: Error - {e}")
    
    # ORB with different parameters
    print("\n--- ORB Parameter Variations ---")
    orb_configs = [
        {'n_features': 500, 'name': 'ORB-500'},
        {'n_features': 1000, 'name': 'ORB-1000'},
        {'scale_factor': 1.1, 'name': 'ORB-fine-scale'},
        {'scale_factor': 1.4, 'name': 'ORB-coarse-scale'},
    ]
    
    for config in orb_configs:
        name = config.pop('name')
        try:
            extractor = FeatureExtractorFactory.create('orb', **config)
            features = extractor.extract(image)
            print(f"{name:20}: {len(features):4d} features")
        except Exception as e:
            print(f"{name:20}: Error - {e}")


def main():
    parser = argparse.ArgumentParser(description='Feature Extractor Demo')
    parser.add_argument('--extractor', type=str, 
                       help='Specific extractor to test (default: test all)')
    parser.add_argument('--image', type=str, 
                       help='Path to image file (default: use generated test image)')
    parser.add_argument('--pattern', type=str, default='noise',
                       choices=['noise', 'checkerboard', 'circles'],
                       help='Test image pattern (default: noise)')
    parser.add_argument('--size', type=str, default='auto',
                       help='Test image size as "height width" or "auto" (default: auto)')
    parser.add_argument('--max-keypoints', type=int, default=1000,
                       help='Maximum number of keypoints to extract (default: 1000)')
    parser.add_argument('--visualize', action='store_true',
                       help='Draw keypoints on images and save visualization')
    parser.add_argument('--output-dir', type=str, default='feature_extractor_output',
                       help='Output directory for visualizations (default: feature_extractor_output)')
    parser.add_argument('--show-response', action='store_true',
                       help='Show response values on keypoints')
    parser.add_argument('--color-by-response', action='store_true',
                       help='Color keypoints by response values')
    parser.add_argument('--max-display', type=int, default=100,
                       help='Maximum keypoints to display in visualization (default: 100)')
    parser.add_argument('--list', action='store_true',
                       help='List available extractors and exit')
    
    args = parser.parse_args()
    
    # List available extractors
    if args.list:
        print("Available extractors:")
        for extractor in get_available_extractors():
            print(f"  - {extractor}")
        return
    
    print("Feature Extractor Demo")
    print("=====================")
    
    # Load or create image
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return
        
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return
        
        print(f"Loaded image: {image_path} ({image.shape}) - size auto-detected")
    else:
        # Handle size argument
        if args.size == 'auto':
            # Default size for generated test images
            image_size = (480, 640)
            print(f"Using default test image size: {image_size}")
        else:
            try:
                # Parse size from string like "480 640"
                size_parts = args.size.split()
                if len(size_parts) != 2:
                    raise ValueError("Size must be two integers")
                image_size = (int(size_parts[0]), int(size_parts[1]))
            except (ValueError, IndexError):
                print(f"Error: Invalid size format '{args.size}'. Use 'auto' or 'height width'")
                return
        
        image = create_test_image(image_size, args.pattern)
        print(f"Created test image: {image.shape} ({args.pattern} pattern)")
    
    print(f"Available extractors: {get_available_extractors()}")
    
    # Prepare visualization options
    vis_options = None
    if args.visualize:
        vis_options = {
            'show_response': args.show_response,
            'color_by_response': args.color_by_response,
            'max_display': args.max_display
        }
        print(f"Visualization enabled. Output directory: {args.output_dir}")
    
    # Test specific extractor or all extractors
    if args.extractor:
        if args.extractor not in get_available_extractors():
            print(f"Error: Unknown extractor '{args.extractor}'")
            print(f"Available: {get_available_extractors()}")
            return
        
        result = demo_single_extractor(
            args.extractor, image, args.max_keypoints, 
            visualize=args.visualize, output_dir=args.output_dir, vis_options=vis_options
        )
        if result['success']:
            print(f"\nSuccess! Extracted {len(result['features'])} features")
        else:
            print(f"\nFailed: {result['error']}")
    else:
        # Run comprehensive demo
        try:
            demo_traditional_extractors(image, args.max_keypoints)
            demo_learned_extractors(image, args.max_keypoints)
            demo_extractor_comparison(image, args.max_keypoints, args.visualize, args.output_dir)
            demo_extractor_parameters(image)
            
            print("\n=== Demo completed successfully ===")
            
        except Exception as e:
            print(f"\nDemo failed: {e}")
            print("Make sure all dependencies are installed: uv sync")


if __name__ == "__main__":
    main()