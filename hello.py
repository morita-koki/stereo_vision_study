import numpy as np
from stereo_vision.feature_extractor import (
    FeatureExtractorFactory,
    get_available_extractors,
    print_feature_statistics
)


def demo_feature_extraction():
    """特徴量抽出のデモンストレーション"""
    print("=== Feature Extractor Demo ===")
    
    # 利用可能な抽出器を表示
    print(f"Available extractors: {get_available_extractors()}")
    
    # テスト用の画像を生成（実際の画像がない場合）
    test_image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
    
    # 各抽出器をテスト
    extractors_to_test = ['sift', 'orb']
    
    for extractor_type in extractors_to_test:
        print(f"\n--- Testing {extractor_type.upper()} ---")
        
        try:
            # 抽出器を作成
            extractor = FeatureExtractorFactory.create(extractor_type)
            print(f"Created {extractor}")
            
            # 特徴量を抽出
            features = extractor.extract(test_image)
            
            # 統計情報を表示
            print_feature_statistics(features, f"{extractor_type.upper()} Features")
            
        except Exception as e:
            print(f"Error testing {extractor_type}: {e}")
    
    print("\n=== Demo completed ===")


def main():
    print("Hello from stereo-vision!")
    print("Starting feature extraction demo...")
    
    try:
        demo_feature_extraction()
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure all dependencies are installed: uv sync")


if __name__ == "__main__":
    main()
