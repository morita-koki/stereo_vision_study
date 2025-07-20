import os
import hashlib
import requests
from pathlib import Path
from typing import Dict, Optional, Union, Any
import json
import shutil
from urllib.parse import urlparse
import warnings


class ModelManager:
    """深層学習モデルの管理クラス"""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        ModelManagerの初期化
        
        Args:
            cache_dir: キャッシュディレクトリ（省略時は ~/.cache/stereo_vision）
        """
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'stereo_vision'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # モデル情報ファイル
        self.model_info_file = self.cache_dir / 'model_info.json'
        self.model_info = self._load_model_info()
    
    def _load_model_info(self) -> Dict[str, Any]:
        """モデル情報を読み込み"""
        if self.model_info_file.exists():
            try:
                with open(self.model_info_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_model_info(self):
        """モデル情報を保存"""
        try:
            with open(self.model_info_file, 'w') as f:
                json.dump(self.model_info, f, indent=2)
        except IOError as e:
            warnings.warn(f"Failed to save model info: {e}")
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """ファイルのSHA256ハッシュを計算"""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _download_file(self, url: str, dest_path: Path, expected_hash: Optional[str] = None) -> bool:
        """
        ファイルをダウンロード
        
        Args:
            url: ダウンロードURL
            dest_path: 保存先パス
            expected_hash: 期待されるハッシュ値
            
        Returns:
            bool: ダウンロード成功かどうか
        """
        try:
            print(f"Downloading {url} to {dest_path}")
            
            # ディレクトリ作成
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # ダウンロード
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # ファイルに保存
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # ハッシュ値チェック
            if expected_hash:
                actual_hash = self._compute_file_hash(dest_path)
                if actual_hash != expected_hash:
                    print(f"Hash mismatch: expected {expected_hash}, got {actual_hash}")
                    dest_path.unlink()  # ファイルを削除
                    return False
            
            print(f"Successfully downloaded {dest_path}")
            return True
            
        except Exception as e:
            print(f"Download failed: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False
    
    def download_model(self, 
                      model_name: str, 
                      url: str, 
                      expected_hash: Optional[str] = None,
                      force: bool = False) -> Optional[Path]:
        """
        モデルをダウンロード
        
        Args:
            model_name: モデル名
            url: ダウンロードURL
            expected_hash: 期待されるSHA256ハッシュ
            force: 強制再ダウンロード
            
        Returns:
            Optional[Path]: ダウンロードされたファイルのパス
        """
        # ファイル名を推測
        parsed_url = urlparse(url)
        filename = Path(parsed_url.path).name
        if not filename:
            filename = f"{model_name}.pth"
        
        model_path = self.cache_dir / model_name / filename
        
        # 既存ファイルチェック
        if model_path.exists() and not force:
            # ハッシュチェック
            if expected_hash:
                actual_hash = self._compute_file_hash(model_path)
                if actual_hash == expected_hash:
                    print(f"Model {model_name} already exists with correct hash")
                    return model_path
                else:
                    print(f"Model {model_name} exists but hash mismatch, re-downloading")
            else:
                print(f"Model {model_name} already exists")
                return model_path
        
        # ダウンロード
        if self._download_file(url, model_path, expected_hash):
            # モデル情報を更新
            self.model_info[model_name] = {
                'path': str(model_path),
                'url': url,
                'hash': expected_hash,
                'filename': filename
            }
            self._save_model_info()
            return model_path
        
        return None
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """
        モデルのパスを取得
        
        Args:
            model_name: モデル名
            
        Returns:
            Optional[Path]: モデルファイルのパス
        """
        if model_name in self.model_info:
            model_path = Path(self.model_info[model_name]['path'])
            if model_path.exists():
                return model_path
        
        return None
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        ダウンロード済みモデルの一覧を取得
        
        Returns:
            Dict[str, Dict[str, Any]]: モデル情報
        """
        available_models = {}
        for model_name, info in self.model_info.items():
            model_path = Path(info['path'])
            if model_path.exists():
                available_models[model_name] = info
        return available_models
    
    def remove_model(self, model_name: str) -> bool:
        """
        モデルを削除
        
        Args:
            model_name: モデル名
            
        Returns:
            bool: 削除成功かどうか
        """
        if model_name in self.model_info:
            model_path = Path(self.model_info[model_name]['path'])
            
            try:
                # ファイル削除
                if model_path.exists():
                    model_path.unlink()
                
                # ディレクトリも削除（空の場合）
                if model_path.parent.exists() and not any(model_path.parent.iterdir()):
                    model_path.parent.rmdir()
                
                # モデル情報から削除
                del self.model_info[model_name]
                self._save_model_info()
                
                print(f"Model {model_name} removed successfully")
                return True
                
            except Exception as e:
                print(f"Failed to remove model {model_name}: {e}")
                return False
        
        return False
    
    def clear_cache(self) -> bool:
        """
        キャッシュを全削除
        
        Returns:
            bool: 削除成功かどうか
        """
        try:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.model_info = {}
            print("Cache cleared successfully")
            return True
        except Exception as e:
            print(f"Failed to clear cache: {e}")
            return False
    
    def get_cache_size(self) -> int:
        """
        キャッシュサイズを取得（バイト）
        
        Returns:
            int: キャッシュサイズ
        """
        total_size = 0
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                file_path = Path(root) / file
                total_size += file_path.stat().st_size
        return total_size
    
    def get_cache_size_mb(self) -> float:
        """
        キャッシュサイズを取得（MB）
        
        Returns:
            float: キャッシュサイズ（MB）
        """
        return self.get_cache_size() / (1024 * 1024)


# グローバルインスタンス
_global_manager = ModelManager()


def download_model(model_name: str, 
                  url: str, 
                  expected_hash: Optional[str] = None,
                  force: bool = False) -> Optional[Path]:
    """
    モデルをダウンロード（グローバル関数）
    
    Args:
        model_name: モデル名
        url: ダウンロードURL
        expected_hash: 期待されるSHA256ハッシュ
        force: 強制再ダウンロード
        
    Returns:
        Optional[Path]: ダウンロードされたファイルのパス
    """
    return _global_manager.download_model(model_name, url, expected_hash, force)


def get_model_path(model_name: str) -> Optional[Path]:
    """
    モデルのパスを取得（グローバル関数）
    
    Args:
        model_name: モデル名
        
    Returns:
        Optional[Path]: モデルファイルのパス
    """
    return _global_manager.get_model_path(model_name)


def list_models() -> Dict[str, Dict[str, Any]]:
    """
    ダウンロード済みモデルの一覧を取得（グローバル関数）
    
    Returns:
        Dict[str, Dict[str, Any]]: モデル情報
    """
    return _global_manager.list_models()


def remove_model(model_name: str) -> bool:
    """
    モデルを削除（グローバル関数）
    
    Args:
        model_name: モデル名
        
    Returns:
        bool: 削除成功かどうか
    """
    return _global_manager.remove_model(model_name)


def clear_cache() -> bool:
    """
    キャッシュを全削除（グローバル関数）
    
    Returns:
        bool: 削除成功かどうか
    """
    return _global_manager.clear_cache()


def get_cache_info() -> Dict[str, Any]:
    """
    キャッシュ情報を取得（グローバル関数）
    
    Returns:
        Dict[str, Any]: キャッシュ情報
    """
    return {
        'cache_dir': str(_global_manager.cache_dir),
        'size_mb': _global_manager.get_cache_size_mb(),
        'models': list(_global_manager.list_models().keys())
    }


# 既知のモデルURL（今後のために準備）
KNOWN_MODELS = {
    'superpoint': {
        'url': 'https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth',
        'hash': None,  # 実際のハッシュ値は後で設定
        'description': 'SuperPoint feature extractor'
    },
    'superglue_indoor': {
        'url': 'https://github.com/rpautrat/SuperGlue/raw/master/superglue_indoor.pth',
        'hash': None,
        'description': 'SuperGlue indoor matcher'
    },
    'superglue_outdoor': {
        'url': 'https://github.com/rpautrat/SuperGlue/raw/master/superglue_outdoor.pth',
        'hash': None,
        'description': 'SuperGlue outdoor matcher'
    }
}


def download_known_model(model_name: str, force: bool = False) -> Optional[Path]:
    """
    既知のモデルをダウンロード
    
    Args:
        model_name: モデル名
        force: 強制再ダウンロード
        
    Returns:
        Optional[Path]: ダウンロードされたファイルのパス
    """
    if model_name not in KNOWN_MODELS:
        available_models = list(KNOWN_MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
    
    model_info = KNOWN_MODELS[model_name]
    return download_model(model_name, model_info['url'], model_info['hash'], force)