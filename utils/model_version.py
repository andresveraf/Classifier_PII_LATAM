"""
Model versioning and metadata management.
Tracks model versions, metrics, and training metadata.
"""
import json
import os
import pickle
from datetime import datetime
from typing import Dict, Optional, Any


class ModelVersionManager:
    """Manages model versions and metadata."""
    
    def __init__(self, base_dir: str = "models"):
        """
        Initialize version manager.
        
        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = base_dir
        self.versions_dir = os.path.join(base_dir, "versions")
        self.metadata_file = os.path.join(base_dir, "metadata.json")
        
        # Create directories
        os.makedirs(self.versions_dir, exist_ok=True)
    
    def save_model_version(self, model, scaler, version: Optional[str] = None,
                          metrics: Optional[Dict] = None, 
                          metadata: Optional[Dict] = None) -> str:
        """
        Save a new model version with metadata.
        
        Args:
            model: Trained model object
            scaler: Fitted scaler object
            version: Version string (default: timestamp)
            metrics: Training metrics
            metadata: Additional metadata
            
        Returns:
            Version identifier
        """
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        version_dir = os.path.join(self.versions_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Save model and scaler
        model_path = os.path.join(version_dir, "model.pkl")
        scaler_path = os.path.join(version_dir, "scaler.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save version metadata
        version_metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'model_path': model_path,
            'scaler_path': scaler_path,
            'metrics': metrics or {},
            'metadata': metadata or {}
        }
        
        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(version_metadata, f, indent=2)
        
        # Update global metadata
        self._update_global_metadata(version, version_metadata)
        
        return version
    
    def load_model_version(self, version: str) -> tuple:
        """
        Load a specific model version.
        
        Args:
            version: Version identifier
            
        Returns:
            Tuple of (model, scaler, metadata)
        """
        version_dir = os.path.join(self.versions_dir, version)
        
        if not os.path.exists(version_dir):
            raise ValueError(f"Version {version} not found")
        
        # Load model and scaler
        model_path = os.path.join(version_dir, "model.pkl")
        scaler_path = os.path.join(version_dir, "scaler.pkl")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, scaler, metadata
    
    def load_latest_version(self) -> tuple:
        """
        Load the most recent model version.
        
        Returns:
            Tuple of (model, scaler, metadata)
        """
        global_metadata = self._load_global_metadata()
        
        if not global_metadata or 'versions' not in global_metadata:
            raise ValueError("No model versions found")
        
        # Get latest version
        versions = global_metadata['versions']
        latest_version = max(versions.keys())
        
        return self.load_model_version(latest_version)
    
    def list_versions(self) -> Dict[str, Dict]:
        """
        List all available model versions.
        
        Returns:
            Dictionary mapping version to metadata
        """
        global_metadata = self._load_global_metadata()
        
        if not global_metadata or 'versions' not in global_metadata:
            return {}
        
        return global_metadata['versions']
    
    def _update_global_metadata(self, version: str, version_metadata: Dict):
        """Update global metadata file."""
        global_metadata = self._load_global_metadata()
        
        if 'versions' not in global_metadata:
            global_metadata['versions'] = {}
        
        global_metadata['versions'][version] = version_metadata
        global_metadata['latest_version'] = version
        global_metadata['last_updated'] = datetime.now().isoformat()
        
        with open(self.metadata_file, 'w') as f:
            json.dump(global_metadata, f, indent=2)
    
    def _load_global_metadata(self) -> Dict:
        """Load global metadata file."""
        if not os.path.exists(self.metadata_file):
            return {}
        
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
