"""
Two-stage inference pipeline for PII validation.
Stage 1: Deterministic rules (hard filters)
Stage 2: XGBoost classification with entity-specific thresholds
"""

import pickle
import json
import time
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from validation_rules.deterministic import validate_entity, ValidationResult
from feature_extraction.extractors import FeatureExtractor
from utils import PIILogger, InputValidator, ValidationError, PerformanceMonitor


class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass


class PII_ValidationPipeline:
    """Two-stage PII validation pipeline."""
    
    def __init__(self, model_dir: str = None):
        """Initialize the validation pipeline."""
        # Initialize utilities
        self.logger = PIILogger("pii_pipeline")
        self.validator = InputValidator()
        self.monitor = PerformanceMonitor()
        
        try:
            if model_dir is None:
                model_dir = Path(__file__).parent.parent / "models"
            else:
                model_dir = Path(model_dir)
            
            if not model_dir.exists():
                raise PipelineError(f"Model directory not found: {model_dir}")
            
            # Load model
            model_path = model_dir / "xgboost_pii_classifier.pkl"
            if not model_path.exists():
                raise PipelineError(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load label encoders
            encoders_path = model_dir / "label_encoders.pkl"
            if not encoders_path.exists():
                raise PipelineError(f"Label encoders not found: {encoders_path}")
            
            with open(encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            
            # Load thresholds
            thresholds_path = model_dir / "optimized_thresholds.json"
            if not thresholds_path.exists():
                raise PipelineError(f"Thresholds file not found: {thresholds_path}")
            
            with open(thresholds_path, 'r') as f:
                self.thresholds = json.load(f)
            
            # Load feature names
            features_path = model_dir / "feature_names.json"
            if not features_path.exists():
                raise PipelineError(f"Feature names not found: {features_path}")
            
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
            
            self.logger.info("Pipeline initialized successfully", model_dir=str(model_dir))
            
        except Exception as e:
            self.logger.error("Failed to initialize pipeline", error=e)
            raise PipelineError(f"Pipeline initialization failed: {str(e)}") from e
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
    
    def validate(
        self,
        text: str,
        entity_type: str,
        country: str = "CL"
    ) -> Dict:
        """
        Validate if detected entity is truly PII.
        
        Args:
            text: Entity text to validate
            entity_type: One of PER, LOC, ID, PHONE, EMAIL, SEX, DATE
            country: Country code (CL, BR, UY, CO)
        
        Returns:
            Dictionary with validation results:
            {
                'is_pii': bool,
                'confidence': float,
                'validation_path': str,
                'reason': str,
                'details': dict
            }
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            text = self.validator.validate_text(text)
            entity_type = self.validator.validate_entity_type(entity_type)
            country = self.validator.validate_country(country)
            
        except ValidationError as e:
            # Handle edge case: empty/None text returns NOT PII
            self.logger.warning("Input validation failed", error=str(e), text=text[:50] if text else None)
            self.monitor.record_operation('validate', time.time() - start_time, success=False)
            return {
                'is_pii': False,
                'confidence': 1.0,
                'validation_path': 'validation_error',
                'reason': str(e),
                'details': {'error': str(e)}
            }
        
        try:
            # Stage 1: Deterministic validation
            deterministic_result, deterministic_reason = validate_entity(text, entity_type, country)
            
            # If deterministically invalid, reject immediately
            if deterministic_result == ValidationResult.INVALID:
                duration = time.time() - start_time
                self.monitor.record_operation('validate', duration, success=True)
                self.logger.debug("Deterministic reject", text=text[:50], entity_type=entity_type)
                
                return {
                    'is_pii': False,
                    'confidence': 1.0,
                    'validation_path': 'deterministic_reject',
                    'reason': deterministic_reason,
                    'details': {
                        'text': text,
                        'entity_type': entity_type,
                        'country': country,
                        'deterministic_result': 'INVALID'
                    }
                }
            
            # If deterministically valid, still run ML for confirmation
            # (catches edge cases where format is correct but context is wrong)
            
            # Stage 2: ML classification
            ml_result = self._ml_classify(text, entity_type, country)
            
            # If deterministically valid and ML agrees, accept with high confidence
            if deterministic_result == ValidationResult.VALID and ml_result['is_pii']:
                duration = time.time() - start_time
                self.monitor.record_operation('validate', duration, success=True)
                
                return {
                    'is_pii': True,
                    'confidence': min(ml_result['probability'] * 1.1, 1.0),  # Boost confidence
                    'validation_path': 'deterministic_pass_ml_confirm',
                    'reason': f"{deterministic_reason} + ML confirmation",
                    'details': {
                        'text': text,
                        'entity_type': entity_type,
                        'country': country,
                        'deterministic_result': 'VALID',
                        'ml_probability': ml_result['probability'],
                        'threshold': ml_result['threshold']
                    }
                }
        
            # If deterministic says valid but ML says no, trust ML (might be contextually wrong)
            if deterministic_result == ValidationResult.VALID and not ml_result['is_pii']:
                duration = time.time() - start_time
                self.monitor.record_operation('validate', duration, success=True)
                
                return {
                    'is_pii': False,
                    'confidence': 1.0 - ml_result['probability'],
                    'validation_path': 'deterministic_pass_ml_reject',
                    'reason': f"Format valid but ML detected false positive",
                    'details': {
                        'text': text,
                        'entity_type': entity_type,
                        'country': country,
                        'deterministic_result': 'VALID',
                        'ml_probability': ml_result['probability'],
                        'threshold': ml_result['threshold']
                    }
                }
        
        # Ambiguous case - rely on ML
            return {
                'is_pii': ml_result['is_pii'],
                'confidence': ml_result['probability'] if ml_result['is_pii'] else 1.0 - ml_result['probability'],
                'validation_path': 'ml_classification',
                'reason': f"ML classification (ambiguous format)",
                'details': {
                    'text': text,
                    'entity_type': entity_type,
                    'country': country,
                    'deterministic_result': 'AMBIGUOUS',
                    'ml_probability': ml_result['probability'],
                    'threshold': ml_result['threshold']
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.record_operation('validate', duration, success=False)
            self.logger.error("Validation failed", error=e, text=text[:50], entity_type=entity_type)
            raise PipelineError(f"Validation failed: {str(e)}") from e
    
    def _ml_classify(self, text: str, entity_type: str, country: str) -> Dict:
        """Run ML classification."""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(text, entity_type, country)
            
            # Encode categorical features
            for col in ['entity_type', 'country']:
                if col in self.label_encoders:
                    features[col] = self.label_encoders[col].transform([features[col]])[0]
            
            # Create feature vector in correct order
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0))
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Predict probability
            probability = self.model.predict_proba(feature_vector)[0, 1]
            
            # Get entity-specific threshold
            threshold = self.thresholds.get(entity_type, 0.5)
            
            # Make prediction
            is_pii = probability >= threshold
            
        except Exception as e:
            self.logger.error("ML classification failed", error=e, text=text[:50])
            raise PipelineError(f"ML classification failed: {str(e)}") from e
        return {
            'is_pii': bool(is_pii),
            'probability': float(probability),
            'threshold': float(threshold)
        }
    
    def validate_batch(
        self,
        entities: List[Dict]
    ) -> List[Dict]:
        """
        Validate a batch of entities with error recovery.
        
        Args:
            entities: List of dicts with keys: text, entity_type, country
        
        Returns:
            List of validation results
        """
        start_time = time.time()
        
        try:
            # Validate batch inputs
            texts = [e.get('text', '') for e in entities]
            entity_types = [e.get('entity_type', '') for e in entities]
            countries = [e.get('country', 'CL') for e in entities]
            
            self.validator.validate_batch(texts, entity_types, countries)
            
        except ValidationError as e:
            self.logger.error("Batch validation failed", error=e, batch_size=len(entities))
            raise PipelineError(f"Batch validation failed: {str(e)}") from e
        
        results = []
        errors = 0
        
        for i, entity in enumerate(entities):
            try:
                result = self.validate(
                    entity['text'],
                    entity['entity_type'],
                    entity.get('country', 'CL')
                )
                results.append(result)
            except Exception as e:
                errors += 1
                self.logger.warning(
                    "Batch item failed, continuing", 
                    index=i, 
                    error=str(e),
                    text=entity.get('text', '')[:50]
                )
                # Continue processing remaining items
                results.append({
                    'is_pii': False,
                    'confidence': 0.0,
                    'validation_path': 'error',
                    'reason': f"Processing error: {str(e)}",
                    'details': {'error': str(e), 'index': i}
                })
        
        duration = time.time() - start_time
        self.monitor.record_operation('validate_batch', duration, success=(errors == 0))
        self.logger.info(
            "Batch validation complete",
            total=len(entities),
            errors=errors,
            duration_s=f"{duration:.2f}"
        )
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        return self.monitor.get_stats()


def main():
    """Example usage of the pipeline."""
    # Initialize pipeline
    pipeline = PII_ValidationPipeline()
    
    # Test cases
    test_cases = [
        # TRUE PII examples
        {"text": "15.783.037-6", "entity_type": "ID", "country": "CL"},
        {"text": "andres.vera@gmail.com", "entity_type": "EMAIL", "country": "CL"},
        {"text": "+56 9 8765 4321", "entity_type": "PHONE", "country": "CL"},
        {"text": "Juan Pérez González", "entity_type": "PER", "country": "CL"},
        {"text": "Avenida Providencia 1234, Santiago", "entity_type": "LOC", "country": "CL"},
        {"text": "15/03/1985", "entity_type": "DATE", "country": "CL"},
        {"text": "masculino", "entity_type": "SEX", "country": "CL"},
        
        # Brazilian examples
        {"text": "123.456.789-09", "entity_type": "ID", "country": "BR"},
        {"text": "+55 (11) 9 8765-4321", "entity_type": "PHONE", "country": "BR"},
        
        # FALSE PII examples
        {"text": "12345678-9", "entity_type": "ID", "country": "CL"},  # Invalid checksum
        {"text": "test@test", "entity_type": "EMAIL", "country": "CL"},  # Invalid email
        {"text": "1111111111", "entity_type": "PHONE", "country": "CL"},  # Repeated digits
        {"text": "Santiago", "entity_type": "PER", "country": "CL"},  # City name, not person
        {"text": "Centro", "entity_type": "LOC", "country": "CL"},  # Generic location
        {"text": "32/13/2023", "entity_type": "DATE", "country": "CL"},  # Invalid date
    ]
    
    print("\n=== Testing PII Validation Pipeline ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        result = pipeline.validate(**test_case)
        
        print(f"Test {i}:")
        print(f"  Text: {test_case['text']}")
        print(f"  Type: {test_case['entity_type']}")
        print(f"  Result: {'✓ PII' if result['is_pii'] else '✗ NOT PII'}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Path: {result['validation_path']}")
        print(f"  Reason: {result['reason']}")
        print()


if __name__ == "__main__":
    main()
