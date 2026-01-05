#!/usr/bin/env python3
"""
Quick start script to set up and run the PII classifier.
"""

import sys
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def check_dependencies():
    """Check if required packages are installed."""
    print_header("Step 1: Checking Dependencies")
    
    required_packages = [
        'pandas',
        'numpy',
        'xgboost',
        'sklearn',
        'regex'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def generate_data():
    """Generate training datasets."""
    print_header("Step 2: Generating Training Data")
    
    print("This will generate ~84,000 samples (may take 2-3 minutes)...")
    print("Samples: 1,500 per entity type per country")
    print("Entity types: ID, PHONE, EMAIL, PER, LOC, DATE, SEX")
    print("Countries: Chile, Brazil, Uruguay, Colombia\n")
    
    response = input("Generate data? [y/N]: ").strip().lower()
    
    if response == 'y':
        from data_generation.generate_dataset import generate_complete_dataset, split_dataset
        
        output_dir = Path(__file__).parent / "datasets"
        
        # Generate dataset
        df = generate_complete_dataset(
            samples_per_entity_country=1500,
            output_path=output_dir / "complete_dataset.csv"
        )
        
        # Split into train/val/test
        split_dataset(
            df,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            output_dir=output_dir
        )
        
        print("\n✓ Data generation complete!")
        return True
    else:
        print("Skipping data generation.")
        return False


def train_model():
    """Train the XGBoost model."""
    print_header("Step 3: Training XGBoost Model")
    
    # Check if datasets exist
    data_dir = Path(__file__).parent / "datasets"
    if not (data_dir / "train.csv").exists():
        print("⚠️  Training data not found!")
        print("Please generate data first (Step 2)")
        return False
    
    print("This will train the XGBoost model (may take 5-10 minutes)...")
    print("- Feature extraction")
    print("- Model training")
    print("- Threshold optimization")
    print("- Feature importance analysis\n")
    
    response = input("Train model? [y/N]: ").strip().lower()
    
    if response == 'y':
        import pandas as pd
        from train_model import PII_Classifier_Trainer
        
        # Load datasets
        print("\nLoading datasets...")
        train_df = pd.read_csv(data_dir / "train.csv")
        val_df = pd.read_csv(data_dir / "val.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        
        # Initialize trainer
        trainer = PII_Classifier_Trainer()
        
        # Train model (without grid search for speed)
        model = trainer.train(train_df, val_df, use_grid_search=False)
        
        # Optimize thresholds
        thresholds = trainer.optimize_thresholds(val_df)
        
        # Show feature importance
        trainer.get_feature_importance(top_n=20)
        
        # Evaluate on test set
        print("\n=== Final Test Set Performance ===")
        X_test, y_test, _ = trainer.prepare_data(test_df)
        trainer._evaluate(X_test, y_test, test_df)
        
        # Save model
        model_dir = Path(__file__).parent / "models"
        trainer.save_model(model_dir)
        
        print("\n✓ Model training complete!")
        return True
    else:
        print("Skipping model training.")
        return False


def test_inference():
    """Test the inference pipeline."""
    print_header("Step 4: Testing Inference Pipeline")
    
    # Check if model exists
    model_dir = Path(__file__).parent / "models"
    if not (model_dir / "xgboost_pii_classifier.pkl").exists():
        print("⚠️  Trained model not found!")
        print("Please train the model first (Step 3)")
        return False
    
    print("Running inference tests with example entities...\n")
    
    response = input("Run tests? [y/N]: ").strip().lower()
    
    if response == 'y':
        from inference.pipeline import PII_ValidationPipeline
        
        pipeline = PII_ValidationPipeline()
        
        test_cases = [
            {"text": "15.783.037-6", "entity_type": "ID", "country": "CL", "label": "Chilean RUT (valid)"},
            {"text": "15.783.037-5", "entity_type": "ID", "country": "CL", "label": "Chilean RUT (invalid)"},
            {"text": "user@gmail.com", "entity_type": "EMAIL", "country": "CL", "label": "Valid email"},
            {"text": "test@test", "entity_type": "EMAIL", "country": "CL", "label": "Invalid email"},
            {"text": "+56 9 8765 4321", "entity_type": "PHONE", "country": "CL", "label": "Chilean mobile"},
            {"text": "Juan Pérez", "entity_type": "PER", "country": "CL", "label": "Person name"},
            {"text": "Santiago", "entity_type": "PER", "country": "CL", "label": "City name (false positive)"},
        ]
        
        print("\nTest Results:")
        print("-" * 70)
        
        for test in test_cases:
            label = test.pop('label')
            result = pipeline.validate(**test)
            
            status = "✓ PII" if result['is_pii'] else "✗ NOT PII"
            
            print(f"{label:<35s} | {status:<10s} | Conf: {result['confidence']:.1%}")
        
        print("\n✓ Inference tests complete!")
        return True
    else:
        print("Skipping inference tests.")
        return False


def main():
    """Main quick start workflow."""
    print("\n" + "=" * 70)
    print("  PII Classifier - Quick Start Setup")
    print("=" * 70)
    print("\nThis script will guide you through:")
    print("  1. Checking dependencies")
    print("  2. Generating training data")
    print("  3. Training the XGBoost model")
    print("  4. Testing inference pipeline")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n⚠️  Please install dependencies first: pip install -r requirements.txt")
        return
    
    # Step 2: Generate data
    data_generated = generate_data()
    
    # Step 3: Train model
    if data_generated:
        model_trained = train_model()
    else:
        # Check if data already exists
        data_dir = Path(__file__).parent / "datasets"
        if (data_dir / "train.csv").exists():
            print("\n✓ Existing training data found")
            model_trained = train_model()
        else:
            print("\n⚠️  No training data found. Please generate data first.")
            model_trained = False
    
    # Step 4: Test inference
    if model_trained:
        test_inference()
    else:
        # Check if model already exists
        model_dir = Path(__file__).parent / "models"
        if (model_dir / "xgboost_pii_classifier.pkl").exists():
            print("\n✓ Existing trained model found")
            test_inference()
        else:
            print("\n⚠️  No trained model found. Please train model first.")
    
    # Final summary
    print_header("Setup Complete!")
    print("You can now use the classifier:")
    print("\n  from inference.pipeline import PII_ValidationPipeline")
    print("  pipeline = PII_ValidationPipeline()")
    print('  result = pipeline.validate("15.783.037-6", "ID", "CL")')
    print("\nFor more examples, see: example_usage.py")
    print("Documentation: README.md\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
