"""
Unified data generation pipeline.
Combines all countries and entity types, extracts features, generates balanced dataset.
"""

import pandas as pd
import sys
from pathlib import Path
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_generation.cl.generators import ChileDataGenerator
from data_generation.br.generators import BrazilDataGenerator
from data_generation.uy.generators import UruguayDataGenerator
from data_generation.co.generators import ColombiaDataGenerator
from feature_extraction.extractors import FeatureExtractor


def generate_complete_dataset(
    samples_per_entity_country: int = 1000,
    output_path: str = None
) -> pd.DataFrame:
    """
    Generate complete dataset with all countries and entity types.
    
    Args:
        samples_per_entity_country: Number of samples per entity type per country (50% TRUE, 50% FALSE)
        output_path: Path to save CSV file. If None, returns DataFrame only.
    
    Returns:
        DataFrame with all features and labels
    """
    # Initialize generators
    generators = {
        'CL': ChileDataGenerator(),
        'BR': BrazilDataGenerator(),
        'UY': UruguayDataGenerator(),
        'CO': ColombiaDataGenerator()
    }
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Entity type mapping to generator methods
    entity_methods = {
        'ID': {'CL': 'generate_rut', 'BR': 'generate_cpf', 'UY': 'generate_ci', 'CO': 'generate_cc'},
        'PHONE': 'generate_phones',
        'EMAIL': 'generate_emails',
        'PER': 'generate_names',
        'LOC': 'generate_addresses',
        'DATE': 'generate_dates',
        'SEX': 'generate_gender'
    }
    
    all_data = []
    total_entities = len(entity_methods) * len(generators)
    current = 0
    
    print(f"Generating {samples_per_entity_country} samples per entity type per country...")
    print(f"Total combinations: {total_entities}")
    print(f"Total samples: {samples_per_entity_country * total_entities}")
    
    # Generate data for each country and entity type
    for country_code, generator in generators.items():
        for entity_type, method_name in entity_methods.items():
            current += 1
            print(f"[{current}/{total_entities}] Generating {entity_type} for {country_code}...")
            
            # Get the appropriate method name for this entity/country
            if isinstance(method_name, dict):
                method_name = method_name[country_code]
            
            method = getattr(generator, method_name)
            
            # Generate TRUE examples (50%)
            true_count = samples_per_entity_country // 2
            true_samples = method(valid=True, count=true_count)
            
            # Generate FALSE examples (50%)
            false_count = samples_per_entity_country - true_count
            false_samples = method(valid=False, count=false_count)
            
            # Combine and extract features
            all_samples = true_samples + false_samples
            
            for text, is_pii in all_samples:
                # Extract features
                features = extractor.extract_features(text, entity_type, country_code)
                
                # Add metadata
                features['text'] = text
                features['is_pii'] = int(is_pii)
                
                all_data.append(features)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"\nSamples by entity type:")
    print(df.groupby('entity_type')['is_pii'].count())
    print(f"\nSamples by country:")
    print(df.groupby('country')['is_pii'].count())
    print(f"\nPII vs Non-PII distribution:")
    print(df['is_pii'].value_counts())
    print(f"\nPII ratio by entity type:")
    print(df.groupby('entity_type')['is_pii'].mean())
    
    # Save if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nDataset saved to: {output_file}")
        print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    return df


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_dir: str = None
) -> tuple:
    """
    Split dataset into train/val/test sets with stratification.
    
    Args:
        df: Complete dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        output_dir: Directory to save split datasets
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    # Stratify by entity_type and is_pii
    df['stratify_col'] = df['entity_type'] + '_' + df['is_pii'].astype(str)
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['stratify_col'],
        random_state=42
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df['stratify_col'],
        random_state=42
    )
    
    # Remove stratification column
    train_df = train_df.drop('stratify_col', axis=1)
    val_df = val_df.drop('stratify_col', axis=1)
    test_df = test_df.drop('stratify_col', axis=1)
    
    print(f"\n=== Dataset Split ===")
    print(f"Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Save if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(output_path / 'train.csv', index=False)
        val_df.to_csv(output_path / 'val.csv', index=False)
        test_df.to_csv(output_path / 'test.csv', index=False)
        
        print(f"\nSplit datasets saved to: {output_path}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Configuration
    SAMPLES_PER_ENTITY_COUNTRY = 1500  # 1500 per entity/country = 84,000 total samples
    OUTPUT_DIR = Path(__file__).parent.parent / "datasets"
    
    # Generate complete dataset
    print("Starting dataset generation...")
    df = generate_complete_dataset(
        samples_per_entity_country=SAMPLES_PER_ENTITY_COUNTRY,
        output_path=OUTPUT_DIR / "complete_dataset.csv"
    )
    
    # Split into train/val/test
    print("\nSplitting dataset...")
    train_df, val_df, test_df = split_dataset(
        df,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        output_dir=OUTPUT_DIR
    )
    
    print("\n✓ Dataset generation complete!")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
