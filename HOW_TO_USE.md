# How to Use the PII Classifier

This guide provides step-by-step instructions on how to use the PII Classifier for LATAM countries, from initial setup through model training to inference.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Step 1: Generate Training Dataset](#step-1-generate-training-dataset)
3. [Step 2: Train the Classifier](#step-2-train-the-classifier)
4. [Step 3: Use the Model for Inference](#step-3-use-the-model-for-inference)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Navigate to Repository

```bash
cd Classifier_PII_LATAM
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **xgboost** (ML classifier)
- **scikit-learn** (preprocessing)
- **pandas** (data handling)
- **numpy** (numerical operations)
- **faker** (synthetic data generation)

### Verify Installation

```bash
python3 -c "import xgboost; import faker; print('✓ All dependencies installed!')"
```

---

## Step 1: Generate Training Dataset

### What This Does
Creates a balanced dataset of 42,000 samples (50% valid PII, 50% false positives) using the Faker library for realistic, diverse data.

### Configuration

Before generating data, you can customize settings in `configs/entity_thresholds.json`:

```json
{
  "train_val_test_split": [0.7, 0.15, 0.15],
  "samples_per_entity_per_country": 1500,
  "random_seed": 42,
  "entity_types": ["ID", "PHONE", "EMAIL", "PER", "LOC", "DATE", "SEX"],
  "countries": ["CL", "BR", "UY", "CO"]
}
```

**Parameters**:
- `samples_per_entity_per_country`: How many samples to generate per entity/country combination (default: 1500)
  - Higher = more diverse training data but slower generation
  - Lower = faster but less diverse
- `train_val_test_split`: How to split data (70% train, 15% val, 15% test)
- `random_seed`: For reproducibility (change to 43, 44, etc. for different random data)

### Running Data Generation

```bash
python3 data_generation/generate_dataset.py
```

### Expected Output

```
Starting dataset generation...
Generating 1500 samples per entity type per country...
Total combinations: 28
Total samples: 42000

[1/28] Generating ID for CL...
[2/28] Generating PHONE for CL...
...
[28/28] Generating SEX for CO...

=== Dataset Statistics ===
Total samples: 42000

Samples by entity type:
entity_type
DATE     6000
EMAIL    6000
ID       6000
LOC      6000
PER      6000
PHONE    6000
SEX      6000

Samples by country:
country
BR    10500
CL    10500
CO    10500
UY    10500

PII vs Non-PII distribution:
is_pii
0    21000
1    21000

=== Dataset Split ===
Train: 29400 samples (70.0%)
Val: 6300 samples (15.0%)
Test: 6300 samples (15.0%)

✓ Dataset generation complete!
```

### Output Files

Generated in `/datasets/` folder:
- **complete_dataset.csv** (42,000 rows) - Full balanced dataset
- **train.csv** (29,400 rows) - Training data
- **val.csv** (6,300 rows) - Validation data
- **test.csv** (6,300 rows) - Test data

### What's in the Dataset?

Each row contains:
```
text, entity_type, country, is_pii, features...
"15.783.037-6", "ID", "CL", 1, ...      # Valid Chilean RUT
"15.783.037-5", "ID", "CL", 0, ...      # Invalid RUT (wrong checksum)
"user@example.com", "EMAIL", "BR", 1, ... # Valid email
"test@test", "EMAIL", "BR", 0, ...      # Invalid email (no TLD)
```

---

## Step 2: Train the Classifier

### What This Does
Trains an XGBoost model using the generated dataset with:
- Entity-specific threshold optimization
- Feature importance analysis
- Validation on held-out test set

### Configuration

Edit `configs/entity_thresholds.json` for training parameters:

```json
{
  "id": {
    "threshold": 0.65,
    "hyperparameters": {
      "max_depth": 6,
      "learning_rate": 0.1,
      "n_estimators": 300
    }
  },
  "phone": {
    "threshold": 0.65,
    "hyperparameters": {...}
  },
  "email": {
    "threshold": 0.50,
    "hyperparameters": {...}
  },
  "per": {
    "threshold": 0.35,
    "hyperparameters": {...}
  },
  "loc": {
    "threshold": 0.40,
    "hyperparameters": {...}
  },
  "date": {
    "threshold": 0.50,
    "hyperparameters": {...}
  },
  "sex": {
    "threshold": 0.50,
    "hyperparameters": {...}
  }
}
```

**Key Parameters**:
- **threshold**: Probability cutoff for classifying as PII (0-1)
  - Higher = stricter (fewer false positives, more false negatives)
  - Lower = lenient (more false positives, fewer false negatives)
- **max_depth**: Tree depth (6-10 typical, higher = more complex)
- **learning_rate**: Step size (0.05-0.3, lower = slower but potentially better)
- **n_estimators**: Number of trees (100-500, more = better but slower)

### Running Model Training

```bash
python3 train_model.py
```

### Expected Output

```
Loading datasets...
Preparing training data...
Training samples: 29400
Validation samples: 6300
Number of features: 79

Training with default parameters...
[0]     validation_0-logloss:0.64580
[1]     validation_0-logloss:0.60300
...
[299]   validation_0-logloss:0.01683

=== Validation Set Performance ===
              precision    recall  f1-score   support
     Not PII       1.00      0.99      0.99      3150
         PII       0.99      1.00      0.99      3150

=== Performance by Entity Type ===
EMAIL: F1=1.000, Precision=1.000, Recall=1.000
PHONE: F1=0.973, Precision=0.955, Recall=0.991
...

=== Optimizing Thresholds per Entity Type ===
EMAIL: 0.997 (F1=1.000)
PHONE: 0.533 (F1=0.974)
...

✓ Model saved to: /models
  - xgboost_pii_classifier.pkl
  - label_encoders.pkl
  - optimized_thresholds.json
  - feature_names.json

✓ Training complete!
```

### Output Models

Saved in `/models/` folder:
- **xgboost_pii_classifier.pkl** - The trained model
- **label_encoders.pkl** - Feature encoders for categorical variables
- **optimized_thresholds.json** - Best thresholds per entity type
- **feature_names.json** - Names of all 79 features

### Understanding the Results

- **F1 Score**: 0.99 means the model is very accurate
- **Precision**: How many predicted PIIs are actually correct
- **Recall**: How many actual PIIs are found
- **AUC-ROC**: 0.9999 is excellent (1.0 = perfect)

---

## Step 3: Use the Model for Inference

### Basic Usage: Single Entity Validation

#### Example 1: Validate a Chilean RUT (ID)

```python
from inference.pipeline import PII_ValidationPipeline

# Initialize the pipeline (loads trained model)
pipeline = PII_ValidationPipeline()

# Validate a Chilean RUT
result = pipeline.validate(
    text="15.783.037-6",
    entity_type="ID",
    country="CL"
)

# Print results
print(f"Text: {result['text']}")
print(f"Is PII: {result['is_pii']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Reason: {result['reason']}")

# Output:
# Text: 15.783.037-6
# Is PII: True
# Confidence: 99.50%
# Reason: Valid checksum + high ML confidence (0.9950)
```

#### Example 2: Validate an Invalid RUT

```python
# Invalid RUT (wrong checksum)
result = pipeline.validate(
    text="15.783.037-5",  # Wrong check digit
    entity_type="ID",
    country="CL"
)

print(f"Is PII: {result['is_pii']}")
print(f"Reason: {result['reason']}")

# Output:
# Is PII: False
# Reason: Invalid checksum (Stage 1 deterministic rule)
```

#### Example 3: Validate an Email

```python
result = pipeline.validate(
    text="juan.garcia@example.com",
    entity_type="EMAIL",
    country="CL"
)

print(f"Is PII: {result['is_pii']}")
print(f"Confidence: {result['confidence']:.2%}")

# Output:
# Is PII: True
# Confidence: 99.70%
```

#### Example 4: Validate a Phone Number

```python
result = pipeline.validate(
    text="+56 9 8765 4321",
    entity_type="PHONE",
    country="CL"
)

print(f"Is PII: {result['is_pii']}")
print(f"Confidence: {result['confidence']:.2%}")

# Output:
# Is PII: True
# Confidence: 87.30%
```

#### Example 5: Validate a Person Name

```python
result = pipeline.validate(
    text="Juan García",
    entity_type="PER",
    country="CL"
)

print(f"Is PII: {result['is_pii']}")
print(f"Confidence: {result['confidence']:.2%}")

# Output:
# Is PII: True
# Confidence: 76.50%
```

### Batch Processing: Multiple Entities

```python
from inference.pipeline import PII_ValidationPipeline

pipeline = PII_ValidationPipeline()

# Define multiple entities
entities = [
    {"text": "15.783.037-6", "entity_type": "ID", "country": "CL"},
    {"text": "juan@example.com", "entity_type": "EMAIL", "country": "CL"},
    {"text": "+56 9 8765 4321", "entity_type": "PHONE", "country": "CL"},
    {"text": "Juan García", "entity_type": "PER", "country": "CL"},
    {"text": "01/12/1990", "entity_type": "DATE", "country": "CL"},
    {"text": "Calle 5 #123", "entity_type": "LOC", "country": "CL"},
    {"text": "masculino", "entity_type": "SEX", "country": "CL"}
]

# Validate all entities
results = pipeline.validate_batch(entities)

# Print results
for result in results:
    status = "✓ PII" if result['is_pii'] else "✗ NOT PII"
    print(f"{status} | {result['text']:<20} | {result['entity_type']:<8} | {result['confidence']:.0%}")

# Output:
# ✓ PII | 15.783.037-6         | ID       | 100%
# ✓ PII | juan@example.com     | EMAIL    | 100%
# ✓ PII | +56 9 8765 4321      | PHONE    | 87%
# ✓ PII | Juan García          | PER      | 77%
# ✓ PII | 01/12/1990           | DATE     | 89%
# ✓ PII | Calle 5 #123         | LOC      | 95%
# ✓ PII | masculino            | SEX      | 100%
```

### Integration with NER Models

#### Example with spaCy

```python
import spacy
from inference.pipeline import PII_ValidationPipeline

# Load spaCy NER model
nlp = spacy.load("es_core_news_sm")

# Initialize PII validator
pipeline = PII_ValidationPipeline()

# Process text with NER
text = "Mi nombre es Juan García, mi RUT es 15.783.037-6 y vivo en Santiago"
doc = nlp(text)

print("Named Entities found by spaCy:")
for ent in doc.ents:
    print(f"  {ent.text:<20} -> {ent.label_}")

print("\nValidating with PII Classifier:")

# Map spaCy labels to our entity types
label_mapping = {
    "PERSON": "PER",
    "GPE": "LOC",
    "ORG": "LOC"
}

for ent in doc.ents:
    entity_type = label_mapping.get(ent.label_, ent.label_)
    
    # Validate with our classifier
    result = pipeline.validate(ent.text, entity_type, "CL")
    
    if result['is_pii']:
        print(f"  ⚠️  PII DETECTED: {ent.text} ({entity_type})")
    else:
        print(f"  ✓ Not PII: {ent.text} ({entity_type})")

# Output:
# Named Entities found by spaCy:
#   Juan García -> PERSON
#   15.783.037-6 -> MISC
#   Santiago -> GPE
# 
# Validating with PII Classifier:
#   ⚠️  PII DETECTED: Juan García (PER)
#   ⚠️  PII DETECTED: 15.783.037-6 (MISC)
#   ⚠️  PII DETECTED: Santiago (LOC)
```

#### Example with Hugging Face Transformers

```python
from transformers import pipeline as hf_pipeline
from inference.pipeline import PII_ValidationPipeline

# Load Hugging Face NER model
ner = hf_pipeline("ner", model="xlm-roberta-large-finetuned-conllpp")

# Initialize PII validator
pii_validator = PII_ValidationPipeline()

text = "Mi email es juan@example.com y mi teléfono es +56 9 8765 4321"

# Get entities from Hugging Face
entities = ner(text)

print("Entities found by Hugging Face NER:")
for ent in entities:
    print(f"  {ent['word']:<25} -> {ent['entity']}")

print("\nValidating with PII Classifier:")

# Map HF labels to our entity types
label_mapping = {
    "B-PER": "PER",
    "I-PER": "PER",
    "B-LOC": "LOC",
    "I-LOC": "LOC"
}

for ent in entities:
    entity_type = label_mapping.get(ent['entity'], "PER")
    word = ent['word'].replace("##", "")
    
    result = pii_validator.validate(word, entity_type, "CL")
    
    if result['is_pii']:
        print(f"  🚨 {word:<20} -> PII ({entity_type}) | Confidence: {result['confidence']:.0%}")
    else:
        print(f"  ✓  {word:<20} -> Not PII")
```

### Processing Large Files

```python
import pandas as pd
from inference.pipeline import PII_ValidationPipeline

# Initialize pipeline
pipeline = PII_ValidationPipeline()

# Load data from CSV
df = pd.read_csv("data_to_validate.csv")
# Columns: text, entity_type, country

print(f"Processing {len(df)} rows...")

# Validate all rows
results = []
for idx, row in df.iterrows():
    result = pipeline.validate(
        text=row['text'],
        entity_type=row['entity_type'],
        country=row['country']
    )
    results.append(result)
    
    if (idx + 1) % 1000 == 0:
        print(f"  Processed {idx + 1}/{len(df)} rows...")

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv("validation_results.csv", index=False)

# Print summary
pii_count = results_df['is_pii'].sum()
total = len(results_df)
print(f"\n✓ Validation complete!")
print(f"  Total: {total}")
print(f"  PII found: {pii_count} ({pii_count/total:.1%})")
print(f"  Not PII: {total - pii_count} ({(total-pii_count)/total:.1%})")
```

---

## Advanced Usage

### Adjusting Confidence Thresholds

```python
from inference.pipeline import PII_ValidationPipeline
import json

pipeline = PII_ValidationPipeline()

# Load current thresholds
with open("models/optimized_thresholds.json", "r") as f:
    thresholds = json.load(f)

print("Current thresholds:")
for entity_type, threshold in thresholds.items():
    print(f"  {entity_type}: {threshold}")

# Make stricter (fewer false positives)
thresholds["PER"] = 0.5  # Increase from 0.359 to 0.5
thresholds["LOC"] = 0.7  # Increase from 0.988 to 0.7

# Make more lenient (fewer false negatives)
thresholds["ID"] = 0.7   # Decrease from 0.871 to 0.7
thresholds["PHONE"] = 0.4  # Decrease from 0.533 to 0.4

# Save custom thresholds
with open("models/custom_thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)

print("\n✓ Custom thresholds saved!")
```

### Understanding Predictions

```python
from inference.pipeline import PII_ValidationPipeline

pipeline = PII_ValidationPipeline()

# Get detailed prediction info
result = pipeline.validate("15.783.037-6", "ID", "CL")

print("Full result object:")
print(f"  text: {result['text']}")
print(f"  entity_type: {result['entity_type']}")
print(f"  country: {result['country']}")
print(f"  is_pii: {result['is_pii']}")
print(f"  confidence: {result['confidence']:.4f}")
print(f"  reason: {result['reason']}")

# Understanding confidence:
# - High (0.95+): Very confident in prediction
# - Medium (0.7-0.95): Reasonably confident
# - Low (0.5-0.7): Less certain, borderline case
# - Very Low (<0.5): Below threshold, classified as False
```

---

## Troubleshooting

### Issue: Model not found

**Error**: `FileNotFoundError: models/xgboost_pii_classifier.pkl not found`

**Solution**: Run training first
```bash
python3 train_model.py
```

### Issue: Dataset not found

**Error**: `FileNotFoundError: datasets/train.csv not found`

**Solution**: Generate dataset first
```bash
python3 data_generation/generate_dataset.py
```

### Issue: Out of memory during data generation

**Error**: `MemoryError`

**Solution**: Reduce samples per entity
```json
{
  "samples_per_entity_per_country": 500
}
```

### Issue: Low accuracy/poor results

**Solutions**:
1. Generate more diverse data (increase `samples_per_entity_per_country`)
2. Retrain with more iterations
3. Adjust entity-specific thresholds based on your use case
4. Add more features or update feature extraction logic

### Issue: Slow inference

**Solutions**:
1. Use batch processing instead of single validation
2. Check available system resources
3. Reduce feature computation (advanced)

### Issue: spaCy model not found

**Error**: `OSError: [E050] Can't find model 'es_core_news_sm'`

**Solution**: Download the model
```bash
python3 -m spacy download es_core_news_sm
```

---

## Quick Command Reference

```bash
# Full pipeline from scratch
python3 data_generation/generate_dataset.py
python3 train_model.py
python3 example_usage.py

# Just retrain (if data already exists)
python3 train_model.py

# Just inference (if model already trained)
python3 example_usage.py
```

---

## Next Steps

1. **Try the examples** - Run the code snippets above
2. **Process your data** - Use the batch processing for large datasets
3. **Integrate with NER** - Connect to your existing NER pipeline
4. **Tune thresholds** - Adjust for your specific use case
5. **Retrain periodically** - Generate new data and retrain as needed

For more information, see [README.md](README.md).
