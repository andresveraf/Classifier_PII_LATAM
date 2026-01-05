# PII Classifier for LATAM Countries

A robust XGBoost-based PII (Personally Identifiable Information) validation classifier for Chile, Uruguay, Colombia, and Brazil. This post-processing classifier validates entities detected by NER models to reduce false positives.

## Features

- **Multi-country support**: Chile (CL), Uruguay (UY), Colombia (CO), Brazil (BR)
- **7 entity types**: ID, PHONE, EMAIL, PER (names), LOC (addresses), DATE, SEX (gender)
- **Two-stage validation**:
  - Stage 1: Deterministic rules (checksums, regex, format validation)
  - Stage 2: XGBoost ML classification with 79 engineered features
- **Entity-specific thresholds**: Optimized probability thresholds per entity type
- **Diverse training data**: Generated using Faker library with locale-specific providers
- **Exceptional accuracy**: 99% F1 score on validation/test sets

## Architecture

### Two-Stage Pipeline

**Stage 1 - Deterministic Rules**: Hard validation filters
- ID checksum validation (RUT modulo-11, CPF double-digit, CI, CC)
- Email format validation (@ symbol, TLD presence)
- Phone format validation (country codes, area codes)
- Date range validation (1900-2033)
- Sequential/repeated digit detection

**Stage 2 - ML Classification**: XGBoost with context-aware features
- 79 engineered features per entity
- Entity-specific probability thresholds
- Confusion matrix optimization
- Feature importance tracking

## Project Structure

```
Classifier_PII_LATAM/
├── configs/
│   ├── country_patterns.json          # Country-specific patterns, names, phone formats
│   └── entity_thresholds.json         # ML configuration and threshold settings
├── data_generation/
│   ├── cl/generators.py               # Chilean data generator (Faker locale: es_CL)
│   ├── br/generators.py               # Brazilian data generator (Faker locale: pt_BR)
│   ├── uy/generators.py               # Uruguayan data generator (Faker locale: es_AR)
│   ├── co/generators.py               # Colombian data generator (Faker locale: es_ES)
│   └── generate_dataset.py            # Unified dataset generation pipeline
├── datasets/
│   ├── complete_dataset.csv           # Full balanced dataset (42,000 samples)
│   ├── train.csv                      # Training set (29,400 samples, 70%)
│   ├── val.csv                        # Validation set (6,300 samples, 15%)
│   └── test.csv                       # Test set (6,300 samples, 15%)
├── feature_extraction/
│   └── extractors.py                  # Feature engineering (79 features total)
├── validation_rules/
│   └── deterministic.py               # Checksum validators (RUT, CPF, CI, CC)
├── models/
│   ├── xgboost_pii_classifier.pkl     # Trained XGBoost model
│   ├── label_encoders.pkl             # Categorical feature encoders
│   ├── optimized_thresholds.json      # Entity-specific prediction thresholds
│   └── feature_names.json             # Complete feature list
├── inference/
│   └── pipeline.py                    # Two-stage validation inference pipeline
├── train_model.py                     # XGBoost training with hyperparameter tuning
├── example_usage.py                   # Usage examples
├── quick_start.py                     # Interactive setup script
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Installation

```bash
# Clone the repository
cd Classifier_PII_LATAM

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- **xgboost>=2.0.0** - ML classifier
- **scikit-learn>=1.3.0** - Preprocessing and metrics
- **pandas>=2.0.0** - Data handling
- **numpy>=1.24.0** - Numerical operations
- **faker>=20.0.0** - Synthetic data generation (locale-specific)

## Quick Start

### 1. Generate Training Data

```bash
python data_generation/generate_dataset.py
```

**Output**: 42,000 balanced samples across all entity types and countries
- **Distribution**: 50% TRUE PII / 50% FALSE positives
- **Per entity**: 6,000 samples (3,000 valid + 3,000 invalid)
- **Per country**: 10,500 samples each
- **Data generation**: Uses Faker library with locale-specific providers
  - Brazil: `Faker('pt_BR')` - Brazilian Portuguese
  - Chile: `Faker('es_CL')` - Chilean Spanish
  - Uruguay: `Faker('es_AR')` - Argentine Spanish (es_UY not available)
  - Colombia: `Faker('es_ES')` - Castilian Spanish (es_CO not available)

### 2. Train the Model

```bash
python train_model.py
```

**Training includes**:
- XGBoost model with early stopping
- Entity-specific threshold optimization
- Feature importance analysis
- Validation and test set evaluation

**Expected results**:
- Train samples: 29,400 (70%)
- Validation samples: 6,300 (15%)
- Test samples: 6,300 (15%)

### 3. Run Inference

```bash
python example_usage.py
```

Or use the pipeline directly:

```python
from inference.pipeline import PII_ValidationPipeline

# Initialize pipeline
pipeline = PII_ValidationPipeline()

# Validate a single entity
result = pipeline.validate(
    text="15.783.037-6",
    entity_type="ID",
    country="CL"
)

print(f"Is PII: {result['is_pii']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Reason: {result['reason']}")
```

## Training Results

### Overall Performance

**Validation Set** (6,300 samples):
- Accuracy: 99%
- Precision: 0.99
- Recall: 0.99
- F1 Score: 0.99
- AUC-ROC: 0.9998

**Test Set** (6,300 samples):
- Accuracy: 99%
- Precision: 0.99
- Recall: 0.99
- F1 Score: 0.99
- AUC-ROC: 0.9999

### Performance by Entity Type

| Entity | F1 Score | Precision | Recall | Threshold |
|--------|----------|-----------|--------|-----------|
| EMAIL | 1.000 | 1.000 | 1.000 | 0.997 |
| LOC | 1.000 | 1.000 | 1.000 | 0.988 |
| SEX | 1.000 | 1.000 | 1.000 | 0.999 |
| ID | 0.998 | 0.996 | 1.000 | 0.871 |
| DATE | 0.993 | 0.987 | 1.000 | 0.690 |
| PER | 0.987 | 0.987 | 0.987 | 0.359 |
| PHONE | 0.975 | 0.965 | 0.984 | 0.533 |

### Top 10 Most Important Features

1. **word_count** (0.410) - Number of words in text
2. **validation_valid** (0.341) - Deterministic rule validation result
3. **per_word_count** (0.064) - Word count for person names
4. **id_has_separator** (0.054) - ID separator presence check
5. **validation_invalid** (0.022) - Invalid validation flag
6. **digit_count** (0.018) - Count of digits
7. **upper_count** (0.016) - Uppercase letter count
8. **phone_has_separators** (0.013) - Phone separator presence
9. **loc_has_numbers** (0.011) - Numbers in location
10. **loc_word_count** (0.006) - Word count for locations

## Entity Types

### ID (Identification Numbers)
Supports country-specific ID formats with validation:

**Chile - RUT (Rol Único Tributario)**
- Format: `XX.XXX.XXX-K` or `XXXXXXXXK`
- Validation: Modulo-11 checksum
- Example: `15.783.037-6` (valid), `15.783.037-5` (invalid)

**Brazil - CPF (Cadastro de Pessoas Físicas)**
- Format: `XXX.XXX.XXX-XX`
- Validation: Double-digit algorithm (Verhoeff check)
- Example: Uses Faker's `fake.cpf()` for realistic generation

**Uruguay - CI (Cédula de Identidad)**
- Format: Standard 8-digit with possible check digit
- Validation: Basic format and range checking
- Example: 8-digit identity numbers

**Colombia - CC (Cédula de Ciudadanía)**
- Format: Numeric identifier
- Validation: Range and format checking
- Example: Standard Colombian ID numbers

### PHONE (Phone Numbers)
Country-specific phone formats and validation:
- Area codes and country codes (+56, +55, +598, +57)
- Mobile and landline patterns
- Multiple format support (parentheses, hyphens, spaces)
- Length validation (8-15 digits)

### EMAIL (Email Addresses)
RFC 5322 compliant validation:
- Required: @ symbol and TLD (.com, .cl, .br, .uy, .co)
- Detection: Test emails, placeholder emails
- Format: local@domain.tld

### PER (Person Names)
Person name validation using word count and patterns:
- Valid: Realistic names from Faker library
- Invalid: Brand names, titles, single words, numbers

### LOC (Addresses/Locations)
Address and location validation:
- Valid: Full addresses with street, number, neighborhood
- Invalid: Single words, generic locations, incomplete patterns

### DATE (Dates)
Multiple date format support:
- Formats: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD
- Range: 1900-2033 (filters future/historical dates)
- Validation: Valid day/month combinations

### SEX (Gender)
Gender information validation:
- Valid (Spanish): masculino, femenino, M, F, hombre, mujer, varón
- Valid (Portuguese): masculino, feminino, M, F, homem, mulher
- Invalid: Unrelated words, arbitrary letters

## Data Generation with Faker

The project uses Faker library for generating diverse, realistic synthetic data:

### Why Faker?
- **Diversity**: Generates unlimited unique names, addresses, emails
- **Locale-specific**: Realistic data per country and language
- **Realistic patterns**: Valid phone numbers, proper email formats
- **Prevents overfitting**: No repetition of hardcoded lists
- **Easy extensibility**: Simple to add more data variations

### Locales Used
```python
# Brazil - Brazilian Portuguese
Faker('pt_BR')
fake.cpf()           # Valid Brazilian CPFs
fake.cnpj()          # Company IDs
fake.phone_number()  # Brazilian phone numbers
fake.name()          # Brazilian names
fake.address()       # Brazilian addresses
fake.email()         # Email addresses

# Chile - Chilean Spanish
Faker('es_CL')
fake.name()          # Chilean names
fake.address()       # Chilean addresses
fake.email()         # Email addresses
fake.city()          # Chilean cities

# Uruguay - Argentine Spanish (closest available)
Faker('es_AR')
fake.name()          # Argentine/Uruguayan names
fake.address()       # Addresses
fake.email()         # Email addresses

# Colombia - Castilian Spanish (closest available)
Faker('es_ES')
fake.name()          # Spanish/Colombian names
fake.address()       # Addresses
fake.email()         # Email addresses
```

### Dataset Statistics

**Complete Dataset**: 42,000 samples

**By Entity Type** (6,000 each):
- ID: 3,000 valid IDs + 3,000 false positives
- PHONE: 3,000 valid phones + 3,000 false positives
- EMAIL: 3,000 valid emails + 3,000 false positives
- PER: 3,000 valid names + 3,000 false positives
- LOC: 3,000 valid locations + 3,000 false positives
- DATE: 3,000 valid dates + 3,000 false positives
- SEX: 3,000 valid genders + 3,000 false positives

**By Country** (10,500 each):
- Brazil: 10,500 samples
- Chile: 10,500 samples
- Uruguay: 10,500 samples
- Colombia: 10,500 samples

## Feature Engineering

### Feature Categories (79 total)

**Format Features** (23):
- text length, word count, digit count
- alphabetic/numeric ratios
- special character positions
- capitalization patterns (UPPER, lower, Title case)
- character counts (dots, slashes, hyphens, parentheses)

**Validation Features** (3):
- deterministic validation result (valid/invalid)
- entity type classification
- country information

**Statistical Features** (5):
- digit entropy, character entropy
- character repetition patterns
- unique character ratio

**Dictionary Features** (6):
- name frequency scores
- surname presence checks
- street keyword detection
- location word presence

**Pattern Features** (14):
- email domain structure, TLD presence
- phone format compliance (separators, length)
- ID separator detection
- date format patterns
- sequential/repeated digit detection

**Entity-Specific Features** (28):
- ID: length validation, separator presence, digit patterns
- Phone: area code validation, digit ratio, separator patterns
- Email: local length, domain validation, common TLDs
- Person: all caps check, word patterns
- Location: word count, number presence, mixed formatting
- Date: valid ranges, separator types
- Gender: length constraints, valid term checks

## Validation Rules (Stage 1)

### Deterministic Validators

**ID Numbers**:
```python
validate_chile_rut(text)        # Modulo-11 checksum
validate_brazil_cpf(text)       # Double-digit verification
validate_uruguay_ci(text)       # Format and range check
validate_colombia_cc(text)      # Basic format validation
```

**Phone Numbers**:
- Country code validation
- Area code format checking
- Length validation (8-15 digits)
- Separator pattern matching

**Emails**:
- @ symbol requirement
- TLD validation
- Format compliance

**Dates**:
- Day range (1-31)
- Month range (1-12)
- Year range (1900-2033)
- Valid combinations (no Feb 30, etc.)

**Special Patterns**:
- Sequential digits (111111...)
- Repeated digits (123123...)
- Placeholder values (00000, 99999)

## Configuration Files

### `configs/country_patterns.json`
Contains country-specific patterns:
```json
{
  "chile": {
    "id_pattern": "^\\d{1,2}\\.\\d{3}\\.\\d{3}-[0-9K]$",
    "common_first_names": [...],
    "common_surnames": [...],
    "cities": [...],
    "street_types": [...]
  }
}
```

### `configs/entity_thresholds.json`
Contains ML and training configuration:
```json
{
  "id": {"threshold": 0.871, "min_samples": 100},
  "phone": {"threshold": 0.533, "min_samples": 100},
  ...
}
```

## Usage Examples

### Single Entity Validation

```python
from inference.pipeline import PII_ValidationPipeline

pipeline = PII_ValidationPipeline()

# Validate Chilean RUT
result = pipeline.validate("15.783.037-6", "ID", "CL")
# Output: {
#   'is_pii': True,
#   'confidence': 0.99,
#   'reason': 'Valid RUT checksum + high ML confidence'
# }

# Validate Brazilian CPF
result = pipeline.validate("123.456.789-09", "ID", "BR")

# Validate Email
result = pipeline.validate("user@example.com", "EMAIL", "CL")

# Validate Phone Number
result = pipeline.validate("+56 9 8765 4321", "PHONE", "CL")
```

### Batch Validation

```python
entities = [
    {"text": "15.783.037-6", "entity_type": "ID", "country": "CL"},
    {"text": "user@test.com", "entity_type": "EMAIL", "country": "CL"},
    {"text": "+56 9 8765 4321", "entity_type": "PHONE", "country": "CL"},
    {"text": "Juan García", "entity_type": "PER", "country": "CL"},
    {"text": "01/12/1990", "entity_type": "DATE", "country": "CL"}
]

results = pipeline.validate_batch(entities)
for result in results:
    print(f"{result['text']}: {result['is_pii']} ({result['confidence']:.2%})")
```

### Integration with NER

```python
# Example: Integrate with spaCy NER output
import spacy
from inference.pipeline import PII_ValidationPipeline

nlp = spacy.load("es_core_news_sm")
pipeline = PII_ValidationPipeline()

text = "Mi RUT es 15.783.037-6 y mi email es juan@example.com"
doc = nlp(text)

for ent in doc.ents:
    result = pipeline.validate(ent.text, ent.label_, "CL")
    if result['is_pii']:
        print(f"Found PII: {ent.text} (type: {ent.label_})")
```

## Performance Tuning

### Adjusting Entity Thresholds

Edit `configs/entity_thresholds.json` to change prediction thresholds:

```json
{
  "id": {"threshold": 0.85},      // Stricter (fewer false positives)
  "phone": {"threshold": 0.50},   // Balanced
  "email": {"threshold": 0.95},   // Very strict
  "per": {"threshold": 0.30}      // Lenient (fewer false negatives)
}
```

**Threshold Strategy**:
- **High (0.80+)**: For high-confidence entity types (ID, EMAIL) - cost of false positives > false negatives
- **Medium (0.50)**: For balanced tradeoff (PHONE, DATE)
- **Low (0.30-0.40)**: For sensitive entity types (PER, LOC) - cost of false negatives > false positives

### Retraining the Model

To retrain with new data:

```bash
# Generate new dataset
python data_generation/generate_dataset.py

# Train with new parameters
python train_model.py
```

## License

This project is for educational and research purposes.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional countries (Peru, Argentina, Mexico, etc.)
- More entity types (financial data, medical IDs)
- Improved name dictionaries
- Context-aware validation
- Integration with popular NER frameworks (spaCy, HuggingFace)

## References

- **XGBoost**: https://xgboost.readthedocs.io/
- **Faker**: https://faker.readthedocs.io/
- **Feature Engineering**: Inspired by production PII validation systems

## Contact

For questions or issues, please open a GitHub issue.
