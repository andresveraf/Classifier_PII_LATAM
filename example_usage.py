"""
Example usage of the PII Validation Classifier.
Demonstrates various use cases and integration patterns.
"""

from inference.pipeline import PII_ValidationPipeline
import pandas as pd


def example_single_validation():
    """Example: Validate a single entity."""
    print("=" * 60)
    print("Example 1: Single Entity Validation")
    print("=" * 60)
    
    pipeline = PII_ValidationPipeline()
    
    # Test a Chilean RUT
    result = pipeline.validate(
        text="15.783.037-6",
        entity_type="ID",
        country="CL"
    )
    
    print(f"\nInput: 15.783.037-6 (Chilean RUT)")
    print(f"Is PII: {result['is_pii']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Validation Path: {result['validation_path']}")
    print(f"Reason: {result['reason']}")
    print(f"\nDetails:")
    for key, value in result['details'].items():
        print(f"  {key}: {value}")


def example_batch_validation():
    """Example: Validate multiple entities at once."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Validation")
    print("=" * 60)
    
    pipeline = PII_ValidationPipeline()
    
    entities = [
        {"text": "15.783.037-6", "entity_type": "ID", "country": "CL"},
        {"text": "andres.vera@gmail.com", "entity_type": "EMAIL", "country": "CL"},
        {"text": "+56 9 8765 4321", "entity_type": "PHONE", "country": "CL"},
        {"text": "Juan Pérez González", "entity_type": "PER", "country": "CL"},
        {"text": "test@test", "entity_type": "EMAIL", "country": "CL"},  # Invalid
        {"text": "1111111111", "entity_type": "PHONE", "country": "CL"},  # Invalid
    ]
    
    results = pipeline.validate_batch(entities)
    
    print("\nValidation Results:")
    print("-" * 60)
    for entity, result in zip(entities, results):
        status = "✓ PII" if result['is_pii'] else "✗ NOT PII"
        print(f"{entity['text']:30s} | {status:10s} | Conf: {result['confidence']:.2%}")


def example_multi_country():
    """Example: Validate entities from different countries."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Country Validation")
    print("=" * 60)
    
    pipeline = PII_ValidationPipeline()
    
    test_cases = [
        # Chile
        {"text": "15.783.037-6", "entity_type": "ID", "country": "CL", "label": "Chilean RUT"},
        
        # Brazil
        {"text": "123.456.789-09", "entity_type": "ID", "country": "BR", "label": "Brazilian CPF"},
        {"text": "+55 (11) 9 8765-4321", "entity_type": "PHONE", "country": "BR", "label": "Brazilian Mobile"},
        
        # Uruguay
        {"text": "1.234.567-8", "entity_type": "ID", "country": "UY", "label": "Uruguayan CI"},
        
        # Colombia
        {"text": "12345678", "entity_type": "ID", "country": "CO", "label": "Colombian CC"},
        {"text": "+57 310 123 4567", "entity_type": "PHONE", "country": "CO", "label": "Colombian Mobile"},
    ]
    
    print("\nResults:")
    print("-" * 80)
    print(f"{'Label':<25s} {'Text':<25s} {'Result':<12s} {'Confidence':>10s}")
    print("-" * 80)
    
    for test_case in test_cases:
        label = test_case.pop('label')
        result = pipeline.validate(**test_case)
        
        status = "✓ PII" if result['is_pii'] else "✗ NOT PII"
        print(f"{label:<25s} {test_case['text']:<25s} {status:<12s} {result['confidence']:>9.1%}")


def example_false_positives():
    """Example: Testing false positive detection."""
    print("\n" + "=" * 60)
    print("Example 4: False Positive Detection")
    print("=" * 60)
    
    pipeline = PII_ValidationPipeline()
    
    # These should be detected as NOT PII
    false_positives = [
        {"text": "FAC-12345678", "entity_type": "ID", "country": "CL", "description": "Invoice number"},
        {"text": "ORD-123456-7", "entity_type": "ID", "country": "CL", "description": "Order number"},
        {"text": "Santiago", "entity_type": "PER", "country": "CL", "description": "City name"},
        {"text": "Nike", "entity_type": "PER", "country": "CL", "description": "Brand name"},
        {"text": "Centro", "entity_type": "LOC", "country": "CL", "description": "Generic location"},
        {"text": "32/13/2023", "entity_type": "DATE", "country": "CL", "description": "Invalid date"},
        {"text": "test@test", "entity_type": "EMAIL", "country": "CL", "description": "Test email"},
    ]
    
    print("\nFalse Positive Tests (should all be NOT PII):")
    print("-" * 80)
    
    correct = 0
    for test_case in false_positives:
        desc = test_case.pop('description')
        result = pipeline.validate(**test_case)
        
        is_correct = not result['is_pii']
        correct += is_correct
        
        status = "✗ NOT PII" if not result['is_pii'] else "✓ PII"
        check = "✓" if is_correct else "✗ WRONG"
        
        print(f"{check} {desc:<20s} | {test_case['text']:<20s} | {status}")
    
    print(f"\nCorrectly identified: {correct}/{len(false_positives)} ({correct/len(false_positives)*100:.0f}%)")


def example_dataframe_integration():
    """Example: Integrate with pandas DataFrame (typical NER output)."""
    print("\n" + "=" * 60)
    print("Example 5: DataFrame Integration")
    print("=" * 60)
    
    # Simulate NER model output
    ner_output = pd.DataFrame([
        {"entity": "15.783.037-6", "entity_type": "ID", "country": "CL"},
        {"entity": "andres.vera@gmail.com", "entity_type": "EMAIL", "country": "CL"},
        {"entity": "test@test", "entity_type": "EMAIL", "country": "CL"},
        {"entity": "Juan Pérez", "entity_type": "PER", "country": "CL"},
        {"entity": "Santiago", "entity_type": "PER", "country": "CL"},
        {"entity": "+56 9 8765 4321", "entity_type": "PHONE", "country": "CL"},
    ])
    
    print("\nOriginal NER Output:")
    print(ner_output)
    
    # Validate all entities
    pipeline = PII_ValidationPipeline()
    
    entities = ner_output.to_dict('records')
    entities = [{"text": e["entity"], "entity_type": e["entity_type"], "country": e["country"]} 
                for e in entities]
    
    results = pipeline.validate_batch(entities)
    
    # Add validation results to DataFrame
    ner_output['is_pii'] = [r['is_pii'] for r in results]
    ner_output['confidence'] = [r['confidence'] for r in results]
    ner_output['validation_path'] = [r['validation_path'] for r in results]
    
    # Filter to keep only validated PII
    validated_pii = ner_output[ner_output['is_pii']].copy()
    
    print("\n\nAfter PII Validation (filtered):")
    print(validated_pii)
    
    print(f"\n\nSummary:")
    print(f"  Original entities: {len(ner_output)}")
    print(f"  Validated as PII: {len(validated_pii)}")
    print(f"  Filtered out (false positives): {len(ner_output) - len(validated_pii)}")


if __name__ == "__main__":
    # Run all examples
    example_single_validation()
    example_batch_validation()
    example_multi_country()
    example_false_positives()
    example_dataframe_integration()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
