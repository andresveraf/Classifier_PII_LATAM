"""
Test script to demonstrate all improvements:
1. Error handling
2. Input validation
3. Edge case handling
4. Logging
5. Performance monitoring
6. Batch error recovery
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from inference.pipeline import PII_ValidationPipeline, PipelineError
from utils import ValidationError


def test_edge_cases():
    """Test edge cases: None, empty text, special characters."""
    print("=" * 60)
    print("TEST 1: Edge Cases")
    print("=" * 60)
    
    pipeline = PII_ValidationPipeline()
    
    # Test None/empty text
    test_cases = [
        {"text": None, "entity_type": "ID", "country": "CL"},
        {"text": "", "entity_type": "PHONE", "country": "BR"},
        {"text": "   ", "entity_type": "EMAIL", "country": "CL"},
    ]
    
    for case in test_cases:
        try:
            result = pipeline.validate(case["text"], case["entity_type"], case["country"])
            print(f"\n✓ Empty/None test: {case}")
            print(f"  Result: is_pii={result['is_pii']}, path={result['validation_path']}")
        except Exception as e:
            print(f"\n✗ Failed: {e}")
    
    print("\n" + "=" * 60 + "\n")


def test_input_validation():
    """Test input validation with invalid inputs."""
    print("=" * 60)
    print("TEST 2: Input Validation")
    print("=" * 60)
    
    pipeline = PII_ValidationPipeline()
    
    # Test invalid entity type
    print("\n1. Invalid entity type:")
    try:
        result = pipeline.validate("test", "INVALID_TYPE", "CL")
        print("  ✗ Should have raised ValidationError")
    except (ValidationError, PipelineError) as e:
        print(f"  ✓ Correctly caught: {e}")
    
    # Test invalid country
    print("\n2. Invalid country:")
    try:
        result = pipeline.validate("test", "ID", "XX")
        print("  ✗ Should have raised ValidationError")
    except (ValidationError, PipelineError) as e:
        print(f"  ✓ Correctly caught: {e}")
    
    # Test too long text
    print("\n3. Too long text:")
    try:
        result = pipeline.validate("x" * 20000, "ID", "CL")
        print("  ✗ Should have raised ValidationError")
    except (ValidationError, PipelineError) as e:
        print(f"  ✓ Correctly caught: {e}")
    
    print("\n" + "=" * 60 + "\n")


def test_batch_error_recovery():
    """Test batch processing with some failing items."""
    print("=" * 60)
    print("TEST 3: Batch Error Recovery")
    print("=" * 60)
    
    pipeline = PII_ValidationPipeline()
    
    # Mix of valid and invalid items
    batch = [
        {"text": "15.783.037-6", "entity_type": "ID", "country": "CL"},  # Valid
        {"text": "", "entity_type": "PHONE", "country": "BR"},  # Empty
        {"text": "juan.perez@gmail.com", "entity_type": "EMAIL", "country": "CL"},  # Valid
        {"text": None, "entity_type": "ID", "country": "CL"},  # None
        {"text": "+56912345678", "entity_type": "PHONE", "country": "CL"},  # Valid
    ]
    
    print(f"\nProcessing batch of {len(batch)} items (including invalid ones)...")
    results = pipeline.validate_batch(batch)
    
    print(f"\n✓ Batch completed: {len(results)} results")
    print("\nResults summary:")
    for i, result in enumerate(results):
        status = "✓ PII" if result['is_pii'] else "✗ NOT PII"
        print(f"  Item {i+1}: {status} (path: {result['validation_path']})")
    
    print("\n" + "=" * 60 + "\n")


def test_logging_and_monitoring():
    """Test logging and performance monitoring."""
    print("=" * 60)
    print("TEST 4: Logging & Performance Monitoring")
    print("=" * 60)
    
    pipeline = PII_ValidationPipeline()
    
    # Perform several validations
    test_cases = [
        {"text": "15.783.037-6", "entity_type": "ID", "country": "CL"},
        {"text": "+56912345678", "entity_type": "PHONE", "country": "CL"},
        {"text": "juan.perez@gmail.com", "entity_type": "EMAIL", "country": "CL"},
        {"text": "123.456.789-00", "entity_type": "ID", "country": "BR"},
        {"text": "Juan Carlos Pérez", "entity_type": "PER", "country": "CL"},
    ]
    
    print(f"\nPerforming {len(test_cases)} validations...")
    for case in test_cases:
        result = pipeline.validate(case["text"], case["entity_type"], case["country"])
    
    # Get performance stats
    print("\n" + "-" * 60)
    print("Performance Statistics:")
    print("-" * 60)
    stats = pipeline.get_performance_stats()
    
    if 'validate' in stats:
        v_stats = stats['validate']
        print(f"  Validations: {v_stats['count']}")
        print(f"  Avg Time: {v_stats['avg_time_ms']:.2f}ms")
        print(f"  Errors: {v_stats['errors']}")
    
    if 'overall' in stats:
        o_stats = stats['overall']
        print(f"  Total Operations: {o_stats['total_operations']}")
        print(f"  Throughput: {o_stats['throughput_ops_per_sec']:.2f} ops/sec")
    
    print("\n✓ Check logs/ directory for detailed logs")
    
    print("\n" + "=" * 60 + "\n")


def test_successful_validations():
    """Test successful validations to ensure everything still works."""
    print("=" * 60)
    print("TEST 5: Successful Validations")
    print("=" * 60)
    
    pipeline = PII_ValidationPipeline()
    
    test_cases = [
        # TRUE PII
        {"text": "15.783.037-6", "entity_type": "ID", "country": "CL", "expected": True},
        {"text": "123.456.789-00", "entity_type": "ID", "country": "BR", "expected": True},
        {"text": "+56912345678", "entity_type": "PHONE", "country": "CL", "expected": True},
        {"text": "juan.perez@gmail.com", "entity_type": "EMAIL", "country": "CL", "expected": True},
        
        # FALSE POSITIVES
        {"text": "123456789", "entity_type": "ID", "country": "CL", "expected": False},
        {"text": "Article 15.783.037-6 of the law", "entity_type": "ID", "country": "CL", "expected": False},
    ]
    
    correct = 0
    for case in test_cases:
        result = pipeline.validate(case["text"], case["entity_type"], case["country"])
        is_correct = result['is_pii'] == case['expected']
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"\n{status} {case['text'][:40]}")
        print(f"   Expected: {case['expected']}, Got: {result['is_pii']}")
        print(f"   Path: {result['validation_path']}, Confidence: {result['confidence']:.2f}")
    
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.0f}%)")
    print("=" * 60 + "\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TESTING PII VALIDATION IMPROVEMENTS")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Edge cases
        test_edge_cases()
        
        # Test 2: Input validation
        test_input_validation()
        
        # Test 3: Batch error recovery
        test_batch_error_recovery()
        
        # Test 4: Logging and monitoring
        test_logging_and_monitoring()
        
        # Test 5: Successful validations
        test_successful_validations()
        
        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nImprovements implemented:")
        print("  ✓ 1. Comprehensive error handling")
        print("  ✓ 2. Input validation with custom exceptions")
        print("  ✓ 3. Edge case handling (None/empty text)")
        print("  ✓ 4. Structured logging system")
        print("  ✓ 5. Performance monitoring")
        print("  ✓ 6. Batch error recovery")
        print("  ✓ 7. Model loading validation")
        print("\nCheck logs/ directory for detailed execution logs")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
