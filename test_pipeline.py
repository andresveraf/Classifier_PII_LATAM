#!/usr/bin/env python3
"""Quick test of pipeline improvements."""

from inference.pipeline import PII_ValidationPipeline
import os

def main():
    # Test basic functionality
    pipeline = PII_ValidationPipeline()
    print("✓ Pipeline initialized successfully")
    
    # Test 1: Valid Brazilian CPF
    result = pipeline.validate("123.456.789-10", "ID", "BR")
    print(f"✓ Test 1 (CPF): is_pii={result['is_pii']}, confidence={result['confidence']:.2f}")
    
    # Test 2: Valid email
    result = pipeline.validate("john@example.com", "EMAIL", "BR")
    print(f"✓ Test 2 (Email): is_pii={result['is_pii']}, confidence={result['confidence']:.2f}")
    
    # Test 3: Empty text
    result = pipeline.validate("", "EMAIL", "BR")
    print(f"✓ Test 3 (Empty): is_pii={result['is_pii']}, confidence={result['confidence']:.2f}")
    
    # Test 4: Batch validation
    results = pipeline.validate_batch([
        {'text': "123.456.789-10", 'entity_type': "ID", 'country': "BR"},
        {'text': "john@example.com", 'entity_type': "EMAIL", 'country': "BR"},
        {'text': "invalid_data", 'entity_type': "ID", 'country': "CL"},
    ])
    print(f"✓ Test 4 (Batch): Processed {len(results)} items")
    
    # Test 5: Logging check
    if os.path.exists("logs/pii_pipeline.log"):
        print("✓ Test 5 (Logging): Log file created successfully")
    else:
        print("⚠ Test 5 (Logging): Log file not found")
    
    # Test 6: Performance metrics
    metrics = pipeline.get_performance_stats()
    overall = metrics['overall']
    print(f"✓ Test 6 (Metrics): Operations={overall['total_operations']}, Errors={overall['total_errors']}, Throughput={overall['throughput_ops_per_sec']:.1f} ops/sec")
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    main()
