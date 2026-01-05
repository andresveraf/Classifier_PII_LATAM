"""
Performance monitoring for PII classification pipeline.
Tracks latency, throughput, and error rates.
"""
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional


class PerformanceMonitor:
    """Monitors performance metrics for PII classification."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'errors': 0,
            'last_updated': None
        })
        self.start_time = time.time()
    
    def record_operation(self, operation: str, duration: float, 
                        success: bool = True):
        """
        Record an operation's performance.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            success: Whether operation succeeded
        """
        metric = self.metrics[operation]
        metric['count'] += 1
        metric['total_time'] += duration
        if not success:
            metric['errors'] += 1
        metric['last_updated'] = datetime.now()
    
    def get_stats(self, operation: Optional[str] = None) -> Dict:
        """
        Get performance statistics.
        
        Args:
            operation: Specific operation to get stats for, or None for all
            
        Returns:
            Dictionary with performance statistics
        """
        if operation:
            if operation not in self.metrics:
                return {}
            
            metric = self.metrics[operation]
            avg_time = metric['total_time'] / metric['count'] if metric['count'] > 0 else 0
            error_rate = metric['errors'] / metric['count'] if metric['count'] > 0 else 0
            
            return {
                'operation': operation,
                'count': metric['count'],
                'avg_time_ms': avg_time * 1000,
                'total_time_s': metric['total_time'],
                'errors': metric['errors'],
                'error_rate': error_rate,
                'last_updated': metric['last_updated']
            }
        
        # Return all metrics
        stats = {}
        for op, metric in self.metrics.items():
            avg_time = metric['total_time'] / metric['count'] if metric['count'] > 0 else 0
            error_rate = metric['errors'] / metric['count'] if metric['count'] > 0 else 0
            
            stats[op] = {
                'count': metric['count'],
                'avg_time_ms': avg_time * 1000,
                'total_time_s': metric['total_time'],
                'errors': metric['errors'],
                'error_rate': error_rate,
                'last_updated': metric['last_updated']
            }
        
        # Add overall stats
        total_operations = sum(m['count'] for m in self.metrics.values())
        total_errors = sum(m['errors'] for m in self.metrics.values())
        uptime = time.time() - self.start_time
        
        stats['overall'] = {
            'total_operations': total_operations,
            'total_errors': total_errors,
            'uptime_s': uptime,
            'throughput_ops_per_sec': total_operations / uptime if uptime > 0 else 0
        }
        
        return stats
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.start_time = time.time()
    
    def __str__(self) -> str:
        """String representation of performance stats."""
        stats = self.get_stats()
        lines = ["Performance Statistics:"]
        lines.append("-" * 50)
        
        for op, metrics in stats.items():
            if op == 'overall':
                lines.append(f"\nOverall:")
                lines.append(f"  Total Operations: {metrics['total_operations']}")
                lines.append(f"  Total Errors: {metrics['total_errors']}")
                lines.append(f"  Uptime: {metrics['uptime_s']:.2f}s")
                lines.append(f"  Throughput: {metrics['throughput_ops_per_sec']:.2f} ops/sec")
            else:
                lines.append(f"\n{op}:")
                lines.append(f"  Count: {metrics['count']}")
                lines.append(f"  Avg Time: {metrics['avg_time_ms']:.2f}ms")
                lines.append(f"  Errors: {metrics['errors']} ({metrics['error_rate']*100:.1f}%)")
        
        return "\n".join(lines)
