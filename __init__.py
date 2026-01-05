"""
PII Classifier for LATAM Countries

A robust XGBoost-based PII validation classifier for Chile, Uruguay, 
Colombia, and Brazil. This post-processing classifier validates entities 
detected by NER models to reduce false positives.
"""

__version__ = "1.0.0"
__author__ = "PII Classifier Team"

from inference.pipeline import PII_ValidationPipeline

__all__ = ['PII_ValidationPipeline']
