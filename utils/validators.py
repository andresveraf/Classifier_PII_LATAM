"""
Input validation utilities for PII classification pipeline.
Validates text inputs, entity types, and countries.
"""
import re
from typing import List, Optional


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """Validates inputs for PII classification."""
    
    VALID_ENTITY_TYPES = ['ID', 'PHONE', 'EMAIL', 'PER', 'LOC', 'DATE', 'SEX']
    VALID_COUNTRIES = ['CL', 'BR', 'UY', 'CO']
    
    @staticmethod
    def validate_text(text: str, max_length: int = 10000) -> str:
        """
        Validate and sanitize text input.
        
        Args:
            text: Input text to validate
            max_length: Maximum allowed text length
            
        Returns:
            Sanitized text
            
        Raises:
            ValidationError: If text is invalid
        """
        if not isinstance(text, str):
            raise ValidationError(f"Text must be string, got {type(text)}")
        
        if not text or not text.strip():
            raise ValidationError("Text cannot be empty")
        
        if len(text) > max_length:
            raise ValidationError(f"Text exceeds maximum length of {max_length}")
        
        # Remove null bytes and other problematic characters
        sanitized = text.replace('\x00', '')
        
        return sanitized
    
    @staticmethod
    def validate_entity_type(entity_type: str) -> str:
        """
        Validate entity type.
        
        Args:
            entity_type: Entity type to validate
            
        Returns:
            Validated entity type (uppercase)
            
        Raises:
            ValidationError: If entity type is invalid
        """
        if not isinstance(entity_type, str):
            raise ValidationError(f"Entity type must be string, got {type(entity_type)}")
        
        entity_type = entity_type.upper().strip()
        
        if entity_type not in InputValidator.VALID_ENTITY_TYPES:
            raise ValidationError(
                f"Invalid entity type '{entity_type}'. "
                f"Must be one of {InputValidator.VALID_ENTITY_TYPES}"
            )
        
        return entity_type
    
    @staticmethod
    def validate_country(country: str) -> str:
        """
        Validate country code.
        
        Args:
            country: Country code to validate
            
        Returns:
            Validated country code (uppercase)
            
        Raises:
            ValidationError: If country code is invalid
        """
        if not isinstance(country, str):
            raise ValidationError(f"Country must be string, got {type(country)}")
        
        country = country.upper().strip()
        
        if country not in InputValidator.VALID_COUNTRIES:
            raise ValidationError(
                f"Invalid country '{country}'. "
                f"Must be one of {InputValidator.VALID_COUNTRIES}"
            )
        
        return country
    
    @staticmethod
    def validate_batch(texts: List[str], entity_types: List[str], 
                      countries: List[str]) -> None:
        """
        Validate batch inputs.
        
        Args:
            texts: List of texts
            entity_types: List of entity types
            countries: List of countries
            
        Raises:
            ValidationError: If batch inputs are invalid
        """
        if not isinstance(texts, list):
            raise ValidationError("texts must be a list")
        
        if not isinstance(entity_types, list):
            raise ValidationError("entity_types must be a list")
        
        if not isinstance(countries, list):
            raise ValidationError("countries must be a list")
        
        if not texts:
            raise ValidationError("texts list cannot be empty")
        
        if len(texts) != len(entity_types) or len(texts) != len(countries):
            raise ValidationError(
                "texts, entity_types, and countries must have the same length"
            )
