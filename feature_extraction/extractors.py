"""
Comprehensive feature extraction for PII validation.
Extracts 50-100 features from entity text for XGBoost classification.
"""

import re
import json
import math
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any
import sys
sys.path.append(str(Path(__file__).parent.parent))

from validation_rules.deterministic import validate_entity, ValidationResult


class FeatureExtractor:
    """Extract features from entity text for PII classification."""
    
    def __init__(self, config_path: str = None):
        """Initialize with country patterns config."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "country_patterns.json"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Build name dictionaries
        self.name_dict = self._build_name_dictionary()
        self.surname_dict = self._build_surname_dictionary()
    
    def _build_name_dictionary(self) -> Dict[str, float]:
        """Build frequency dictionary of first names across all countries."""
        name_freq = {}
        for country_data in [self.config['chile'], self.config['uruguay'], 
                            self.config['colombia'], self.config['brazil']]:
            for idx, name in enumerate(country_data['common_first_names']):
                # Higher score for more common names (inverse of position)
                score = 1.0 - (idx / len(country_data['common_first_names']))
                name_freq[name.lower()] = max(name_freq.get(name.lower(), 0), score)
        return name_freq
    
    def _build_surname_dictionary(self) -> Dict[str, float]:
        """Build frequency dictionary of surnames across all countries."""
        surname_freq = {}
        for country_data in [self.config['chile'], self.config['uruguay'], 
                             self.config['colombia'], self.config['brazil']]:
            for idx, surname in enumerate(country_data['common_surnames']):
                score = 1.0 - (idx / len(country_data['common_surnames']))
                surname_freq[surname.lower()] = max(surname_freq.get(surname.lower(), 0), score)
        return surname_freq
    
    def extract_features(self, text: str, entity_type: str, country: str) -> Dict[str, Any]:
        """
        Extract all features from entity text.
        
        Args:
            text: The entity text to extract features from
            entity_type: One of PER, LOC, ID, PHONE, EMAIL, SEX, DATE
            country: Country code (CL, UY, CO, BR)
        
        Returns:
            Dictionary of feature names to values
        """
        # Handle edge cases: None or empty text
        if text is None or not str(text).strip():
            return self._get_default_features(entity_type, country)
        
        text = str(text).strip()
        
        features = {}
        
        # Format features
        features.update(self._extract_format_features(text))
        
        # Validation features
        features.update(self._extract_validation_features(text, entity_type, country))
        
        # Statistical features
        features.update(self._extract_statistical_features(text))
        
        # Dictionary features
        features.update(self._extract_dictionary_features(text, entity_type))
        
        # Pattern features
        features.update(self._extract_pattern_features(text))
        
        # Entity-specific features
        features.update(self._extract_entity_specific_features(text, entity_type, country))
        
        # Categorical encodings
        features['entity_type'] = entity_type
        features['country'] = country
        
        return features
    
    def _extract_format_features(self, text: str) -> Dict[str, Any]:
        """Extract basic format features."""
        return {
            'length': len(text),
            'word_count': len(text.split()),
            'digit_count': sum(c.isdigit() for c in text),
            'alpha_count': sum(c.isalpha() for c in text),
            'upper_count': sum(c.isupper() for c in text),
            'lower_count': sum(c.islower() for c in text),
            'space_count': text.count(' '),
            'dot_count': text.count('.'),
            'dash_count': text.count('-'),
            'underscore_count': text.count('_'),
            'at_count': text.count('@'),
            'slash_count': text.count('/'),
            'paren_count': text.count('(') + text.count(')'),
            'digit_ratio': sum(c.isdigit() for c in text) / len(text) if len(text) > 0 else 0,
            'alpha_ratio': sum(c.isalpha() for c in text) / len(text) if len(text) > 0 else 0,
            'upper_ratio': sum(c.isupper() for c in text) / len(text) if len(text) > 0 else 0,
            'special_char_count': sum(not c.isalnum() and not c.isspace() for c in text),
            'has_numbers': 1 if any(c.isdigit() for c in text) else 0,
            'has_letters': 1 if any(c.isalpha() for c in text) else 0,
            'starts_with_digit': 1 if text and text[0].isdigit() else 0,
            'starts_with_upper': 1 if text and text[0].isupper() else 0,
            'ends_with_digit': 1 if text and text[-1].isdigit() else 0,
        }
    
    def _extract_validation_features(self, text: str, entity_type: str, country: str) -> Dict[str, Any]:
        """Extract validation-based features."""
        result, reason = validate_entity(text, entity_type, country)
        
        return {
            'validation_valid': 1 if result == ValidationResult.VALID else 0,
            'validation_invalid': 1 if result == ValidationResult.INVALID else 0,
            'validation_ambiguous': 1 if result == ValidationResult.AMBIGUOUS else 0,
        }
    
    def _extract_statistical_features(self, text: str) -> Dict[str, Any]:
        """Extract statistical features about character distribution."""
        if len(text) == 0:
            return {
                'char_entropy': 0,
                'digit_entropy': 0,
                'char_repetition_max': 0,
                'char_repetition_mean': 0,
            }
        
        # Character entropy
        char_counts = Counter(text.lower())
        total_chars = len(text)
        char_entropy = -sum((count/total_chars) * math.log2(count/total_chars) 
                           for count in char_counts.values())
        
        # Digit entropy
        digits = [c for c in text if c.isdigit()]
        if digits:
            digit_counts = Counter(digits)
            total_digits = len(digits)
            digit_entropy = -sum((count/total_digits) * math.log2(count/total_digits) 
                                for count in digit_counts.values())
        else:
            digit_entropy = 0
        
        # Character repetition
        repetitions = [count for count in char_counts.values()]
        
        return {
            'char_entropy': char_entropy,
            'digit_entropy': digit_entropy,
            'char_repetition_max': max(repetitions) if repetitions else 0,
            'char_repetition_mean': sum(repetitions) / len(repetitions) if repetitions else 0,
            'unique_char_ratio': len(char_counts) / len(text),
        }
    
    def _extract_dictionary_features(self, text: str, entity_type: str) -> Dict[str, Any]:
        """Extract dictionary-based features."""
        words = text.lower().split()
        
        # Name features
        name_scores = [self.name_dict.get(word, 0) for word in words]
        surname_scores = [self.surname_dict.get(word, 0) for word in words]
        
        # Street keywords
        street_keywords = ['avenida', 'calle', 'pasaje', 'rua', 'carrera', 
                          'av', 'cl', 'psj', 'bulevar', 'rambla']
        has_street_keyword = any(word in street_keywords for word in words)
        
        # Common place words
        place_words = ['centro', 'norte', 'sur', 'este', 'oeste', 'provincia']
        has_place_word = any(word in place_words for word in words)
        
        return {
            'max_name_score': max(name_scores) if name_scores else 0,
            'max_surname_score': max(surname_scores) if surname_scores else 0,
            'mean_name_score': sum(name_scores) / len(name_scores) if name_scores else 0,
            'mean_surname_score': sum(surname_scores) / len(surname_scores) if surname_scores else 0,
            'has_street_keyword': 1 if has_street_keyword else 0,
            'has_place_word': 1 if has_place_word else 0,
        }
    
    def _extract_pattern_features(self, text: str) -> Dict[str, Any]:
        """Extract regex pattern features."""
        return {
            # Email patterns
            'has_at_symbol': 1 if '@' in text else 0,
            'has_domain_pattern': 1 if re.search(r'\.[a-z]{2,}$', text.lower()) else 0,
            
            # Phone patterns
            'has_plus_prefix': 1 if text.startswith('+') else 0,
            'has_country_code': 1 if re.match(r'^\+?\d{2,3}', text) else 0,
            
            # ID patterns
            'has_checksum_pattern': 1 if re.search(r'-[0-9Kk]$', text) else 0,
            'has_dot_separator': 1 if '.' in text and any(c.isdigit() for c in text) else 0,
            
            # Date patterns
            'matches_date_format': 1 if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text) else 0,
            
            # General patterns
            'all_caps': 1 if text.isupper() and len(text) > 1 else 0,
            'title_case': 1 if text.istitle() else 0,
            'all_lowercase': 1 if text.islower() and len(text) > 1 else 0,
            'mixed_case': 1 if any(c.isupper() for c in text) and any(c.islower() for c in text) else 0,
            
            # Sequential/repetitive patterns
            'has_sequential_digits': 1 if re.search(r'(0123|1234|2345|3456|4567|5678|6789)', text) else 0,
            'has_repeated_digits': 1 if re.search(r'(\d)\1{2,}', text) else 0,
            'all_same_char': 1 if len(set(text)) == 1 and len(text) > 1 else 0,
        }
    
    def _extract_entity_specific_features(self, text: str, entity_type: str, country: str) -> Dict[str, Any]:
        """Extract features specific to entity type."""
        features = {}
        
        if entity_type == "ID":
            features.update(self._extract_id_features(text, country))
        elif entity_type == "PHONE":
            features.update(self._extract_phone_features(text, country))
        elif entity_type == "EMAIL":
            features.update(self._extract_email_features(text))
        elif entity_type == "PER":
            features.update(self._extract_person_features(text))
        elif entity_type == "LOC":
            features.update(self._extract_location_features(text))
        elif entity_type == "DATE":
            features.update(self._extract_date_features(text))
        elif entity_type == "SEX":
            features.update(self._extract_gender_features(text))
        
        return features
    
    def _extract_id_features(self, text: str, country: str) -> Dict[str, Any]:
        """Extract ID-specific features."""
        clean_text = text.replace(".", "").replace("-", "").replace(" ", "")
        
        return {
            'id_length_valid': 1 if 7 <= len(clean_text) <= 12 else 0,
            'id_has_separator': 1 if '-' in text else 0,
            'id_numeric_only': 1 if clean_text.replace('K', '').replace('k', '').isdigit() else 0,
            'id_has_k_suffix': 1 if text.upper().endswith('K') else 0,
        }
    
    def _extract_phone_features(self, text: str, country: str) -> Dict[str, Any]:
        """Extract phone-specific features."""
        clean = text.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        
        return {
            'phone_length_valid': 1 if 8 <= len(clean) <= 15 else 0,
            'phone_has_separators': 1 if any(c in text for c in [' ', '-', '(', ')']) else 0,
            'phone_mostly_digits': 1 if (len(clean) > 0 and sum(c.isdigit() for c in clean) / len(clean) > 0.8) else 0,
        }
    
    def _extract_email_features(self, text: str) -> Dict[str, Any]:
        """Extract email-specific features."""
        parts = text.split('@')
        
        features = {
            'email_has_single_at': 1 if text.count('@') == 1 else 0,
            'email_local_length': len(parts[0]) if len(parts) > 0 else 0,
            'email_has_tld': 1 if len(parts) == 2 and '.' in parts[1] else 0,
        }
        
        if len(parts) == 2:
            domain = parts[1]
            features['email_domain_length'] = len(domain)
            features['email_has_common_tld'] = 1 if domain.endswith(('.com', '.cl', '.br', '.co', '.uy')) else 0
        else:
            features['email_domain_length'] = 0
            features['email_has_common_tld'] = 0
        
        return features
    
    def _extract_person_features(self, text: str) -> Dict[str, Any]:
        """Extract person name-specific features."""
        words = text.split()
        
        return {
            'per_word_count': len(words),
            'per_avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'per_all_capitalized': 1 if all(w[0].isupper() for w in words if w) else 0,
            'per_has_multiple_words': 1 if len(words) >= 2 else 0,
        }
    
    def _extract_location_features(self, text: str) -> Dict[str, Any]:
        """Extract location-specific features."""
        words = text.split()
        
        return {
            'loc_word_count': len(words),
            'loc_has_numbers': 1 if any(c.isdigit() for c in text) else 0,
            'loc_has_comma': 1 if ',' in text else 0,
            'loc_mixed_format': 1 if any(c.isdigit() for c in text) and any(c.isalpha() for c in text) else 0,
        }
    
    def _extract_date_features(self, text: str) -> Dict[str, Any]:
        """Extract date-specific features."""
        return {
            'date_has_slashes': 1 if '/' in text else 0,
            'date_has_dashes': 1 if '-' in text else 0,
            'date_has_dots': 1 if '.' in text else 0,
            'date_numeric_only': 1 if all(c.isdigit() or c in '/-.' for c in text) else 0,
        }
    
    def _extract_gender_features(self, text: str) -> Dict[str, Any]:
        """Extract gender-specific features."""
        return {
            'sex_very_short': 1 if len(text) <= 2 else 0,
            'sex_single_char': 1 if len(text) == 1 else 0,
            'sex_reasonable_length': 1 if 1 <= len(text) <= 15 else 0,
        }
    
    def _get_default_features(self, entity_type: str, country: str) -> Dict[str, Any]:
        """
        Return default features for None/empty text.
        All numeric features set to 0, categorical features set correctly.
        """
        return {
            'entity_type': entity_type.upper(),
            'country': country.upper(),
            'text_length': 0,
            'num_digits': 0,
            'num_letters': 0,
            'num_spaces': 0,
            'num_special_chars': 0,
            'digit_ratio': 0.0,
            'letter_ratio': 0.0,
            'space_ratio': 0.0,
            'special_char_ratio': 0.0,
            'has_uppercase': 0,
            'has_lowercase': 0,
            'uppercase_ratio': 0.0,
            'has_hyphen': 0,
            'has_dot': 0,
            'has_comma': 0,
            'has_slash': 0,
            'has_underscore': 0,
            'has_at_sign': 0,
            'num_words': 0,
            'avg_word_length': 0.0,
            'max_word_length': 0,
            'min_word_length': 0,
            'starts_with_digit': 0,
            'ends_with_digit': 0,
            'starts_with_letter': 0,
            'ends_with_letter': 0,
            'all_digits': 0,
            'all_letters': 0,
            'alphanumeric_only': 0,
            'digit_blocks': 0,
            'max_digit_block_length': 0,
            'letter_blocks': 0,
            'max_letter_block_length': 0,
            'alternating_digit_letter': 0,
            'digit_entropy': 0.0,
            'char_entropy': 0.0,
            'deterministic_valid': 0,
            'deterministic_invalid': 0,
            'deterministic_unknown': 1,
            'name_in_dict': 0,
            'surname_in_dict': 0,
            'name_score': 0.0,
            'surname_score': 0.0,
            'combined_name_score': 0.0,
            'proper_case': 0,
            'all_caps': 0,
            'contains_title': 0,
            'contains_common_name': 0,
            'contains_common_surname': 0,
            'phone_pattern_match': 0,
            'phone_country_code': 0,
            'phone_length_valid': 0,
            'phone_has_plus': 0,
            'phone_has_parentheses': 0,
            'phone_mostly_digits': 0,
            'email_has_at': 0,
            'email_has_dot': 0,
            'email_domain_valid': 0,
            'email_username_length': 0,
            'email_domain_length': 0,
            'email_common_domain': 0,
            'date_has_slash': 0,
            'date_has_hyphen': 0,
            'date_pattern_match': 0,
            'date_reasonable_day': 0,
            'date_reasonable_month': 0,
            'date_reasonable_year': 0,
            'sex_very_short': 0,
            'sex_single_char': 0,
            'sex_reasonable_length': 0,
        }


def extract_features_batch(texts: List[str], entity_types: List[str], 
                          countries: List[str]) -> List[Dict[str, Any]]:
    """
    Extract features for a batch of entities.
    
    Args:
        texts: List of entity texts
        entity_types: List of entity types
        countries: List of country codes
    
    Returns:
        List of feature dictionaries
    """
    extractor = FeatureExtractor()
    features_list = []
    
    for text, entity_type, country in zip(texts, entity_types, countries):
        features = extractor.extract_features(text, entity_type, country)
        features_list.append(features)
    
    return features_list
