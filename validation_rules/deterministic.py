"""
Deterministic validation rules for PII entities.
These rules provide hard filters for obvious valid/invalid cases.
"""

import re
from datetime import datetime
from typing import Literal, Tuple
from enum import Enum


class ValidationResult(Enum):
    """Result of deterministic validation"""
    VALID = "valid"  # Definitely valid PII
    INVALID = "invalid"  # Definitely not valid PII
    AMBIGUOUS = "ambiguous"  # Cannot determine, needs ML


def validate_chile_rut(rut: str) -> ValidationResult:
    """
    Validate Chilean RUT using modulo-11 algorithm.
    Format: XX.XXX.XXX-X or XXXXXXXX-X
    """
    # Clean the RUT
    rut_clean = rut.replace(".", "").replace(" ", "").upper()
    
    if not re.match(r'^\d{7,8}-[0-9Kk]$', rut_clean):
        return ValidationResult.INVALID
    
    # Split number and check digit
    parts = rut_clean.split("-")
    if len(parts) != 2:
        return ValidationResult.INVALID
    
    number_str, check_digit = parts
    
    try:
        number = int(number_str)
    except ValueError:
        return ValidationResult.INVALID
    
    # Check range
    if number < 1000000 or number > 50000000:
        return ValidationResult.INVALID
    
    # Calculate check digit using modulo-11
    reversed_digits = str(number)[::-1]
    multipliers = [2, 3, 4, 5, 6, 7]
    total = 0
    
    for i, digit in enumerate(reversed_digits):
        multiplier = multipliers[i % 6]
        total += int(digit) * multiplier
    
    remainder = total % 11
    calculated_check = 11 - remainder
    
    if calculated_check == 11:
        expected_check = "0"
    elif calculated_check == 10:
        expected_check = "K"
    else:
        expected_check = str(calculated_check)
    
    if check_digit == expected_check:
        return ValidationResult.VALID
    else:
        return ValidationResult.INVALID


def validate_brazil_cpf(cpf: str) -> ValidationResult:
    """
    Validate Brazilian CPF using double-digit validation algorithm.
    Format: XXX.XXX.XXX-XX or XXXXXXXXXXX
    """
    # Clean the CPF
    cpf_clean = cpf.replace(".", "").replace("-", "").replace(" ", "")
    
    if not re.match(r'^\d{11}$', cpf_clean):
        return ValidationResult.INVALID
    
    # Check for known invalid CPFs (all same digit)
    if cpf_clean == cpf_clean[0] * 11:
        return ValidationResult.INVALID
    
    # Calculate first check digit
    sum1 = sum(int(cpf_clean[i]) * (10 - i) for i in range(9))
    digit1 = 11 - (sum1 % 11)
    if digit1 >= 10:
        digit1 = 0
    
    if int(cpf_clean[9]) != digit1:
        return ValidationResult.INVALID
    
    # Calculate second check digit
    sum2 = sum(int(cpf_clean[i]) * (11 - i) for i in range(10))
    digit2 = 11 - (sum2 % 11)
    if digit2 >= 10:
        digit2 = 0
    
    if int(cpf_clean[10]) != digit2:
        return ValidationResult.INVALID
    
    return ValidationResult.VALID


def validate_uruguay_ci(ci: str) -> ValidationResult:
    """
    Validate Uruguayan CI (Cédula de Identidad).
    Format: X.XXX.XXX-X or XXXXXXX-X
    Note: Uruguay uses a validation digit algorithm
    """
    # Clean the CI
    ci_clean = ci.replace(".", "").replace(" ", "")
    
    if not re.match(r'^\d{7,8}-\d$', ci_clean):
        return ValidationResult.INVALID
    
    parts = ci_clean.split("-")
    if len(parts) != 2:
        return ValidationResult.INVALID
    
    number_str, check_digit = parts
    
    try:
        number = int(number_str)
        check = int(check_digit)
    except ValueError:
        return ValidationResult.INVALID
    
    # Check range
    if number < 1000000 or number > 9000000:
        return ValidationResult.INVALID
    
    # Uruguay uses a specific validation algorithm
    # Simplified version - in production, use the exact algorithm
    digits = [int(d) for d in number_str]
    random_multipliers = [2, 9, 8, 7, 6, 3, 4]
    
    total = sum(d * m for d, m in zip(digits, random_multipliers[:len(digits)]))
    calculated_check = total % 10
    
    if calculated_check == check:
        return ValidationResult.VALID
    else:
        # Some edge cases might be ambiguous
        return ValidationResult.AMBIGUOUS


def validate_colombia_cc(cc: str) -> ValidationResult:
    """
    Validate Colombian CC (Cédula de Ciudadanía).
    Format: 8-10 digits, no check digit algorithm
    """
    # Clean the CC
    cc_clean = cc.replace(".", "").replace(",", "").replace(" ", "")
    
    if not re.match(r'^\d{8,10}$', cc_clean):
        return ValidationResult.INVALID
    
    try:
        number = int(cc_clean)
    except ValueError:
        return ValidationResult.INVALID
    
    # Check range
    if number < 1000000 or number > 1800000000:
        return ValidationResult.INVALID
    
    # No checksum algorithm exists for Colombian CC
    # If format and range are valid, it's ambiguous
    return ValidationResult.AMBIGUOUS


def validate_email(email: str) -> ValidationResult:
    """
    Validate email address using RFC-compliant regex.
    """
    # Strict RFC 5322 compliant regex (simplified)
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        return ValidationResult.INVALID
    
    # Check for common test/placeholder emails
    test_patterns = [
        r'test@test',
        r'example@example',
        r'user@test',
        r'correo@correo',
        r'email@email'
    ]
    
    email_lower = email.lower()
    for pattern in test_patterns:
        if re.search(pattern, email_lower):
            return ValidationResult.INVALID
    
    return ValidationResult.VALID


def validate_phone(phone: str, country: str) -> ValidationResult:
    """
    Validate phone number based on country-specific patterns.
    """
    # Clean phone number
    phone_clean = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    
    # Country-specific validation
    patterns = {
        "CL": [
            r'^\+?569\d{8}$',  # Chilean mobile
            r'^\+?562\d{8}$'   # Chilean landline
        ],
        "UY": [
            r'^\+?5989[1-9]\d{6}$',  # Uruguayan mobile
            r'^\+?5982\d{7}$'        # Uruguayan landline
        ],
        "CO": [
            r'^\+?573\d{9}$',   # Colombian mobile
            r'^\+?57[1-8]\d{7}$'  # Colombian landline
        ],
        "BR": [
            r'^\+?55\d{2}9\d{8}$',  # Brazilian mobile
            r'^\+?55\d{2}\d{8}$'    # Brazilian landline
        ]
    }
    
    if country not in patterns:
        return ValidationResult.AMBIGUOUS
    
    for pattern in patterns[country]:
        if re.match(pattern, phone_clean):
            return ValidationResult.VALID
    
    # Check for obviously invalid patterns
    if re.match(r'^(\d)\1+$', phone_clean):  # All same digit
        return ValidationResult.INVALID
    
    if re.match(r'^1234567890?$', phone_clean):  # Sequential
        return ValidationResult.INVALID
    
    return ValidationResult.AMBIGUOUS


def validate_date(date_str: str) -> ValidationResult:
    """
    Validate date information.
    Checks if it's a valid date and within reasonable range for personal data.
    """
    # Try different date formats
    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%Y-%m-%d",
        "%d/%m/%y",
        "%d-%m-%y"
    ]
    
    parsed_date = None
    for fmt in formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue
    
    if parsed_date is None:
        return ValidationResult.INVALID
    
    # Check if date is reasonable for personal data
    current_year = datetime.now().year
    
    # Invalid if too far in past (before 1900) or future
    if parsed_date.year < 1900 or parsed_date.year > current_year + 10:
        return ValidationResult.INVALID
    
    # Check for placeholder dates
    if date_str in ["01/01/0001", "01/01/1900", "00/00/0000", "99/99/9999"]:
        return ValidationResult.INVALID
    
    return ValidationResult.VALID


def validate_gender(text: str, language: Literal["spanish", "portuguese"]) -> ValidationResult:
    """
    Validate gender/sex information.
    """
    text_lower = text.lower().strip()
    
    valid_terms = {
        "spanish": ["masculino", "femenino", "m", "f", "hombre", "mujer", "varón"],
        "portuguese": ["masculino", "feminino", "m", "f", "homem", "mulher"]
    }
    
    if text_lower in valid_terms.get(language, []):
        return ValidationResult.VALID
    
    # Check if it's part of another word (false positive)
    if len(text) > 15:
        return ValidationResult.INVALID
    
    return ValidationResult.AMBIGUOUS


def validate_id(text: str, country: str) -> ValidationResult:
    """
    Validate identification number based on country.
    """
    validators = {
        "CL": validate_chile_rut,
        "BR": validate_brazil_cpf,
        "UY": validate_uruguay_ci,
        "CO": validate_colombia_cc
    }
    
    validator = validators.get(country)
    if validator is None:
        return ValidationResult.AMBIGUOUS
    
    return validator(text)


def validate_entity(
    text: str,
    entity_type: str,
    country: str = "CL"
) -> Tuple[ValidationResult, str]:
    """
    Main validation function for any entity type.
    
    Returns:
        Tuple of (ValidationResult, reason)
    """
    if entity_type == "ID":
        result = validate_id(text, country)
        reason = f"ID validation for {country}"
    elif entity_type == "EMAIL":
        result = validate_email(text)
        reason = "Email RFC validation"
    elif entity_type == "PHONE":
        result = validate_phone(text, country)
        reason = f"Phone validation for {country}"
    elif entity_type == "DATE":
        result = validate_date(text)
        reason = "Date format and range validation"
    elif entity_type == "SEX":
        language = "portuguese" if country == "BR" else "spanish"
        result = validate_gender(text, language)
        reason = f"Gender term validation ({language})"
    elif entity_type in ["PER", "LOC"]:
        # Names and locations are hard to validate deterministically
        result = ValidationResult.AMBIGUOUS
        reason = f"{entity_type} requires ML classification"
    else:
        result = ValidationResult.AMBIGUOUS
        reason = "Unknown entity type"
    
    return result, reason
