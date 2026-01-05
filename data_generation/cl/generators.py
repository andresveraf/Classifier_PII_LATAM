"""
Data generators for Chile (CL) using Faker for diversity.
Generates TRUE and FALSE PII examples for all entity types.
"""

import random
import json
from pathlib import Path
from typing import List, Tuple, Dict
from faker import Faker


class ChileDataGenerator:
    """Generate synthetic PII data for Chile using Faker."""
    
    def __init__(self):
        config_path = Path(__file__).parent.parent.parent / "configs" / "country_patterns.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.patterns = config['chile']
        self.country_code = "CL"
        self.fake = Faker('es_CL')
        Faker.seed(None)  # Use different data each time
    
    def generate_rut(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Chilean RUT numbers."""
        results = []
        
        for _ in range(count):
            if valid:
                # Generate valid RUT with correct checksum
                number = random.randint(self.patterns['id_patterns']['min_value'], 
                                       self.patterns['id_patterns']['max_value'])
                check_digit = self._calculate_rut_check_digit(number)
                
                # Random formatting
                if random.random() < 0.5:
                    rut = f"{number:,}".replace(',', '.') + f"-{check_digit}"
                else:
                    rut = f"{number}-{check_digit}"
                
                results.append((rut, True))
            else:
                # Generate FALSE positives
                choice = random.randint(0, 4)
                
                if choice == 0:
                    # Invalid checksum
                    number = random.randint(10000000, 30000000)
                    wrong_check = random.choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'K'])
                    rut = f"{number}-{wrong_check}"
                elif choice == 1:
                    # Out of range
                    number = random.randint(100000, 999999)
                    check = self._calculate_rut_check_digit(number)
                    rut = f"{number}-{check}"
                elif choice == 2:
                    # Invoice-like number
                    rut = f"FAC-{random.randint(10000000, 99999999)}"
                elif choice == 3:
                    # Order number
                    rut = f"ORD-{random.randint(1000000, 9999999)}-{random.randint(1, 9)}"
                else:
                    # Random number with dashes
                    rut = f"{random.randint(1000000, 9999999)}-{random.randint(10, 99)}"
                
                results.append((rut, False))
        
        return results
    
    def _calculate_rut_check_digit(self, number: int) -> str:
        """Calculate RUT check digit using modulo-11."""
        reversed_digits = str(number)[::-1]
        multipliers = [2, 3, 4, 5, 6, 7]
        total = sum(int(digit) * multipliers[i % 6] for i, digit in enumerate(reversed_digits))
        
        remainder = total % 11
        check = 11 - remainder
        
        if check == 11:
            return "0"
        elif check == 10:
            return "K"
        else:
            return str(check)
    
    def generate_phones(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Chilean phone numbers."""
        results = []
        
        for _ in range(count):
            if valid:
                # Generate valid phone
                if random.random() < 0.7:
                    # Mobile
                    phone = f"+56 9 {random.randint(1000, 9999)} {random.randint(1000, 9999)}"
                else:
                    # Landline
                    area_code = random.choice(self.patterns['phone_patterns']['area_codes'])
                    phone = f"+56 {area_code} {random.randint(1000, 9999)} {random.randint(1000, 9999)}"
                
                results.append((phone, True))
            else:
                # Generate FALSE positives
                choice = random.randint(0, 3)
                
                if choice == 0:
                    # Sequential numbers
                    phone = "1234567890"
                elif choice == 1:
                    # Repeated digits
                    digit = random.choice(['1', '2', '3', '5', '7', '9'])
                    phone = digit * 10
                elif choice == 2:
                    # Invalid area code
                    phone = f"+56 99 {random.randint(1000, 9999)} {random.randint(1000, 9999)}"
                else:
                    # Account number-like
                    phone = f"{random.randint(10000000, 99999999)}"
                
                results.append((phone, False))
        
        return results
    
    def generate_names(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Chilean person names using Faker."""
        results = []
        
        for _ in range(count):
            if valid:
                # Use Faker for realistic Chilean names
                name = self.fake.name()
                results.append((name, True))
            else:
                # Generate FALSE positives
                choice = random.randint(0, 5)
                
                if choice == 0:
                    # City name from Faker
                    name = self.fake.city()
                elif choice == 1:
                    # Company name (first word)
                    name = self.fake.company().split()[0]
                elif choice == 2:
                    # Title without name
                    name = random.choice(['Doctor', 'Ingeniero', 'Profesor', 'Señor', 'Doctora'])
                elif choice == 3:
                    # Occupation
                    name = random.choice(['Ingeniero Civil', 'Médico Cirujano', 'Contador Auditor'])
                elif choice == 4:
                    # Common noun
                    name = random.choice(['Casa', 'Mesa', 'Libro', 'Empresa', 'Oficina'])
                else:
                    # Region/province
                    name = random.choice(['Santiago', 'Valparaíso', 'Concepción', 'Antofagasta'])
                
                results.append((name, False))
        
        return results
    
    def generate_addresses(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Chilean addresses using Faker."""
        results = []
        
        for _ in range(count):
            if valid:
                # Use Faker for realistic addresses
                address = self.fake.address().replace('\n', ', ')
                results.append((address, True))
            else:
                # Generate FALSE positives
                choice = random.randint(0, 4)
                
                if choice == 0:
                    # Just city name
                    address = self.fake.city()
                elif choice == 1:
                    # Generic location
                    address = random.choice(['Centro', 'Zona Norte', 'Sector Oriente', 'Barrio Alto'])
                elif choice == 2:
                    # Business name
                    address = random.choice(['Centro Comercial', 'Mall Plaza', 'Edificio Corporativo'])
                elif choice == 3:
                    # Just commune
                    address = random.choice(self.patterns['communes'])
                else:
                    # Region only
                    address = random.choice(['Metropolitana', 'Valparaíso', 'Biobío'])
                
                results.append((address, False))
        
        return results
    
    def generate_emails(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate email addresses using Faker."""
        results = []
        
        for _ in range(count):
            if valid:
                # Use Faker for realistic emails
                email = self.fake.email()
                results.append((email, True))
            else:
                # Generate FALSE positives
                choice = random.randint(0, 4)
                
                if choice == 0:
                    # No @
                    email = "usuario.dominio.cl"
                elif choice == 1:
                    # Missing domain
                    email = "usuario@"
                elif choice == 2:
                    # Test email
                    email = "test@test.test"
                elif choice == 3:
                    # Generic placeholder
                    email = "email@email.com"
                else:
                    # Malformed
                    email = f"correo{random.randint(1, 999)}"
                
                results.append((email, False))
        
        return results
    
    def generate_dates(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate date strings using Faker."""
        results = []
        
        for _ in range(count):
            if valid:
                # Use Faker for realistic birth dates
                date_obj = self.fake.date_of_birth(minimum_age=18, maximum_age=90)
                formats = [
                    date_obj.strftime("%d/%m/%Y"),
                    date_obj.strftime("%d-%m-%Y"),
                    date_obj.strftime("%d.%m.%Y"),
                    date_obj.strftime("%Y-%m-%d")
                ]
                date = random.choice(formats)
                results.append((date, True))
            else:
                # Generate FALSE positives
                choice = random.randint(0, 4)
                
                if choice == 0:
                    # Invalid day
                    date = f"32/{random.randint(1, 12):02d}/{random.randint(1990, 2023)}"
                elif choice == 1:
                    # Historical date (too old)
                    date = "12/10/1492"
                elif choice == 2:
                    # Placeholder
                    date = "01/01/0001"
                elif choice == 3:
                    # Far future
                    date = f"01/01/{random.randint(2100, 2999)}"
                else:
                    # Partial date
                    date = f"{random.randint(1, 12)}/{random.randint(2000, 2023)}"
                
                results.append((date, False))
        
        return results
    
    def generate_gender(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate gender information."""
        results = []
        valid_terms = ['masculino', 'femenino', 'M', 'F', 'hombre', 'mujer', 'varón']
        
        for _ in range(count):
            if valid:
                term = random.choice(valid_terms)
                results.append((term, True))
            else:
                # Generate FALSE positives
                choice = random.randint(0, 2)
                
                if choice == 0:
                    # Word containing gender term
                    term = random.choice(['fermentación', 'mascarilla', 'feminista'])
                elif choice == 1:
                    # Random letter
                    term = random.choice(['A', 'B', 'X', 'Y', 'Z'])
                else:
                    # Other word
                    term = random.choice(['indefinido', 'otro', 'no aplica'])
                
                results.append((term, False))
        
        return results
