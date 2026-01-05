"""
Data generators for Uruguay (UY) using Faker for diversity.
Generates TRUE and FALSE PII examples for all entity types.
"""

import random
import json
from pathlib import Path
from typing import List, Tuple
from faker import Faker


class UruguayDataGenerator:
    """Generate synthetic PII data for Uruguay using Faker."""
    
    def __init__(self):
        config_path = Path(__file__).parent.parent.parent / "configs" / "country_patterns.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.patterns = config['uruguay']
        self.country_code = "UY"
        self.fake = Faker('es_AR')  # Using es_AR as es_UY is not available
        Faker.seed(None)
    
    def generate_ci(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Uruguayan CI (Cédula) numbers."""
        results = []
        
        for _ in range(count):
            if valid:
                number = random.randint(self.patterns['id_patterns']['min_value'], 
                                       self.patterns['id_patterns']['max_value'])
                check = self._calculate_ci_check(number)
                
                if random.random() < 0.5:
                    ci = f"{number:,}".replace(',', '.') + f"-{check}"
                else:
                    ci = f"{number}-{check}"
                
                results.append((ci, True))
            else:
                choice = random.randint(0, 3)
                if choice == 0:
                    ci = f"FAC-{random.randint(1000000, 9000000)}"
                elif choice == 1:
                    ci = f"{random.randint(100000, 999999)}-{random.randint(0, 9)}"
                elif choice == 2:
                    ci = f"{random.randint(10000000, 99999999)}-{random.randint(0, 9)}"
                else:
                    ci = f"ORD-{random.randint(1000000, 9000000)}-{random.randint(0, 9)}"
                
                results.append((ci, False))
        
        return results
    
    def _calculate_ci_check(self, number: int) -> int:
        """Simplified CI check digit calculation."""
        digits = [int(d) for d in str(number)]
        multipliers = [2, 9, 8, 7, 6, 3, 4]
        total = sum(d * m for d, m in zip(digits, multipliers[:len(digits)]))
        return total % 10
    
    def generate_phones(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Uruguayan phone numbers."""
        results = []
        
        for _ in range(count):
            if valid:
                if random.random() < 0.7:
                    phone = f"+598 9{random.randint(1, 9)} {random.randint(100, 999)} {random.randint(100, 999)}"
                else:
                    phone = f"+598 2 {random.randint(100, 999)} {random.randint(1000, 9999)}"
                results.append((phone, True))
            else:
                choice = random.randint(0, 2)
                if choice == 0:
                    phone = "1111111111"
                elif choice == 1:
                    phone = "1234567890"
                else:
                    phone = f"{random.randint(10000000, 99999999)}"
                results.append((phone, False))
        
        return results
    
    def generate_names(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Uruguayan person names using Faker."""
        results = []
        
        for _ in range(count):
            if valid:
                name = self.fake.name()
                results.append((name, True))
            else:
                name = random.choice([
                    self.fake.city(),
                    self.fake.company().split()[0],
                    'Montevideo', 'Uruguay', 'Doctor', 'Ingeniero', 'Centro', 'Plaza'
                ])
                results.append((name, False))
        
        return results
    
    def generate_addresses(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Uruguayan addresses using Faker."""
        results = []
        
        for _ in range(count):
            if valid:
                address = self.fake.address().replace('\n', ', ')
                results.append((address, True))
            else:
                address = random.choice([
                    'Centro', 'Zona Norte', 'Shopping',
                    self.fake.city(),
                    random.choice(self.patterns['neighborhoods'])
                ])
                results.append((address, False))
        
        return results
    
    def generate_emails(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate email addresses using Faker."""
        results = []
        
        for _ in range(count):
            if valid:
                email = self.fake.email()
                results.append((email, True))
            else:
                email = random.choice(['test@test', 'correo@', 'ejemplo@ejemplo.com', 'usuario'])
                results.append((email, False))
        
        return results
    
    def generate_dates(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate date strings using Faker."""
        results = []
        for _ in range(count):
            if valid:
                date_obj = self.fake.date_of_birth(minimum_age=18, maximum_age=90)
                date = date_obj.strftime("%d/%m/%Y")
                results.append((date, True))
            else:
                date = random.choice(['32/01/2020', '01/13/2020', '99/99/99', 'fecha'])
                results.append((date, False))
        return results
    
    def generate_gender(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate gender information."""
        results = []
        valid_terms = ['masculino', 'femenino', 'M', 'F', 'hombre', 'mujer']
        
        for _ in range(count):
            if valid:
                results.append((random.choice(valid_terms), True))
            else:
                results.append((random.choice(['X', 'Y', 'indefinido', 'fermentación']), False))
        return results


__all__ = ['UruguayDataGenerator']
