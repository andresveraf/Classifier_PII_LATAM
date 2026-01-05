"""
Data generators for Colombia (CO) using Faker for diversity.
Generates TRUE and FALSE PII examples for all entity types.
"""

import random
import json
from pathlib import Path
from typing import List, Tuple
from faker import Faker


class ColombiaDataGenerator:
    """Generate synthetic PII data for Colombia using Faker."""
    
    def __init__(self):
        config_path = Path(__file__).parent.parent.parent / "configs" / "country_patterns.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.patterns = config['colombia']
        self.country_code = "CO"
        self.fake = Faker('es_ES')  # Using es_ES as es_CO is not available
        Faker.seed(None)
    
    def generate_cc(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Colombian CC (Cédula) numbers."""
        results = []
        
        for _ in range(count):
            if valid:
                number = random.randint(self.patterns['id_patterns']['min_value'], 
                                       self.patterns['id_patterns']['max_value'])
                cc = str(number)
                results.append((cc, True))
            else:
                choice = random.randint(0, 3)
                if choice == 0:
                    cc = f"FAC-{random.randint(10000000, 99999999)}"
                elif choice == 1:
                    cc = f"{random.randint(100000, 999999)}"
                elif choice == 2:
                    cc = f"{random.randint(2000000000, 9999999999)}"
                else:
                    cc = f"INV-{random.randint(10000000, 99999999)}"
                results.append((cc, False))
        
        return results
    
    def generate_phones(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Colombian phone numbers."""
        results = []
        
        for _ in range(count):
            if valid:
                if random.random() < 0.7:
                    phone = f"+57 3{random.randint(10, 29)} {random.randint(100, 999)} {random.randint(1000, 9999)}"
                else:
                    area = random.choice(self.patterns['phone_patterns']['area_codes'])
                    phone = f"+57 {area} {random.randint(100, 999)} {random.randint(1000, 9999)}"
                results.append((phone, True))
            else:
                phone = random.choice(['1111111111', '1234567890', str(random.randint(100000, 999999))])
                results.append((phone, False))
        
        return results
    
    def generate_names(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Colombian person names using Faker."""
        results = []
        
        for _ in range(count):
            if valid:
                name = self.fake.name()
                results.append((name, True))
            else:
                name = random.choice([
                    self.fake.city(),
                    self.fake.company().split()[0],
                    'Bogotá', 'Colombia', 'Doctor', 'Ingeniero', 'Empresa', 'Central'
                ])
                results.append((name, False))
        
        return results
    
    def generate_addresses(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Colombian addresses using Faker."""
        results = []
        
        for _ in range(count):
            if valid:
                address = self.fake.address().replace('\n', ', ')
                results.append((address, True))
            else:
                address = random.choice([
                    'Centro', 'Zona Norte', 'Centro Comercial',
                    self.fake.city(),
                    random.choice(self.patterns['cities'])
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
                email = random.choice(['test@test', 'correo@', 'ejemplo@ejemplo', 'usuario'])
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
                date = random.choice(['32/12/2020', '15/13/2020', '99/99/9999', 'fecha invalida'])
                results.append((date, False))
        return results
    
    def generate_gender(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate gender information."""
        results = []
        valid_terms = ['masculino', 'femenino', 'M', 'F', 'hombre', 'mujer', 'varón']
        
        for _ in range(count):
            if valid:
                results.append((random.choice(valid_terms), True))
            else:
                results.append((random.choice(['X', 'indefinido', 'mascarilla', 'fermentación']), False))
        return results


__all__ = ['ColombiaDataGenerator']
