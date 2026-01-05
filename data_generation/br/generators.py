"""
Data generators for Brazil (BR) using Faker for diversity.
Generates TRUE and FALSE PII examples for all entity types.
"""

import random
import json
from pathlib import Path
from typing import List, Tuple
from faker import Faker


class BrazilDataGenerator:
    """Generate synthetic PII data for Brazil using Faker."""
    
    def __init__(self):
        config_path = Path(__file__).parent.parent.parent / "configs" / "country_patterns.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.patterns = config['brazil']
        self.country_code = "BR"
        self.fake = Faker('pt_BR')
        Faker.seed(None)  # Use different data each time
    
    def generate_cpf(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Brazilian CPF numbers using Faker."""
        results = []
        
        for _ in range(count):
            if valid:
                # Use Faker's CPF generator for valid CPFs
                cpf = self.fake.cpf()
                results.append((cpf, True))
            else:
                # Generate FALSE positives
                choice = random.randint(0, 4)
                
                if choice == 0:
                    # Invalid checksum
                    cpf = f"{random.randint(100, 999)}.{random.randint(100, 999)}.{random.randint(100, 999)}-{random.randint(10, 99)}"
                elif choice == 1:
                    # All same digit (invalid)
                    digit = random.randint(0, 9)
                    cpf = f"{digit}{digit}{digit}.{digit}{digit}{digit}.{digit}{digit}{digit}-{digit}{digit}"
                elif choice == 2:
                    # CNPJ (company ID, not CPF)
                    cpf = f"{random.randint(10, 99)}.{random.randint(100, 999)}.{random.randint(100, 999)}/0001-{random.randint(10, 99)}"
                elif choice == 3:
                    # Random formatted number
                    cpf = f"NF-{random.randint(10000000, 99999999)}-{random.randint(10, 99)}"
                else:
                    # Wrong length
                    cpf = f"{random.randint(1000000, 9999999)}-{random.randint(10, 99)}"
                
                results.append((cpf, False))
        
        return results
    
    def generate_phones(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Brazilian phone numbers using Faker."""
        results = []
        
        for _ in range(count):
            if valid:
                # Use Faker's phone number generator
                phone = self.fake.phone_number()
                results.append((phone, True))
            else:
                # Generate FALSE positives
                choice = random.randint(0, 3)
                
                if choice == 0:
                    # Sequential
                    phone = "12345678901"
                elif choice == 1:
                    # Repeated digits
                    phone = str(random.randint(1, 9)) * 11
                elif choice == 2:
                    # Invalid area code
                    phone = f"+55 (00) 9 {random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
                else:
                    # CEP code (postal code, not phone)
                    phone = f"{random.randint(10000, 99999)}-{random.randint(100, 999)}"
                
                results.append((phone, False))
        
        return results
    
    def generate_names(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Brazilian person names using Faker."""
        results = []
        
        for _ in range(count):
            if valid:
                # Use Faker for realistic Brazilian names
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
                    # Title
                    name = random.choice(['Doutor', 'Engenheiro', 'Professor', 'Senhor', 'Dra'])
                elif choice == 3:
                    # Common word
                    name = random.choice(['Branco', 'Negro', 'Azul', 'Verde', 'Costa'])
                elif choice == 4:
                    # Occupation
                    name = random.choice(['Médico', 'Cirurgião', 'Advogado', 'Contador'])
                else:
                    # State name
                    name = self.fake.state()
                
                results.append((name, False))
        
        return results
    
    def generate_addresses(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate Brazilian addresses using Faker."""
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
                    # Just city
                    address = self.fake.city()
                elif choice == 1:
                    # Generic location
                    address = random.choice(['Centro', 'Zona Sul', 'Zona Norte'])
                elif choice == 2:
                    # Business
                    address = random.choice(['Shopping Center', 'Edifício Comercial', 'Prédio Comercial'])
                elif choice == 3:
                    # Just neighborhood
                    address = random.choice(['Copacabana', 'Ipanema', 'Jardins', 'Moema'])
                else:
                    # State only
                    address = self.fake.state()
                
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
                choice = random.randint(0, 3)
                
                if choice == 0:
                    # No @
                    email = 'usuario.dominio.com'
                elif choice == 1:
                    # No domain
                    email = 'usuario@'
                elif choice == 2:
                    # Invalid format
                    email = '@dominio.com'
                else:
                    # Wrong extension
                    email = 'usuario@dominio'
                
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
                    date_obj.strftime("%Y-%m-%d")
                ]
                date = random.choice(formats)
                results.append((date, True))
            else:
                # Generate FALSE positives
                choice = random.randint(0, 3)
                
                if choice == 0:
                    # Invalid day
                    date = f"32/12/{random.randint(1990, 2023)}"
                elif choice == 1:
                    # Invalid month
                    date = f"{random.randint(1, 28):02d}/13/{random.randint(1990, 2023)}"
                elif choice == 2:
                    # Future date
                    date = f"01/01/{random.randint(2030, 2050)}"
                else:
                    # Malformed
                    date = f"{random.randint(1, 99)}/{random.randint(1, 99)}/{random.randint(10, 99)}"
                
                results.append((date, False))
        
        return results
    
    def generate_gender(self, valid: bool = True, count: int = 100) -> List[Tuple[str, bool]]:
        """Generate gender information."""
        results = []
        valid_terms = ['masculino', 'feminino', 'M', 'F', 'homem', 'mulher']
        
        for _ in range(count):
            if valid:
                term = random.choice(valid_terms)
                results.append((term, True))
            else:
                choice = random.randint(0, 2)
                
                if choice == 0:
                    term = random.choice(['feminista', 'masculinidade'])
                elif choice == 1:
                    term = random.choice(['A', 'X', 'Y', 'Z'])
                else:
                    term = random.choice(['indefinido', 'outro'])
                
                results.append((term, False))
        
        return results


__all__ = ['BrazilDataGenerator']
