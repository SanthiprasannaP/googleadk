import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import uuid
from scipy import stats
import os
import glob

# Initialize Faker with Indian locale
fake = Faker('en_IN')

# Company name and details
COMPANY_NAME = "SecureDrive Insurance"
COMPANY_BRANCHES = [
    'Mumbai Central', 'Delhi NCR', 'Bangalore Tech Park', 'Chennai Marina',
    'Hyderabad Hitech', 'Kolkata Central', 'Pune IT Park', 'Ahmedabad Corporate',
    'Jaipur Business Park', 'Lucknow City Center'
]

# Vehicle brands and models (focusing on most common in India)
VEHICLE_BRANDS = [
    'Maruti Suzuki', 'Hyundai', 'Tata Motors', 'Mahindra', 'Toyota',
    'Honda', 'Kia', 'Volkswagen', 'Skoda', 'Renault'
]

# Market share of vehicle brands in India
VEHICLE_MARKET_SHARE = {
    'Maruti Suzuki': 0.42,  # 42% market share
    'Hyundai': 0.17,
    'Tata Motors': 0.14,
    'Mahindra': 0.08,
    'Toyota': 0.06,
    'Honda': 0.05,
    'Kia': 0.04,
    'Volkswagen': 0.02,
    'Skoda': 0.01,
    'Renault': 0.01
}

VEHICLE_MODELS = {
    'Maruti Suzuki': ['Swift', 'Baleno', 'Alto', 'WagonR', 'Ertiga'],
    'Hyundai': ['i20', 'Creta', 'Verna', 'Venue', 'Alcazar'],
    'Tata Motors': ['Nexon', 'Harrier', 'Punch', 'Altroz', 'Safari'],
    'Mahindra': ['XUV700', 'Scorpio', 'Thar', 'Bolero', 'XUV300'],
    'Toyota': ['Innova', 'Fortuner', 'Glanza', 'Urban Cruiser', 'Camry'],
    'Honda': ['City', 'Amaze', 'Jazz', 'WR-V', 'Elevate'],
    'Kia': ['Seltos', 'Sonet', 'Carens', 'Carnival', 'EV6'],
    'Volkswagen': ['Polo', 'Virtus', 'Taigun', 'Tiguan', 'ID.4'],
    'Skoda': ['Rapid', 'Kodiaq', 'Superb', 'Karoq', 'Enyaq'],
    'Renault': ['Kwid', 'Triber', 'Kiger', 'Duster', 'Captur']
}

# Model popularity within each brand (based on sales data)
MODEL_POPULARITY = {
    'Maruti Suzuki': {'Swift': 0.25, 'Baleno': 0.20, 'Alto': 0.15, 'WagonR': 0.20, 'Ertiga': 0.20},
    'Hyundai': {'i20': 0.30, 'Creta': 0.25, 'Verna': 0.15, 'Venue': 0.20, 'Alcazar': 0.10},
    'Tata Motors': {'Nexon': 0.30, 'Harrier': 0.20, 'Punch': 0.25, 'Altroz': 0.15, 'Safari': 0.10},
    'Mahindra': {'XUV700': 0.30, 'Scorpio': 0.25, 'Thar': 0.15, 'Bolero': 0.20, 'XUV300': 0.10},
    'Toyota': {'Innova': 0.40, 'Fortuner': 0.20, 'Glanza': 0.15, 'Urban Cruiser': 0.15, 'Camry': 0.10},
    'Honda': {'City': 0.35, 'Amaze': 0.25, 'Jazz': 0.15, 'WR-V': 0.15, 'Elevate': 0.10},
    'Kia': {'Seltos': 0.35, 'Sonet': 0.30, 'Carens': 0.20, 'Carnival': 0.10, 'EV6': 0.05},
    'Volkswagen': {'Polo': 0.30, 'Virtus': 0.25, 'Taigun': 0.25, 'Tiguan': 0.15, 'ID.4': 0.05},
    'Skoda': {'Rapid': 0.30, 'Kodiaq': 0.20, 'Superb': 0.20, 'Karoq': 0.20, 'Enyaq': 0.10},
    'Renault': {'Kwid': 0.35, 'Triber': 0.25, 'Kiger': 0.20, 'Duster': 0.15, 'Captur': 0.05}
}

# Policy types for motor insurance
POLICY_TYPES = ['Comprehensive', 'Third Party', 'Zero Depreciation']
POLICY_TYPE_DISTRIBUTION = {
    'Comprehensive': 0.65,    # 65% of policies
    'Third Party': 0.25,     # 25% of policies
    'Zero Depreciation': 0.10 # 10% of policies
}

CLAIM_STATUS = ['Pending', 'Approved', 'Rejected', 'Under Review', 'Settled']
CLAIM_STATUS_DISTRIBUTION = {
    'Settled': 0.60,     # 60% of claims are settled
    'Pending': 0.15,     # 15% pending
    'Under Review': 0.10, # 10% under review
    'Approved': 0.10,    # 10% approved but not settled
    'Rejected': 0.05     # 5% rejected
}

CLAIM_TYPES = ['Accident', 'Theft', 'Natural Disaster', 'Third Party Damage', 'Own Damage']
CLAIM_TYPE_DISTRIBUTION = {
    'Accident': 0.40,           # 40% of claims
    'Own Damage': 0.25,         # 25% of claims
    'Third Party Damage': 0.20, # 20% of claims
    'Natural Disaster': 0.10,   # 10% of claims
    'Theft': 0.05              # 5% of claims
}

def cleanup_previous_data():
    """Delete previously generated data files"""
    files_to_delete = ['customers.csv', 'policies.csv', 'claims.csv']
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted {file}")

def generate_realistic_age():
    """Generate age based on Indian vehicle owner demographic distribution"""
    # Using a mixture of normal distributions to model different age groups
    # Parameters based on Indian vehicle owner demographic data
    age_groups = [
        (25, 35, 0.35),  # Young professionals (35%)
        (35, 45, 0.30),  # Mid-career (30%)
        (45, 55, 0.20),  # Senior professionals (20%)
        (55, 65, 0.10),  # Pre-retirement (10%)
        (65, 75, 0.05)   # Retired (5%)
    ]
    
    weights = [w for _, _, w in age_groups]
    age_group = random.choices(age_groups, weights=weights)[0]
    
    mean = (age_group[0] + age_group[1]) / 2
    std = (age_group[1] - age_group[0]) / 4
    age = int(stats.truncnorm.rvs(
        (age_group[0] - mean) / std,
        (age_group[1] - mean) / std,
        loc=mean,
        scale=std
    ))
    return max(18, min(75, age))

def generate_realistic_income(age):
    """Generate income based on age and Indian vehicle owner income distribution"""
    base_incomes = {
        (18, 25): (400000, 800000),    # Higher base for vehicle owners
        (26, 35): (600000, 1500000),
        (36, 45): (1000000, 2500000),
        (46, 55): (1500000, 4000000),
        (56, 65): (1800000, 5000000),
        (66, 75): (1200000, 3500000)
    }
    
    for (min_age, max_age), (min_income, max_income) in base_incomes.items():
        if min_age <= age <= max_age:
            mu = np.log((min_income + max_income) / 2)
            sigma = np.log(max_income / min_income) / 4
            income = int(stats.lognorm.rvs(s=sigma, scale=np.exp(mu)))
            return max(min_income, min(max_income, income))
    
    return random.randint(400000, 5000000)

def generate_realistic_coverage_amount(vehicle_value, policy_type):
    """Generate realistic coverage amount based on vehicle value and policy type"""
    # Coverage multipliers based on policy type
    coverage_multipliers = {
        'Comprehensive': (0.9, 1.1),    # 90-110% of vehicle value
        'Third Party': (0.1, 0.2),      # 10-20% of vehicle value
        'Zero Depreciation': (1.0, 1.2)  # 100-120% of vehicle value
    }
    
    min_mult, max_mult = coverage_multipliers[policy_type]
    base_coverage = vehicle_value * random.uniform(min_mult, max_mult)
    return round(base_coverage / 10000) * 10000

def generate_vehicle_value(brand, model, year):
    """Generate realistic vehicle value based on brand, model, and year"""
    # Base values for different segments (in INR)
    base_values = {
        'Maruti Suzuki': {'Swift': 600000, 'Baleno': 700000, 'Alto': 400000, 'WagonR': 500000, 'Ertiga': 900000},
        'Hyundai': {'i20': 700000, 'Creta': 1100000, 'Verna': 900000, 'Venue': 800000, 'Alcazar': 1200000},
        'Tata Motors': {'Nexon': 800000, 'Harrier': 1500000, 'Punch': 600000, 'Altroz': 700000, 'Safari': 1600000},
        'Mahindra': {'XUV700': 1400000, 'Scorpio': 1300000, 'Thar': 1200000, 'Bolero': 800000, 'XUV300': 900000},
        'Toyota': {'Innova': 1800000, 'Fortuner': 3500000, 'Glanza': 700000, 'Urban Cruiser': 900000, 'Camry': 2000000},
        'Honda': {'City': 1100000, 'Amaze': 700000, 'Jazz': 800000, 'WR-V': 900000, 'Elevate': 1100000},
        'Kia': {'Seltos': 1100000, 'Sonet': 700000, 'Carens': 1000000, 'Carnival': 2500000, 'EV6': 6000000},
        'Volkswagen': {'Polo': 700000, 'Virtus': 1100000, 'Taigun': 1100000, 'Tiguan': 3000000, 'ID.4': 5000000},
        'Skoda': {'Rapid': 1000000, 'Kodiaq': 3500000, 'Superb': 3500000, 'Karoq': 2000000, 'Enyaq': 5500000},
        'Renault': {'Kwid': 400000, 'Triber': 600000, 'Kiger': 600000, 'Duster': 1000000, 'Captur': 1200000}
    }
    
    base_value = base_values[brand][model]
    current_year = datetime.now().year
    age_factor = 0.85 ** (current_year - year)  # 15% depreciation per year
    
    # Add some random variation (±5%)
    value = base_value * age_factor * random.uniform(0.95, 1.05)
    return round(value / 10000) * 10000

def generate_realistic_premium(coverage_amount, policy_type, age, vehicle_age, claim_history):
    """Generate realistic premium based on coverage, policy type, age, and history"""
    # Base premium rates by policy type (as percentage of coverage)
    base_rates = {
        'Comprehensive': 0.03,    # 3% of coverage
        'Third Party': 0.015,    # 1.5% of coverage
        'Zero Depreciation': 0.04 # 4% of coverage
    }
    
    # Age factor (premium increases with age)
    age_factor = 1 + (age - 25) * 0.01  # 1% increase per year after 25
    
    # Vehicle age factor (premium increases with vehicle age)
    vehicle_age_factor = 1 + vehicle_age * 0.05  # 5% increase per year
    
    # Claim history factor
    claim_history_factor = 1 + (claim_history * 0.1)  # 10% increase per claim
    
    # Calculate base premium
    base_premium = coverage_amount * base_rates[policy_type] * age_factor * vehicle_age_factor * claim_history_factor
    
    # Add some random variation (±5%)
    premium = base_premium * random.uniform(0.95, 1.05)
    
    # Round to nearest 100
    return round(premium / 100) * 100

def generate_customers(num_customers=1000):
    """Generate customer data with realistic Indian vehicle owner demographics"""
    customers = []
    for _ in range(num_customers):
        age = generate_realistic_age()
        income = generate_realistic_income(age)
        
        customer = {
            'customer_id': str(uuid.uuid4()),
            'name': fake.name(),
            'age': age,
            'gender': random.choice(['M', 'F']),
            'address': fake.address(),
            'city': fake.city(),
            'state': fake.state(),
            'pincode': fake.postcode(),
            'phone': fake.phone_number(),
            'email': fake.email(),
            'income': income,
            'occupation': fake.job(),
            'branch': random.choice(COMPANY_BRANCHES),
            'created_date': fake.date_between(start_date='-5y', end_date='today')
        }
        customers.append(customer)
    return pd.DataFrame(customers)

def calculate_policy_duration(policy_type, vehicle_year):
    """Calculate policy duration based on policy type and vehicle age"""
    current_year = datetime.now().year
    vehicle_age = current_year - vehicle_year
    
    if policy_type == 'Zero Depreciation':
        return 1  # Zero dep is always 1 year
    
    if policy_type == 'Third Party':
        if vehicle_age <= 1:  # New vehicle
            return 5  # 5 years for new vehicles
        else:
            return 1  # 1 year for old vehicles
    
    if policy_type == 'Comprehensive':
        if vehicle_age <= 3:  # New vehicle
            return 3  # 3 years for new vehicles
        else:
            return 1  # 1 year for old vehicles
    
    return 1  # Default to 1 year

def generate_policies(customers_df, num_policies=2000):
    """Generate motor insurance policy data"""
    policies = []
    for _ in range(num_policies):
        customer = customers_df.sample(1).iloc[0]
        
        # Select policy type based on distribution
        policy_type = random.choices(
            list(POLICY_TYPE_DISTRIBUTION.keys()),
            weights=list(POLICY_TYPE_DISTRIBUTION.values())
        )[0]
        
        # Select vehicle details
        brand = random.choices(
            list(VEHICLE_MARKET_SHARE.keys()),
            weights=list(VEHICLE_MARKET_SHARE.values())
        )[0]
        model = random.choices(
            list(MODEL_POPULARITY[brand].keys()),
            weights=list(MODEL_POPULARITY[brand].values())
        )[0]
        year = random.randint(2015, 2024)
        
        # Generate vehicle value and coverage
        vehicle_value = generate_vehicle_value(brand, model, year)
        coverage_amount = generate_realistic_coverage_amount(vehicle_value, policy_type)
        
        # Generate premium
        vehicle_age = datetime.now().year - year
        claim_history = random.randint(0, 3)  # Random number of previous claims
        premium_amount = generate_realistic_premium(
            coverage_amount, policy_type, customer['age'], 
            vehicle_age, claim_history
        )
        
        start_date = fake.date_between(start_date='-3y', end_date='today')
        
        # Calculate policy duration
        policy_duration_years = calculate_policy_duration(policy_type, year)
        end_date = start_date + timedelta(days=365 * policy_duration_years)
        
        policy = {
            'policy_id': str(uuid.uuid4()),
            'customer_id': customer['customer_id'],
            'policy_type': policy_type,
            'insurance_company': COMPANY_NAME,
            'branch': customer['branch'],
            'start_date': start_date,
            'end_date': end_date,
            'policy_duration_years': policy_duration_years,
            'premium_amount': premium_amount,
            'coverage_amount': coverage_amount,
            'vehicle_brand': brand,
            'vehicle_model': model,
            'vehicle_year': year,
            'vehicle_value': vehicle_value,
            'claim_history': claim_history,
            'status': random.choice(['Active', 'Expired', 'Cancelled']),
            'payment_type': 'Upfront',  # All payments are upfront
            'created_date': start_date
        }
        policies.append(policy)
    return pd.DataFrame(policies)

def generate_claims(policies_df, num_claims=500):
    """Generate motor insurance claim data"""
    claims = []
    for _ in range(num_claims):
        policy = policies_df.sample(1).iloc[0]
        claim_date = fake.date_between(start_date=policy['start_date'], end_date=policy['end_date'])
        
        # Select claim type based on distribution
        claim_type = random.choices(
            list(CLAIM_TYPE_DISTRIBUTION.keys()),
            weights=list(CLAIM_TYPE_DISTRIBUTION.values())
        )[0]
        
        # Generate claim amount based on policy type and vehicle value
        if claim_type == 'Theft':
            claim_amount = policy['vehicle_value']  # Full vehicle value for theft
        else:
            # For other claims, use a percentage of vehicle value
            claim_ratios = {
                'Accident': (0.3, 0.7),
                'Own Damage': (0.2, 0.5),
                'Third Party Damage': (0.1, 0.3),
                'Natural Disaster': (0.4, 0.8)
            }
            min_ratio, max_ratio = claim_ratios[claim_type]
            claim_amount = policy['vehicle_value'] * random.uniform(min_ratio, max_ratio)
        
        # Ensure claim amount doesn't exceed coverage
        claim_amount = min(round(claim_amount), policy['coverage_amount'])
        
        # Select claim status based on distribution
        status = random.choices(
            list(CLAIM_STATUS_DISTRIBUTION.keys()),
            weights=list(CLAIM_STATUS_DISTRIBUTION.values())
        )[0]
        
        # Generate settlement details if claim is settled
        settlement_date = None
        settlement_amount = None
        if status == 'Settled':
            settlement_date = claim_date + timedelta(days=random.randint(15, 90))
            # Settlement amount is typically 90-100% of claim amount
            settlement_amount = round(claim_amount * random.uniform(0.9, 1.0))

        claim = {
            'claim_id': str(uuid.uuid4()),
            'policy_id': policy['policy_id'],
            'customer_id': policy['customer_id'],
            'claim_type': claim_type,
            'claim_date': claim_date,
            'claim_amount': claim_amount,
            'status': status,
            # 'description': fake.text(max_nb_chars=200),
            'settlement_date': settlement_date,
            'settlement_amount': settlement_amount,
            'branch': policy['branch']
        }
        claims.append(claim)
    return pd.DataFrame(claims)

def generate_all_data():
    """Generate all motor insurance data tables"""
    print("Cleaning up previous data...")
    cleanup_previous_data()
    
    print("Generating customer data...")
    customers_df = generate_customers()
    
    print("Generating policy data...")
    policies_df = generate_policies(customers_df)
    
    print("Generating claims data...")
    claims_df = generate_claims(policies_df)
    
    # Save to CSV files
    customers_df.to_csv('customers.csv', index=False)
    policies_df.to_csv('policies.csv', index=False)
    claims_df.to_csv('claims.csv', index=False)
    
    print("Data generation complete! Files saved as customers.csv, policies.csv, and claims.csv")
    return customers_df, policies_df, claims_df

if __name__ == "__main__":
    generate_all_data() 