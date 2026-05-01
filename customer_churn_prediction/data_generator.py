import pandas as pd
import numpy as np

def generate_telecom_churn_data(n_samples=5000):
    np.random.seed(42)
    
    # 1. Customer ID
    customer_ids = [f'{i:04d}-ABCD' for i in range(1, n_samples + 1)]
    
    # 2. Demographics
    gender = np.random.choice(['Male', 'Female'], n_samples)
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    partner = np.random.choice(['Yes', 'No'], n_samples)
    dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    
    # 3. Account Information
    tenure = np.random.randint(1, 73, n_samples)  # 1 to 72 months
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.20, 0.25])
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
    payment_method = np.random.choice([
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ], n_samples, p=[0.35, 0.25, 0.2, 0.2])
    
    # 4. Services
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1])
    multiple_lines = np.where(phone_service == 'No', 'No phone service', np.random.choice(['Yes', 'No'], n_samples))
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.45, 0.20])
    
    online_security = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]))
    online_backup = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]))
    device_protection = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]))
    tech_support = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]))
    streaming_tv = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], n_samples))
    streaming_movies = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], n_samples))
    
    # 5. Financials
    # Base charges depend on service
    monthly_charges = np.where(internet_service == 'Fiber optic', np.random.uniform(70, 115, n_samples),
                      np.where(internet_service == 'DSL', np.random.uniform(45, 80, n_samples),
                               np.random.uniform(18, 25, n_samples)))
    
    # Total charges (approximate) - add some nulls for realism 
    total_charges = monthly_charges * tenure
    
    # Add a little noise
    total_charges = total_charges + np.random.normal(0, 10, n_samples)
    total_charges = np.round(total_charges, 2)
    monthly_charges = np.round(monthly_charges, 2)
    
    # Convert total_charges to string type to simulate realistic dirty data
    total_charges_str = total_charges.astype(str)
    # Introduce 10 empty strings for 'TotalCharges' (new customers)
    empty_indices = np.random.choice(n_samples, 10, replace=False)
    for idx in empty_indices:
        total_charges_str[idx] = " "
        tenure[idx] = 0
        
    # 6. Target: Churn
    # Make churn probability depend somewhat systematically on features to give ML a pattern
    churn_prob = np.zeros(n_samples)
    
    # Increase prob for:
    churn_prob += np.where(contract == 'Month-to-month', 0.40, 0.0)
    churn_prob += np.where(internet_service == 'Fiber optic', 0.20, 0.0)
    churn_prob += np.where(paperless_billing == 'Yes', 0.10, 0.0)
    churn_prob -= np.where(tenure > 24, 0.20, 0.0)
    churn_prob -= np.where(tenure > 48, 0.20, 0.0)
    churn_prob -= np.where(online_security == 'Yes', 0.15, 0.0)
    churn_prob -= np.where(tech_support == 'Yes', 0.15, 0.0)
    
    # Normalize and clip probability between 0.05 and 0.8
    churn_prob = np.clip(churn_prob, 0.05, 0.8)
    
    churn = np.array(['Yes' if prob > np.random.random() else 'No' for prob in churn_prob])
    
    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges_str,
        'Churn': churn
    })
    
    return df

if __name__ == '__main__':
    df = generate_telecom_churn_data(5000)
    df.to_csv('churn_data.csv', index=False)
    print("churn_data.csv generated successfully with 5000 samples.")
