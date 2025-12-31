# src/generate_data.py
import pandas as pd
import numpy as np
import os

def generate_salary_data():
    np.random.seed(42) 
    
    # 1. Generate random experience
    experience = np.random.rand(150, 1) * 20 
    
    # 2. Generate Salary
    noise = np.random.randn(150, 1) * 5000
    salary = 30000 + (experience * 8000) + noise
    
    # --- NEW: CLEANING THE DATA ---
    # Round experience to 1 decimal place (e.g., 7.5)
    experience = np.round(experience, 1)
    
    # Round salary to 0 decimals and make it an integer (e.g., 88965)
    salary = np.round(salary).astype(int)
    # ------------------------------

    # 3. Create DataFrame
    data = pd.DataFrame({
        'YearsExperience': experience.flatten(),
        'Salary': salary.flatten()
    })
    
    # 4. Save to CSV
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/salary_data.csv', index=False)
    print("✅ Clean dataset created successfully at data/salary_data.csv")
    
    # Print first 5 rows to show the user the clean data
    print("\nSample Data:")
    print(data.head())

if __name__ == "__main__":
    generate_salary_data()