import pandas as pd
import matplotlib.pyplot as plt
import joblib  
import os      
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main():
    # 1. Loading dataset
    
    print("📂 Loading dataset...")
    
    if not os.path.exists('data/salary_data.csv'):
        print("❌ Error: 'data/salary_data.csv' not found.")
        return

    df = pd.read_csv('data/salary_data.csv')
    print("✅ Dataset loaded successfully.")

    # 2. Splitting dataset

    train_dataset = df.sample(frac=0.8, random_state=42).copy()
    test_dataset = df.drop(train_dataset.index).copy()
    print("✅ Dataset split into training and testing sets.")

    # 3. Training model
    
    reg = LinearRegression()
    predictors = ['YearsExperience']
    target = 'Salary'
    print("🧠 Training model...")

    reg.fit(train_dataset[predictors], train_dataset[target])
    print("✅ Model trained successfully.")

    # 4. Evaluating model

    predictions = reg.predict(test_dataset[predictors])
    
    r2 = r2_score(test_dataset[target], predictions)

    print(f"\n📊 Model Performance:")
    print(f"   R² Score: {r2:.4f} (Accuracy)")

    # 5. Visualizing results

    print("\n📈 Generating Graph...")
    plt.figure(figsize=(10, 6))
 
    plt.scatter(df['YearsExperience'], df['Salary'], color='blue', label='Actual Data')
    
    plt.plot(df['YearsExperience'], reg.predict(df[['YearsExperience']]), color='red', linewidth=2, label='Regression Line')
    
    plt.title('Salary vs Years of Experience')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()
    
    # 5_5. Saving graph
    plt.savefig('salary_graph.png')
    print("✅ Graph saved as 'salary_graph.png'")

    # 6. Saving model

    os.makedirs('models', exist_ok=True)
    joblib.dump(reg, 'models/salary_model.pkl')
    print("✅ Model saved to 'models/salary_model.pkl'")

if __name__ == "__main__":
    main()