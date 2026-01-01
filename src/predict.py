import joblib
import pandas as pd
import os

def predict_salary():
    # 1. Loading the saved model

    if not os.path.exists('models/salary_model.pkl'):
        print("❌ Error: Model not found. Please run main.py first to train it!")
        return

    print("Loading saved model...")
    model = joblib.load('models/salary_model.pkl')
    print("Model loaded!")

    # 2. Asking User for Input

    user_input = input("\nEnter Years of Experience: ")
    
    try:
        years = float(user_input)
    except ValueError:
        print("❌ That doesn't look like a number. Please try again.")
        return

    # 3. Preparing Data
 
    new_data = pd.DataFrame({'YearsExperience': [years]})

    # 4. Predicting

    prediction = model.predict(new_data)
    
    # 5. Showing Result

    print(f"Predicted Salary: ${prediction[0]:,.0f}")

if __name__ == "__main__":
    predict_salary()