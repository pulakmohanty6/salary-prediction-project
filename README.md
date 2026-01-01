# Salary Prediction Project

## 📌 Project Overview
This is a Machine Learning project that predicts an employee's salary based on their years of experience. I built this to understand the basics of **Linear Regression** and how data can be used to make predictions.

## 📂 Project Structure
* `data/` → Contains the dataset (`salary_data.csv`).
* `models/` → Contains the saved model file (`salary_model.pkl`) for future use.
* `src/` → Contains the source code:
  * `generate_data.py`: Script to generate the synthetic dataset.
  * `main.py`: The main script to train the model, evaluate it, and save the .pkl file.
  * `predict.py`: A separate script to load the saved model and predict salary for any new input value.
* `salary_graph.png` → A generated graph visualizing the linear regression line.           
* `README.md` → This file.

## 🛠️ Tools Used
* **Python** (Programming Language)
* **Pandas** (For loading data)
* **Matplotlib** (For drawing graphs)
* **Scikit-Learn** (For the Machine Learning model)
* **VS Code** (Code Editor)

## 📊 Results
* **Algorithm Used:** Linear Regression.
* **Accuracy:** The model achieved approximately **98% accuracy** on the test data.
* **Key Insight:** There is a clear linear relationship between years of experience and salary.
