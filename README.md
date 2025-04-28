

# 📈 Bank Marketing Campaign Analysis

## 📚 Project Overview
This project analyzes data from a **bank's marketing campaign** to uncover the factors influencing customers' decisions to subscribe to a term deposit.  
We perform exploratory data analysis (EDA) and apply various machine learning models to predict the subscription outcome, helping the bank improve its targeting strategies.

---

## 📂 Dataset
The dataset includes:
- Client demographic information (age, job, marital status, education, etc.)
- Contact details (communication type, last contact date)
- Campaign outcome from previous contacts
- Target variable (`y`): Whether the client subscribed (`yes` or `no`)

> **Dataset Source**: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

---

## ⚙️ Requirements

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn plotly
```

**Libraries Used:**
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- plotly

---

## 🚀 How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/bank-marketing-analysis.git
   cd bank-marketing-analysis
   ```
2. Ensure the dataset (`bank.csv`) is placed in the correct directory.
3. Open the notebook:
   ```bash
   jupyter notebook bank\ marketing.ipynb
   ```
4. Run the cells sequentially.

> 🔥 Tip: Update the path to your dataset if necessary:
> ```python
> bank = pd.read_csv("path_to_your_dataset/bank.csv")
> ```

---

## 🛠️ Project Workflow

- Data Loading & Cleaning
- Exploratory Data Analysis (EDA) with visualizations
- Feature Encoding and Scaling
- Building and Training Models:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes Classifier
  - Gradient Boosting Classifier
- Model Evaluation:
  - Accuracy, Precision, Recall, F1 Score
  - Confusion Matrix
  - ROC-AUC Curves
- Hyperparameter Tuning using GridSearchCV
- Model Comparison and Final Selection

---

## ✅ Results

- Multiple models were compared.
- Detailed evaluation metrics such as accuracy, precision, recall, and ROC-AUC were calculated.
- Best-performing models were selected based on metrics and visual inspection.

---

## 🎯 Future Improvements

- Apply advanced models like XGBoost, LightGBM, or CatBoost.
- Perform deeper feature engineering.
- Model deployment via Flask/Streamlit and cloud hosting (AWS, Heroku).

---

## 📬 Contact

Feel free to reach out if you have any questions or suggestions!  
**Email:** shuaibmusthafam@gmail.com  
**LinkedIn:** [shuaib-musthafa](https://www.linkedin.com/in/shuaib-musthafa-771a07259/)

---

## 📌 Acknowledgments
- UCI Machine Learning Repository
- scikit-learn and Python Data Science Community

---

> ⭐ If you like this project, feel free to star ⭐ the repository!

---
