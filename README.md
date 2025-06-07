# Loan-Approval-Prediction-using-Machine-Learning
This project uses machine learning algorithms to predict whether a loan application should be approved based on various applicant features. The model is trained using a real-world dataset containing demographic and financial data such as income, credit history, education, and so on.

# Dataset
You can download the dataset here (add your dataset link).
It includes 13 features:
Loan_ID – Unique identifier
Gender – Male/Female
Married – Yes/No
Dependents – Number of dependents
Education – Graduate/Not Graduate
Self_Employed – Yes/No
ApplicantIncome – Income of applicant
CoapplicantIncome – Income of co-applicant
LoanAmount – In thousands
Loan_Amount_Term – Term in months
Credit_History – 1 (good), 0 (bad)
Property_Area – Urban/Semi-urban/Rural
Loan_Status – Y (Approved) / N (Not Approved)

# Libraries Used
Pandas – Data manipulation
NumPy – Numerical operations
Matplotlib & Seaborn – Visualization
Scikit-learn – ML models & preprocessing

# Data Preprocessing
Dropped the Loan_ID column (not relevant to prediction)
Used Label Encoding to convert categorical data to numeric
Visualized feature distributions using bar plots
Checked and filled missing values with column mean
Used a heatmap to observe feature correlations

# Visualization Samples
Bar Plots – To analyze value distributions
Catplots – To understand Loan_Status across categories
Heatmap – For correlation between features

# Model Building
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

# Algorithms Used:
Random Forest Classifier
K-Nearest Neighbors
Support Vector Machine
Logistic Regression

# Training Accuracy:
Random Forest: 98%
Logistic Regression: 80%
KNN: 78%
SVC: 68%

# Testing Accuracy:
Random Forest: 82.5%
Logistic Regression: 80.8%
SVC: 69.1%
KNN: 63.7%

# Best Performing Model
Random Forest Classifier performed best with an 82.5% test accuracy. For further improvements, ensemble techniques like Bagging and Boosting can be explored.

