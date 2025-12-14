# Multiple-disease-prediction-


The Multiple Disease Prediction System is an end-to-end machine learning project developed to predict the likelihood of Chronic Kidney Disease, Liver Disease, and Parkinson’s Disease using patient medical data. The project demonstrates a complete data science workflow, from data preprocessing to model evaluation, following industry-standard practices.

Data Collection
Medical datasets for kidney disease, liver disease, and Parkinson’s disease were collected and loaded from CSV files. Each dataset contains clinical and biochemical attributes relevant to disease diagnosis.

Data Preprocessing
Inspected datasets for missing values and inconsistent entries
Replaced invalid symbols (such as ?) with null values
Converted necessary features to numeric format
Handled missing values using median imputation for numerical features and mode imputation for categorical features

Feature Engineering
Encoded categorical variables using Label Encoding
Selected medically significant features based on domain relevance
Applied feature scaling (MinMaxScaler) to normalize numerical values

Handling Class Imbalance
Addressed imbalanced target classes using Random Under-Sampling on training data to improve model fairness and performance

Model Development
Split the dataset into training and testing sets using stratified sampling
Trained machine learning models (Logistic Regression and others) for disease prediction
Tuned model parameters to ensure stable convergence and optimal performance

Model Evaluation
Evaluated models using accuracy, confusion matrix, and classification report
Compared predictions across diseases to ensure reliable performance

Tools & Technologies
Python
Pandas, NumPy
Scikit-learn
Imbalanced-learn

Outcome
This project provides a scalable and modular approach to disease prediction and showcases practical experience in data preprocessing, feature engineering, handling imbalanced datasets, and building reliable machine learning models.

And it is displayed in the streamlit app
