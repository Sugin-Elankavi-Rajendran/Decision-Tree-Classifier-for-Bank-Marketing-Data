# Decision Tree Classifier for Bank Marketing Data

This repository contains a Python script implementing a Decision Tree Classifier on the Bank Marketing dataset. The dataset is loaded using Pandas, and the Decision Tree Classifier is trained and evaluated using the scikit-learn library.

## Prerequisites

Make sure you have the required libraries installed. You can install them using the following command:

```bash
pip install pandas scikit-learn
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
```

2. Run the script:
```bash
python decision_tree_classifier.py
```

## Description

The script performs the following steps:

1. Data Loading: The Bank Marketing dataset is loaded from a CSV file using Pandas.
```bash
url = r"D:\OneDrive\Prodigy InfoTech\DS_03\bank+marketing\bank\bank-full.csv"
data = pd.read_csv(url, sep=';')
```

2. Data Preprocessing: The 'duration' column is dropped, and categorical variables are one-hot encoded.
```bash
data = data.drop('duration', axis=1)
data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y'], drop_first=True)
```

3. Splitting Data: The data is split into training and testing sets.
```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

4. Model Training: A Decision Tree Classifier is instantiated and trained on the training data.
```bash
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
```

5. Model Evaluation: The model is evaluated on the test set, and accuracy, confusion matrix, and classification report are printed.
```bash
y_pred = dt_classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```


