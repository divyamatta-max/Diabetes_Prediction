# Diabetes_Prediction
🩺 A Machine Learning project that uses Support Vector Machine (SVM) and feature standardization to predict whether a person is diabetic based on medical diagnostic features.
# 🩺 Diabetes Prediction Using Machine Learning

## 📌 Project Overview

This project is a **Machine Learning-based Diabetes Prediction system** developed using Python.

The project uses medical diagnostic data to train a **Support Vector Machine (SVM)** classifier and predict whether a person is diabetic or not.

The dataset is analyzed, standardized using `StandardScaler`, divided into training and testing data, and then used to train a Linear SVM model.

## 🎯 Objective

The main objective of this project is to build a machine learning model that can:

* Analyze diabetes-related diagnostic data
* Preprocess and standardize the input features
* Train a machine learning classification model
* Evaluate the model using accuracy
* Predict whether a person is diabetic or not based on input values

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Support Vector Machine (SVM)
* StandardScaler
* Jupyter Notebook

## 🤖 Machine Learning Algorithm

### Support Vector Machine (SVM)

The project uses a **Support Vector Machine classifier with a linear kernel** for diabetes classification.

```python
classifier = svm.SVC(kernel='linear')
```

The SVM model is trained using the standardized training data.

## 🔄 Project Workflow

```text
Diabetes Dataset
       ↓
Load Dataset
       ↓
Explore Dataset
       ↓
Separate Features and Outcome
       ↓
Standardize Features
       ↓
Train-Test Split
       ↓
Linear SVM Classifier
       ↓
Model Training
       ↓
Accuracy Evaluation
       ↓
Diabetes Prediction
```

## 📂 Dataset

The project uses a `diabetes.csv` dataset.

The target column used for prediction is:

* `Outcome` – represents the diabetes prediction result

The input features are the remaining columns in the dataset.

## ⚙️ Implementation Steps

### 1. Import Required Libraries

The project imports NumPy, Pandas, and Scikit-learn libraries required for preprocessing, model training, data splitting, and evaluation.

### 2. Load the Dataset

The diabetes dataset is loaded using Pandas.

```python
diabetes_dataset = pd.read_csv('/diabetes.csv')
```

### 3. Explore the Dataset

The project examines the dataset using:

* `head()`
* `shape`
* `describe()`
* `value_counts()`
* `groupby()`

These operations help understand the dataset and the distribution of the target variable.

### 4. Separate Features and Target

The input features are stored in `X`, while the `Outcome` column is stored in `Y`.

```python
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
```

### 5. Standardize the Data

`StandardScaler` is used to standardize the feature values.

```python
scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)
```

The standardized data is then used for model training and prediction.

### 6. Split the Dataset

The dataset is divided into training and testing sets using an **80/20 split**.

The split is stratified according to the target variable.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)
```

### 7. Create the SVM Classifier

A Support Vector Machine classifier with a linear kernel is created.

```python
classifier = svm.SVC(kernel='linear')
```

### 8. Train the Model

The classifier is trained using the training dataset.

```python
classifier.fit(X_train, Y_train)
```

### 9. Evaluate the Model

The model is evaluated on both the training data and test data using `accuracy_score`.

```python
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(
    X_train_prediction, Y_train
)
```

The test data is also used to calculate the test accuracy.

```python
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(
    X_test_prediction, Y_test
)
```

### 10. Make a New Prediction

The project also demonstrates prediction using a new set of input values.

The input is converted into a NumPy array, reshaped, standardized using the same scaler, and passed to the trained classifier.

```python
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict(std_data)
```

The result is then displayed as:

```text
The person is not diabetic
```

or

```text
The person is diabetic
```

## 📊 Model Evaluation

The project evaluates the SVM model using **accuracy score**.

Both training accuracy and test accuracy are calculated in the notebook.

The exact accuracy may depend on the dataset and the execution of the notebook.

## 📁 Project Structure

```text
Diabetes-Prediction/
│
├── Diabetes_prediction_project_python.ipynb
├── diabetes.csv
└── README.md
```

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-link>
```

### 2. Install Required Libraries

```bash
pip install numpy pandas scikit-learn jupyter
```

### 3. Open the Notebook

Open the `.ipynb` file using:

* Jupyter Notebook
* JupyterLab
* Visual Studio Code

### 4. Add the Dataset

Make sure `diabetes.csv` is available at the location expected by the notebook.

### 5. Run the Notebook

Run the cells in sequence to:

* Load the dataset
* Explore the data
* Standardize the features
* Train the SVM model
* Evaluate the model
* Make a diabetes prediction

## 📚 What I Learned

Through this project, I learned about:

* Loading and exploring datasets using Pandas
* Separating features and target variables
* Feature standardization
* Train-test splitting
* Stratified data splitting
* Support Vector Machine classification
* Model training and prediction
* Accuracy evaluation
* Making predictions using new input data

## 🔮 Future Improvements

This project can be further improved by:

* Comparing SVM with other classification algorithms
* Adding additional evaluation metrics such as precision, recall, and F1-score
* Creating visualizations for better understanding of the dataset
* Improving the user interface for entering prediction values
* Building a simple web application for diabetes prediction

## ⚠️ Disclaimer

This project is created for **educational and learning purposes**. The predictions from this machine learning model should not be considered a medical diagnosis. Professional medical advice should always be obtained from a qualified healthcare professional.

## 👩‍💻 Author

**BTech Artificial Intelligence & Machine Learning Student**

This project was developed as part of my learning journey in **Machine Learning and Artificial Intelligence**.
