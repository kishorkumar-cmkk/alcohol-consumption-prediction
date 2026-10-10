# 🍺 Alcohol Consumption Prediction System

A machine learning web application that predicts a country's **Total Litres of Pure Alcohol Consumption** using beverage consumption statistics and geographical information.

The project covers the complete machine learning workflow — from data exploration and preprocessing to model comparison, hyperparameter tuning, Flask web development, and cloud deployment.

## 🌐 Live Application

**Live Demo:**  
https://errordecoder.pythonanywhere.com/

---

## 📌 Project Overview

Alcohol consumption patterns vary significantly between countries and regions. This project uses machine learning to estimate the **total litres of pure alcohol consumed** based on:

- Country
- Beer Servings
- Spirit Servings
- Wine Servings
- Continent

The final trained model is integrated into an interactive Flask web application where users can enter these values and receive an instant prediction.

---

## 🎯 Project Objective

The main objectives of this project are to:

- Explore alcohol consumption patterns across countries and continents.
- Clean and preprocess the dataset for machine learning.
- Build and compare multiple regression algorithms.
- Perform hyperparameter tuning on ensemble models.
- Select the best-performing model using evaluation metrics.
- Build an interactive prediction interface using Flask.
- Deploy the complete application online.

---

## 📊 Dataset

The dataset contains alcohol consumption information for countries across different continents.

### Main Features

| Feature | Description |
|---|---|
| `country` | Name of the country |
| `beer_servings` | Number of beer servings |
| `spirit_servings` | Number of spirit servings |
| `wine_servings` | Number of wine servings |
| `continent` | Continent of the country |
| `total_litres_of_pure_alcohol` | Total litres of pure alcohol consumed |

### Target Variable

```text
total_litres_of_pure_alcohol
```

The original dataset contained **193 records**.

During preprocessing:

- An unnecessary index column was removed.
- Missing target values were removed.
- Missing numerical values were handled through preprocessing.
- Categorical variables were encoded for machine learning.

---

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand consumption patterns and relationships within the dataset.

The analysis included:

- Distribution of beer consumption
- Distribution of spirit consumption
- Distribution of wine consumption
- Distribution of total alcohol consumption
- Consumption comparison across continents
- Correlation analysis between numerical variables
- Country-level alcohol consumption comparison
- Top alcohol-consuming countries
- Outlier analysis using box plots

These insights were also used when designing the dashboard section of the web application.

---

## ⚙️ Data Preprocessing

A Scikit-learn preprocessing pipeline was created to ensure that the same transformations are applied during both training and prediction.

### Numerical Features

```text
beer_servings
spirit_servings
wine_servings
```

Missing numerical values are handled using:

```text
Median Imputation
```

### Categorical Features

```text
country
continent
```

Categorical features are transformed using:

```text
One-Hot Encoding
```

The encoder uses:

```python
handle_unknown="ignore"
```

which allows the prediction pipeline to safely process categorical inputs.

The preprocessing steps were combined using:

- `Pipeline`
- `ColumnTransformer`
- `SimpleImputer`
- `OneHotEncoder`

This approach also helps avoid data leakage by fitting preprocessing operations using the training data.

---

## 🤖 Machine Learning Models

Three regression algorithms were evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

### Initial Model Performance

| Model | R² Score |
|---|---:|
| Linear Regression | **0.9468** |
| Gradient Boosting | 0.9175 |
| Random Forest | 0.8707 |

Linear Regression produced the strongest baseline performance.

---

## 🔧 Hyperparameter Tuning

Hyperparameter tuning was performed using:

```text
GridSearchCV
```

with **5-fold cross-validation**.

The following models were tuned:

### Random Forest Regressor

Parameters explored included:

- Number of estimators
- Maximum tree depth
- Minimum samples required to split
- Minimum samples per leaf

### Gradient Boosting Regressor

Parameters explored included:

- Number of estimators
- Learning rate
- Maximum depth
- Minimum samples required to split

---

## 📈 Final Model Comparison

| Model | R² Score | MAE | MSE | RMSE |
|---|---:|---:|---:|---:|
| **Linear Regression** | **0.9468** | **0.5219** | **0.8936** | **0.9453** |
| Tuned Gradient Boosting | 0.9377 | 0.6242 | 1.0472 | 1.0233 |
| Tuned Random Forest | 0.8912 | 0.8398 | 1.8285 | 1.3522 |

### 🏆 Selected Model

**Linear Regression**

Despite tuning the ensemble models, Linear Regression achieved the:

- Highest R² score
- Lowest MAE
- Lowest MSE
- Lowest RMSE

Therefore, Linear Regression was selected as the final model.

### Final R² Score

```text
0.9468
```

This means the model explains approximately **94.68% of the variation in the evaluation data**.

---

## 🌐 Flask Web Application

The selected machine learning pipeline was saved using `joblib` and integrated into a Flask web application.

The application contains two main sections.

### 📊 Analytics Dashboard

The landing page displays:

- Dataset statistics
- Average beverage consumption
- Continent-level analysis
- Top alcohol-consuming countries
- Model performance
- Machine learning workflow

### 🔮 Prediction Interface

Users can provide:

- Country
- Beer servings
- Spirit servings
- Wine servings

The corresponding continent is automatically determined from the selected country before prediction.

The trained machine learning pipeline then returns the predicted:

```text
Total Litres of Pure Alcohol
```

Server-side validation is also used to ensure consistent country and continent information.

---

## 🛠️ Technologies Used

### Programming & Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Machine Learning Techniques

- Data Cleaning
- Exploratory Data Analysis
- Feature Preprocessing
- Median Imputation
- One-Hot Encoding
- Regression
- Cross-Validation
- Hyperparameter Tuning
- Model Evaluation

### Web Development

- Flask
- HTML
- CSS
- JavaScript
- Chart.js

### Deployment

- PythonAnywhere

---

## 📂 Project Structure

```text
alcohol_prediction/
│
├── app.py
├── model.py
├── best_model.pkl
├── beer-servings-cleaned.csv
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   ├── index.html
│   └── predict.html
│
└── static/
    └── css/
        └── style.css
```

---

## 💻 Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/kishorkumar-cmkk/alcohol-consumption-prediction.git
```

Move into the project directory:

```bash
cd alcohol_prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/macOS

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Flask application

```bash
python app.py
```

Then open the local Flask address displayed in the terminal.

---

## ☁️ Deployment

The Flask application is deployed using **PythonAnywhere**.

The deployment environment uses the same Scikit-learn version used to serialize the trained model to maintain model compatibility.

**Live Application:**  
https://errordecoder.pythonanywhere.com/

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Train-Test Split
   ↓
Preprocessing Pipeline
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Hyperparameter Tuning
   ↓
Final Model Selection
   ↓
Model Serialization
   ↓
Flask Integration
   ↓
Local Testing
   ↓
Cloud Deployment
```

---

## 🚀 Future Improvements

Possible future improvements include:

- Expanding the dataset with additional countries and observations
- Adding demographic and economic variables
- Comparing additional regression algorithms
- Adding prediction explanations or feature-level insights
- Improving dashboard interactivity
- Adding automated model monitoring and retraining workflows

---

## 👨‍💻 Author

**Kishor Kumar M**

B.Sc. Artificial Intelligence and Machine Learning  
Data Analyst | Data Science & AI Enthusiast

**GitHub:** https://github.com/kishorkumar-cmkk

**LinkedIn:** https://www.linkedin.com/in/kishorkumarm24

---

## ⭐ Project Status

**Completed & Deployed** ✅

The machine learning model, Flask interface, analytics dashboard, and cloud deployment are fully functional.
