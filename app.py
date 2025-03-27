from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('data/real_estate.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/slr', methods=['GET', 'POST'])
def simple_linear_regression():
    X = df[['house_size']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)

    prediction = None
    if request.method == 'POST':
        house_size = float(request.form['house_size'])
        prediction = model.predict([[house_size]])[0]
    
    return render_template('result.html', 
                         algorithm='Simple Linear Regression',
                         accuracy=accuracy,
                         prediction=prediction,
                         input_fields={'house_size': 'House Size (sq ft)'})

@app.route('/mlr', methods=['GET', 'POST'])
def multiple_linear_regression():
    X = df[['house_size', 'bedrooms', 'bathrooms', 'age', 'location_score', 'garage_size']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)

    prediction = None
    if request.method == 'POST':
        input_data = {
            'house_size': float(request.form['house_size']),
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'age': int(request.form['age']),
            'location_score': int(request.form['location_score']),
            'garage_size': int(request.form['garage_size'])
        }
        prediction = model.predict([list(input_data.values())])[0]
    
    return render_template('result.html', 
                         algorithm='Multiple Linear Regression',
                         accuracy=accuracy,
                         prediction=prediction,
                         input_fields={
                             'house_size': 'House Size (sq ft)',
                             'bedrooms': 'Number of Bedrooms',
                             'bathrooms': 'Number of Bathrooms',
                             'age': 'Age of House (years)',
                             'location_score': 'Location Score (1-10)',
                             'garage_size': 'Garage Size (cars)'
                         })

@app.route('/polynomial', methods=['GET', 'POST'])
def polynomial_regression():
    X = df[['house_size']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    poly_features = PolynomialFeatures(degree=3)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_pred = model.predict(X_test_poly)
    accuracy = r2_score(y_test, y_pred)

    prediction = None
    if request.method == 'POST':
        house_size = float(request.form['house_size'])
        house_size_poly = poly_features.transform([[house_size]])
        prediction = model.predict(house_size_poly)[0]
    
    return render_template('result.html', 
                         algorithm='Polynomial Regression',
                         accuracy=accuracy,
                         prediction=prediction,
                         input_fields={'house_size': 'House Size (sq ft)'})

@app.route('/logistic', methods=['GET', 'POST'])
def logistic_regression():
    df['price_category'] = (df['price'] > df['price'].mean()).astype(int)
    
    X = df[['house_size', 'bedrooms', 'bathrooms', 'age', 'location_score', 'garage_size']]
    y = df['price_category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    prediction = None
    if request.method == 'POST':
        input_data = {
            'house_size': float(request.form['house_size']),
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'age': int(request.form['age']),
            'location_score': int(request.form['location_score']),
            'garage_size': int(request.form['garage_size'])
        }
        prediction = model.predict([list(input_data.values())])[0]
        prediction = "High Price" if prediction == 1 else "Low Price"
    
    return render_template('result.html', 
                         algorithm='Logistic Regression',
                         accuracy=accuracy,
                         prediction=prediction,
                         input_fields={
                             'house_size': 'House Size (sq ft)',
                             'bedrooms': 'Number of Bedrooms',
                             'bathrooms': 'Number of Bathrooms',
                             'age': 'Age of House (years)',
                             'location_score': 'Location Score (1-10)',
                             'garage_size': 'Garage Size (cars)'
                         })

@app.route('/knn', methods=['GET', 'POST'])
def knn():
    df['price_category'] = (df['price'] > df['price'].mean()).astype(int)
    
    X = df[['house_size', 'bedrooms', 'bathrooms', 'age', 'location_score', 'garage_size']]
    y = df['price_category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    prediction = None
    if request.method == 'POST':
        input_data = {
            'house_size': float(request.form['house_size']),
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'age': int(request.form['age']),
            'location_score': int(request.form['location_score']),
            'garage_size': int(request.form['garage_size'])
        }
        prediction = model.predict([list(input_data.values())])[0]
        prediction = "High Price" if prediction == 1 else "Low Price"
    
    return render_template('result.html', 
                         algorithm='K-Nearest Neighbors',
                         accuracy=accuracy,
                         prediction=prediction,
                         input_fields={
                             'house_size': 'House Size (sq ft)',
                             'bedrooms': 'Number of Bedrooms',
                             'bathrooms': 'Number of Bathrooms',
                             'age': 'Age of House (years)',
                             'location_score': 'Location Score (1-10)',
                             'garage_size': 'Garage Size (cars)'
                         })

if __name__ == '__main__':
    app.run(debug=True) 