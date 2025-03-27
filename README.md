# Machine Learning Algorithms Web Interface

This project provides a web interface to explore and run different machine learning algorithms on a real estate dataset. The application includes implementations of:

1. Simple Linear Regression
2. Multiple Linear Regression
3. Polynomial Regression
4. Logistic Regression
5. K-Nearest Neighbors (KNN)

## Setup Instructions

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the Flask application:
```bash
python app.py
```

4. Open your web browser and navigate to:
```
http://localhost:5000
```

## Project Structure

- `app.py`: Main Flask application with routes and ML implementations
- `data/real_estate.csv`: Sample real estate dataset
- `templates/`: HTML templates for the web interface
  - `index.html`: Main page with algorithm selection
  - `result.html`: Results display page

## Features

- User-friendly web interface
- Real-time algorithm execution
- Visualization of results where applicable
- Accuracy metrics for each algorithm
- No authentication required
- Responsive design using Bootstrap

## Dataset

The application uses a sample real estate dataset with the following features:
- house_size: Size of the house in square feet
- bedrooms: Number of bedrooms
- bathrooms: Number of bathrooms
- age: Age of the house in years
- price: Price of the house in dollars

## Note

The sample dataset is small for demonstration purposes. For better accuracy, you should replace it with a larger, more comprehensive real estate dataset. 