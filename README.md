# Regression Model Example

This project demonstrates a complete regression model implementation for **House Price Prediction**.

## Features

- **Data Generation**: Synthetic house price dataset with realistic features
- **Multiple Linear Regression**: Predicts house prices based on multiple features
- **Model Evaluation**: Comprehensive metrics (MSE, RMSE, MAE, R² Score)
- **Visualization**: Scatter plots and residual analysis
- **Predictions**: Make predictions on new data

## Dataset Features

The model uses the following features to predict house prices:
- **Size_sqft**: House size in square feet (800-5000)
- **Bedrooms**: Number of bedrooms (1-6)
- **Bathrooms**: Number of bathrooms (1-4)
- **Age_years**: Age of the house in years (0-50)
- **Location_Score**: Location quality score (1-10, higher is better)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the regression model:
```bash
python regression_model.py
```

## Model Output

The script will:
1. Generate synthetic house price data
2. Split data into training and test sets
3. Train a linear regression model
4. Display model coefficients
5. Evaluate model performance on both training and test sets
6. Make predictions on new houses
7. Generate visualization plots

## Evaluation Metrics

- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as target
- **MAE (Mean Absolute Error)**: Average absolute difference
- **R² Score**: Coefficient of determination (1.0 = perfect prediction)

## Customization

You can easily modify the model for different scenarios:
- Change the data generation function for different features
- Use different regression algorithms (Ridge, Lasso, Polynomial, etc.)
- Load real data from CSV files instead of generating synthetic data
- Add feature engineering and preprocessing steps

## Example Scenarios

This regression model can be adapted for:
- Sales prediction
- Stock price forecasting
- Temperature prediction
- Customer lifetime value
- Energy consumption prediction
- And many more!

