"""
Regression Model Example: House Price Prediction
This script demonstrates a complete regression model workflow including:
- Data generation/simulation
- Data preprocessing
- Model training (Multiple Linear Regression)
- Model evaluation
- Predictions and visualization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

class HousePriceRegression:
    """House Price Prediction using Linear Regression"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        
    def generate_data(self, n_samples=1000):
        """
        Generate synthetic house price data
        Features:
        - Size (sqft): 800-5000
        - Bedrooms: 1-6
        - Bathrooms: 1-4
        - Age: 0-50 years
        - Location_Score: 1-10 (higher = better location)
        """
        np.random.seed(42)
        
        size = np.random.uniform(800, 5000, n_samples)
        bedrooms = np.random.randint(1, 7, n_samples)
        bathrooms = np.random.uniform(1, 4, n_samples)
        age = np.random.uniform(0, 50, n_samples)
        location_score = np.random.uniform(1, 10, n_samples)
        
        # Generate price based on features with some noise
        # Base price formula: price = 50*size + 20000*bedrooms + 15000*bathrooms - 1000*age + 5000*location_score + noise
        price = (50 * size + 
                20000 * bedrooms + 
                15000 * bathrooms - 
                1000 * age + 
                5000 * location_score + 
                np.random.normal(0, 20000, n_samples))
        
        # Ensure prices are positive
        price = np.maximum(price, 50000)
        
        data = pd.DataFrame({
            'Size_sqft': size,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Age_years': age,
            'Location_Score': location_score,
            'Price': price
        })
        
        return data
    
    def prepare_data(self, data):
        """Prepare data for training"""
        X = data[['Size_sqft', 'Bedrooms', 'Bathrooms', 'Age_years', 'Location_Score']]
        y = data['Price']
        return X, y
    
    def train(self, X, y):
        """Train the regression model"""
        self.model.fit(X, y)
        self.is_trained = True
        print("Model trained successfully!")
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def evaluate(self, X, y_true):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2
        }
        
        return metrics, y_pred
    
    def get_coefficients(self, feature_names):
        """Get model coefficients"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        coefficients = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': self.model.coef_,
            'Intercept': [self.model.intercept_] * len(feature_names)
        })
        return coefficients


def plot_results(y_true, y_pred, title="Predicted vs Actual"):
    """Visualize regression results"""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Scatter plot of predicted vs actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title(f'{title} - Scatter Plot')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals plot
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Price ($)')
    plt.ylabel('Residuals ($)')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'regression_results.png'")
    plt.show()


def main():
    """Main execution function"""
    print("=" * 60)
    print("House Price Prediction - Regression Model")
    print("=" * 60)
    
    # Initialize model
    model = HousePriceRegression()
    
    # Generate data
    print("\n1. Generating synthetic house price data...")
    data = model.generate_data(n_samples=1000)
    print(f"   Generated {len(data)} samples")
    print("\n   First few rows:")
    print(data.head())
    
    # Prepare data
    print("\n2. Preparing data for training...")
    X, y = model.prepare_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train model
    print("\n3. Training regression model...")
    model.train(X_train, y_train)
    
    # Display coefficients
    print("\n4. Model Coefficients:")
    coefficients = model.get_coefficients(X.columns)
    print(coefficients.to_string(index=False))
    
    # Evaluate on training set
    print("\n5. Evaluating on training set...")
    train_metrics, y_train_pred = model.evaluate(X_train, y_train)
    print("   Training Metrics:")
    for metric, value in train_metrics.items():
        print(f"   {metric}: {value:.2f}")
    
    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    test_metrics, y_test_pred = model.evaluate(X_test, y_test)
    print("   Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"   {metric}: {value:.2f}")
    
    # Make predictions on new data
    print("\n7. Making predictions on new houses...")
    new_houses = pd.DataFrame({
        'Size_sqft': [2000, 3500, 1200],
        'Bedrooms': [3, 4, 2],
        'Bathrooms': [2, 3, 1.5],
        'Age_years': [5, 15, 25],
        'Location_Score': [8, 6, 7]
    })
    
    predictions = model.predict(new_houses)
    new_houses['Predicted_Price'] = predictions
    
    print("\n   New House Predictions:")
    print(new_houses.to_string(index=False))
    
    # Visualize results
    print("\n8. Creating visualizations...")
    plot_results(y_test, y_test_pred, "Test Set Predictions")
    
    print("\n" + "=" * 60)
    print("Regression Model Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

