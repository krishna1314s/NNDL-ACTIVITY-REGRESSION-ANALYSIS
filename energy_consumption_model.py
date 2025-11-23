"""
Scenario 4: Energy Consumption Prediction
Predict energy consumption based on weather and building characteristics
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

class EnergyConsumptionModel:
    """Energy Consumption Prediction using Linear Regression"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        
    def generate_data(self, n_samples=1000):
        """
        Generate synthetic energy consumption data
        
        Input Features:
        - Temperature: -10°C to 35°C
        - Building_Size: 500-5000 sqft
        - Occupancy: 1-20 people
        - Insulation_Rating: 1-10 (higher = better insulation)
        """
        np.random.seed(42)
        
        temperature = np.random.uniform(-10, 35, n_samples)  # Celsius
        building_size = np.random.uniform(500, 5000, n_samples)  # sqft
        occupancy = np.random.uniform(1, 20, n_samples)  # number of people
        insulation_rating = np.random.uniform(1, 10, n_samples)  # 1-10 scale
        
        # Energy consumption increases with temperature deviation from optimal (20°C)
        # Energy = 50×(Size/1000) + 100×Occupancy + |Temp-20|×50 - 20×Insulation + noise
        temp_effect = np.abs(temperature - 20) * 2
        energy = (50 * building_size / 1000 + 
                  100 * occupancy + 
                  temp_effect * 50 - 
                  20 * insulation_rating + 
                  np.random.normal(0, 50, n_samples))
        
        # Ensure energy is non-negative
        energy = np.maximum(energy, 0)
        
        data = pd.DataFrame({
            'Temperature_C': temperature,
            'Building_Size_sqft': building_size,
            'Occupancy': occupancy,
            'Insulation_Rating': insulation_rating,
            'Energy_Consumption_kWh': energy
        })
        
        return data
    
    def prepare_data(self, data):
        """Prepare data for training"""
        X = data[['Temperature_C', 'Building_Size_sqft', 'Occupancy', 'Insulation_Rating']]
        y = data['Energy_Consumption_kWh']
        return X, y
    
    def train(self, X, y):
        """Train the regression model"""
        self.model.fit(X, y)
        self.is_trained = True
        print("[OK] Model trained successfully!")
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
            'Impact': ['Positive' if c > 0 else 'Negative' for c in self.model.coef_]
        })
        return coefficients


def plot_results(y_true, y_pred, title="Energy Consumption Predictions"):
    """Visualize regression results"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Scatter plot of predicted vs actual
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Energy Consumption (kWh)', fontsize=11)
    plt.ylabel('Predicted Energy Consumption (kWh)', fontsize=11)
    plt.title(f'{title} - Scatter Plot', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals plot
    plt.subplot(2, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Energy Consumption (kWh)', fontsize=11)
    plt.ylabel('Residuals (kWh)', fontsize=11)
    plt.title('Residuals Plot', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of actual vs predicted
    plt.subplot(2, 2, 3)
    plt.hist(y_true, bins=30, alpha=0.5, label='Actual', color='blue')
    plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted', color='orange')
    plt.xlabel('Energy Consumption (kWh)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Distribution Comparison', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=30, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('Prediction Error (kWh)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Error Distribution', fontsize=12, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', lw=2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('energy_consumption_results.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Visualization saved as 'energy_consumption_results.png'")
    plt.show()


def plot_feature_importance(model, feature_names):
    """Plot feature importance based on coefficients"""
    coefficients = model.model.coef_
    feature_importance = np.abs(coefficients)
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if c > 0 else 'red' for c in coefficients]
    bars = plt.barh(feature_names, feature_importance, color=colors, alpha=0.7, edgecolor='black')
    
    plt.xlabel('Absolute Coefficient Value', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importance (Absolute Coefficients)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, coef) in enumerate(zip(bars, coefficients)):
        plt.text(feature_importance[i] + max(feature_importance) * 0.01, i, 
                f'{coef:.2f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('energy_feature_importance.png', dpi=300, bbox_inches='tight')
    print("[OK] Feature importance plot saved as 'energy_feature_importance.png'")
    plt.show()


def main():
    """Main execution function"""
    print("=" * 80)
    print("SCENARIO 4: ENERGY CONSUMPTION PREDICTION")
    print("=" * 80)
    print("\nProblem: Predict energy consumption based on weather and building features")
    print("\nInput Features:")
    print("  - Temperature: -10C to 35C")
    print("  - Building_Size: 500-5000 sqft")
    print("  - Occupancy: 1-20 people")
    print("  - Insulation_Rating: 1-10 (higher = better insulation)")
    print("\nEnergy Formula:")
    print("  Energy = 50*(Size/1000) + 100*Occupancy + |Temp-20|*50 - 20*Insulation + noise")
    print("=" * 80)
    
    # Initialize model
    model = EnergyConsumptionModel()
    
    # Generate data
    print("\n[STEP 1] Generating synthetic energy consumption data...")
    data = model.generate_data(n_samples=1000)
    print(f"   [OK] Generated {len(data)} samples")
    print("\n   First 10 rows of data:")
    print(data.head(10).to_string(index=False))
    
    # Show data statistics
    print("\n   Data Statistics:")
    print(data.describe().round(2).to_string())
    
    # Prepare data
    print("\n[STEP 2] Preparing data for training...")
    X, y = model.prepare_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   [OK] Training samples: {len(X_train)}")
    print(f"   [OK] Test samples: {len(X_test)}")
    
    # Train model
    print("\n[STEP 3] Training regression model...")
    model.train(X_train, y_train)
    
    # Display coefficients
    print("\n[STEP 4] Model Coefficients (Feature Weights):")
    print("-" * 80)
    coefficients = model.get_coefficients(X.columns)
    print(coefficients.to_string(index=False))
    print(f"\n   Intercept: {model.model.intercept_:.2f} kWh")
    print("\n   Interpretation:")
    print("   - Positive coefficient = increases energy consumption")
    print("   - Negative coefficient = decreases energy consumption")
    
    # Evaluate on training set
    print("\n[STEP 5] Evaluating on training set...")
    train_metrics, y_train_pred = model.evaluate(X_train, y_train)
    print("   Training Metrics:")
    print(f"   - Mean Squared Error (MSE): {train_metrics['MSE']:.2f}")
    print(f"   - Root Mean Squared Error (RMSE): {train_metrics['RMSE']:.2f} kWh")
    print(f"   - Mean Absolute Error (MAE): {train_metrics['MAE']:.2f} kWh")
    print(f"   - R² Score: {train_metrics['R2_Score']:.4f} ({train_metrics['R2_Score']*100:.2f}% variance explained)")
    
    # Evaluate on test set
    print("\n[STEP 6] Evaluating on test set...")
    test_metrics, y_test_pred = model.evaluate(X_test, y_test)
    print("   Test Metrics:")
    print(f"   - Mean Squared Error (MSE): {test_metrics['MSE']:.2f}")
    print(f"   - Root Mean Squared Error (RMSE): {test_metrics['RMSE']:.2f} kWh")
    print(f"   - Mean Absolute Error (MAE): {test_metrics['MAE']:.2f} kWh")
    print(f"   - R² Score: {test_metrics['R2_Score']:.4f} ({test_metrics['R2_Score']*100:.2f}% variance explained)")
    
    # Make predictions on new buildings
    print("\n[STEP 7] Making predictions on new buildings...")
    print("-" * 80)
    new_buildings = pd.DataFrame({
        'Temperature_C': [25, 5, 30, -5, 20],
        'Building_Size_sqft': [2000, 3500, 1500, 4500, 2500],
        'Occupancy': [5, 10, 3, 15, 8],
        'Insulation_Rating': [7, 5, 9, 3, 8]
    })
    
    predictions = model.predict(new_buildings)
    new_buildings['Predicted_Energy_kWh'] = predictions
    
    print("\n   New Building Predictions:")
    print(new_buildings.to_string(index=False))
    
    # Show some example predictions with interpretation
    print("\n   Example Interpretations:")
    for idx, row in new_buildings.iterrows():
        temp = row['Temperature_C']
        size = row['Building_Size_sqft']
        occ = row['Occupancy']
        insul = row['Insulation_Rating']
        pred = row['Predicted_Energy_kWh']
        
        temp_dev = abs(temp - 20)
        print(f"\n   Building {idx+1}:")
        print(f"     - Temp: {temp}C (deviation from 20C: {temp_dev:.1f}C)")
        print(f"     - Size: {size:.0f} sqft, Occupancy: {occ:.0f} people")
        print(f"     - Insulation: {insul:.1f}/10")
        print(f"     - Predicted Energy: {pred:.2f} kWh")
    
    # Visualize results
    print("\n[STEP 8] Creating visualizations...")
    plot_results(y_test, y_test_pred, "Energy Consumption Predictions")
    plot_feature_importance(model, X.columns)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"[OK] Model successfully trained on {len(X_train)} samples")
    print(f"[OK] Test R² Score: {test_metrics['R2_Score']:.4f} - Model explains {test_metrics['R2_Score']*100:.2f}% of variance")
    print(f"[OK] Average prediction error: {test_metrics['RMSE']:.2f} kWh")
    print(f"[OK] Mean absolute error: {test_metrics['MAE']:.2f} kWh")
    print("\nKey Insights:")
    print("  - Energy consumption increases with building size and occupancy")
    print("  - Temperature deviation from 20C increases energy use (heating/cooling)")
    print("  - Better insulation reduces energy consumption")
    print("=" * 80)
    print("\n[OK] Energy Consumption Model Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

