"""
Alternative Regression Scenarios
This file demonstrates how to adapt the regression model for different use cases
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ============================================================================
# SCENARIO 1: Sales Prediction
# ============================================================================

def sales_prediction_model():
    """Predict monthly sales based on advertising spend and other factors"""
    np.random.seed(42)
    
    # Generate data
    n_samples = 500
    tv_advertising = np.random.uniform(0, 100, n_samples)  # in thousands
    radio_advertising = np.random.uniform(0, 50, n_samples)
    social_media_spend = np.random.uniform(0, 30, n_samples)
    seasonality = np.random.choice([0.8, 1.0, 1.2, 1.5], n_samples)  # seasonal factors
    
    # Sales = base + advertising effects + seasonality + noise
    sales = (100 + 
             2.5 * tv_advertising + 
             1.8 * radio_advertising + 
             3.2 * social_media_spend + 
             50 * seasonality + 
             np.random.normal(0, 10, n_samples))
    
    data = pd.DataFrame({
        'TV_Advertising': tv_advertising,
        'Radio_Advertising': radio_advertising,
        'Social_Media_Spend': social_media_spend,
        'Seasonality': seasonality,
        'Sales': sales
    })
    
    X = data[['TV_Advertising', 'Radio_Advertising', 'Social_Media_Spend', 'Seasonality']]
    y = data['Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("Sales Prediction Model:")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Coefficients: {dict(zip(X.columns, model.coef_))}")
    print()
    
    return model, data


# ============================================================================
# SCENARIO 2: Temperature Prediction
# ============================================================================

def temperature_prediction_model():
    """Predict temperature based on time of year and location"""
    np.random.seed(42)
    
    n_samples = 365
    day_of_year = np.arange(1, 366)
    latitude = np.random.uniform(25, 50, n_samples)  # degrees
    elevation = np.random.uniform(0, 2000, n_samples)  # meters
    
    # Temperature = base + seasonal variation + latitude effect + elevation effect
    seasonal = 20 + 15 * np.sin(2 * np.pi * day_of_year / 365)
    temp = (seasonal - 
            0.5 * latitude - 
            0.0065 * elevation + 
            np.random.normal(0, 3, n_samples))
    
    data = pd.DataFrame({
        'Day_of_Year': day_of_year,
        'Latitude': latitude,
        'Elevation': elevation,
        'Temperature_C': temp
    })
    
    X = data[['Day_of_Year', 'Latitude', 'Elevation']]
    y = data['Temperature_C']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("Temperature Prediction Model:")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {rmse:.2f}°C")
    print(f"Coefficients: {dict(zip(X.columns, model.coef_))}")
    print()
    
    return model, data


# ============================================================================
# SCENARIO 3: Customer Lifetime Value (CLV)
# ============================================================================

def customer_lifetime_value_model():
    """Predict customer lifetime value based on customer behavior"""
    np.random.seed(42)
    
    n_samples = 1000
    age = np.random.randint(18, 70, n_samples)
    monthly_income = np.random.uniform(2000, 10000, n_samples)
    purchase_frequency = np.random.uniform(1, 12, n_samples)  # purchases per year
    avg_order_value = np.random.uniform(50, 500, n_samples)
    years_as_customer = np.random.uniform(0, 10, n_samples)
    
    # CLV = base + income effect + purchase behavior + loyalty
    clv = (1000 + 
           0.1 * monthly_income + 
           200 * purchase_frequency + 
           2 * avg_order_value + 
           500 * years_as_customer + 
           np.random.normal(0, 500, n_samples))
    clv = np.maximum(clv, 0)  # Ensure non-negative
    
    data = pd.DataFrame({
        'Age': age,
        'Monthly_Income': monthly_income,
        'Purchase_Frequency': purchase_frequency,
        'Avg_Order_Value': avg_order_value,
        'Years_as_Customer': years_as_customer,
        'CLV': clv
    })
    
    X = data[['Age', 'Monthly_Income', 'Purchase_Frequency', 'Avg_Order_Value', 'Years_as_Customer']]
    y = data['CLV']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use Ridge regression for regularization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("Customer Lifetime Value Model:")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"Coefficients: {dict(zip(X.columns, model.coef_))}")
    print()
    
    return model, data, scaler


# ============================================================================
# SCENARIO 4: Energy Consumption Prediction
# ============================================================================

def energy_consumption_model():
    """Predict energy consumption based on weather and building characteristics"""
    np.random.seed(42)
    
    n_samples = 1000
    temperature = np.random.uniform(-10, 35, n_samples)  # Celsius
    building_size = np.random.uniform(500, 5000, n_samples)  # sqft
    occupancy = np.random.uniform(1, 20, n_samples)  # number of people
    insulation_rating = np.random.uniform(1, 10, n_samples)  # 1-10 scale
    
    # Energy consumption increases with temperature deviation from optimal (20°C)
    temp_effect = np.abs(temperature - 20) * 2
    energy = (50 * building_size / 1000 + 
              100 * occupancy + 
              temp_effect * 50 - 
              20 * insulation_rating + 
              np.random.normal(0, 50, n_samples))
    energy = np.maximum(energy, 0)
    
    data = pd.DataFrame({
        'Temperature': temperature,
        'Building_Size': building_size,
        'Occupancy': occupancy,
        'Insulation_Rating': insulation_rating,
        'Energy_Consumption': energy
    })
    
    X = data[['Temperature', 'Building_Size', 'Occupancy', 'Insulation_Rating']]
    y = data['Energy_Consumption']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("Energy Consumption Model:")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {rmse:.2f} kWh")
    print(f"Coefficients: {dict(zip(X.columns, model.coef_))}")
    print()
    
    return model, data


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Alternative Regression Scenarios")
    print("=" * 70)
    print()
    
    # Run all scenarios
    sales_model, sales_data = sales_prediction_model()
    temp_model, temp_data = temperature_prediction_model()
    clv_model, clv_data, scaler = customer_lifetime_value_model()
    energy_model, energy_data = energy_consumption_model()
    
    print("=" * 70)
    print("All models trained successfully!")
    print("=" * 70)
    
    # Example: Make a prediction with sales model
    print("\nExample Prediction - Sales Model:")
    new_campaign = pd.DataFrame({
        'TV_Advertising': [75],
        'Radio_Advertising': [30],
        'Social_Media_Spend': [20],
        'Seasonality': [1.2]
    })
    predicted_sales = sales_model.predict(new_campaign)
    print(f"Predicted Sales for new campaign: ${predicted_sales[0]:.2f}")

