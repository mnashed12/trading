# advanced_stock_prediction.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle

try:
    df = pd.read_csv('Nvidia_data.csv')
except FileNotFoundError:
    raise FileNotFoundError("CSV file 'Nvidia_data.csv' not found. Please check the file path.")

#Cleaning function
def clean_and_convert_to_float(value):
    try:
        if isinstance(value, str):
            cleaned_value = value.replace('$', '').replace(',', '')
            return float(cleaned_value)
        else:
            return float(value)
    except ValueError:
        return np.nan

for col in df.columns:
    df[col] = df[col].apply(clean_and_convert_to_float)

if len(df) == 0:
    raise ValueError("After cleaning, there are no samples left in the dataset.")

# Features
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Dayofweek'] = df['Date'].dt.dayofweek
df['Dayofyear'] = df['Date'].dt.dayofyear
df['Weekofyear'] = df['Date'].dt.isocalendar().week
df['Is_month_end'] = df['Date'].dt.is_month_end
df['Is_month_start'] = df['Date'].dt.is_month_start
df['Is_quarter_end'] = df['Date'].dt.is_quarter_end
df['Is_quarter_start'] = df['Date'].dt.is_quarter_start
df['Is_year_end'] = df['Date'].dt.is_year_end
df['Is_year_start'] = df['Date'].dt.is_year_start

features = ['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear', 'Weekofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start',
            'Is_year_end', 'Is_year_start', 'Open', 'High', 'Low', 'Volume']
target = 'Close/Last'

X = df[features]
y = df[target]

if len(X) == 0:
    raise ValueError("After feature engineering, there are no samples left in the dataset.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training
model_rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

with open('nvidia_stock_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

def predict_stock_price(year, month, day, dayofweek, dayofyear, weekofyear,
                        is_month_end, is_month_start, is_quarter_end, is_quarter_start,
                        is_year_end, is_year_start, open_price, high, low, volume):
                            
    with open('nvidia_stock_model.pkl', 'rb') as f:
        model = pickle.load(f)

    features = np.array([[
        year, month, day, dayofweek, dayofyear, weekofyear,
        is_month_end, is_month_start, is_quarter_end, is_quarter_start,
        is_year_end, is_year_start, open_price, high, low, volume
    ]])

    prediction = model.predict(features)[0]

    prediction_date = pd.Timestamp(year=year, month=month, day=day)
    prediction_time = prediction_date.strftime('%Y-%m-%d %H:%M:%S')

    return prediction, prediction_time

if __name__ == '__main__':
    #Test
    year = 2024
    month = 6
    day = 14
    dayofweek = 4
    dayofyear = 167
    weekofyear = 24
    is_month_end = False
    is_month_start = False
    is_quarter_end = False
    is_quarter_start = False
    is_year_end = False
    is_year_start = False
    open_price = 131.16
    high = 131.89
    low = 130.82
    volume = 306000000

    predicted_price, prediction_time = predict_stock_price(year, month, day, dayofweek, dayofyear, weekofyear,
                                                          is_month_end, is_month_start, is_quarter_end, is_quarter_start,
                                                          is_year_end, is_year_start, open_price, high, low, volume)
    print(f'Predicted Close/Last Price for Nvidia on {prediction_time}: {predicted_price}')
