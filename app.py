#pip install flask pandas numpy requests scikit-learn
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['GET'])
def predict():
    # API Authentication
    post_request_url = 'https://api.pricesenseng.ng/auth/login'
    username = 'admin1234'
    password = 'password.admin1234'

    payload = {
        'username': username,
        'password': password
    }

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(post_request_url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()

        response_json = response.json()
        token = response_json.get('token')

        if not token:
            raise ValueError("Token not found in the response")

    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 400

    if token:
        get_request_url = "https://api.pricesenseng.ng/admin/analysis"
        headers = {
            'Authorization': f'Bearer {token}'
        }

        get_response = requests.get(get_request_url, headers=headers)

        if get_response.status_code == 200:
            data = get_response.json().get('data', [])
            normalized_data = pd.json_normalize(data)
            df = pd.DataFrame(normalized_data)

            # Data Cleaning and Preprocessing
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df[df['date'].dt.year >= 2024]

            groups = df.groupby(['food_item.name', 'distribution_type', 'brand', 'measurement'])
            new_rows = []

            for (food_name, dist_type, brand, measurement), group in groups:
                for index, row in group.iterrows():
                    new_row = {
                        'food_group': f"Food: {food_name}, Distribution Type: {dist_type}, Brand: {brand}, Measurement: {measurement}",
                        'date': row['date'],
                        'price': row['price'],
                        'market.name': row['market.name']
                    }
                    new_rows.append(new_row)

            df2 = pd.DataFrame(new_rows)
            df2['date'] = pd.to_datetime(df2['date'], errors='coerce')

            def process_group(group):
                prices = group['price']
                Q1 = prices.quantile(0.25)
                Q3 = prices.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (prices < lower_bound) | (prices > upper_bound)
                mean_price = prices[~outliers].mean()
                group['price'].fillna(mean_price, inplace=True)
                group.loc[outliers, 'price'] = mean_price
                return group

            df2 = df2.groupby(['food_group', 'date']).apply(process_group).reset_index(drop=True)

            future_dates = ['2024-07-01', '2024-10-01', '2025-01-01', '2025-04-01']
            future_dates = [datetime.strptime(date, '%Y-%m-%d') for date in future_dates]

            predictions = pd.DataFrame(columns=['food_group', 'Q3 2024', 'Q4 2024', 'Q1 2025', 'Q2 2025'])
            df2['date'] = pd.to_datetime(df2['date'])
            df2['date_ordinal'] = df2['date'].apply(lambda x: x.toordinal())
            mean_prices = df2.groupby('food_group')['price'].mean().reset_index()
            mean_prices.columns = ['food_group', 'CurrentPrice_mean']
            food_groups = df2['food_group'].unique()
            true_values = []
            predicted_values = []

            for food_group in food_groups:
                data = df2[df2['food_group'] == food_group]
                if data.shape[0] < 2:
                    continue
                X = data['date_ordinal'].values.reshape(-1, 1)
                y = data['price'].values
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                true_values.extend(y_test)
                predicted_values.extend(y_pred)
                future_dates_ordinal = [date.toordinal() for date in future_dates]
                future_X = np.array(future_dates_ordinal).reshape(-1, 1)
                future_predictions = model.predict(future_X)
                mean_price = mean_prices[mean_prices['food_group'] == food_group]['CurrentPrice_mean'].values[0]
                future_predictions = [max(pred, mean_price) if pred <= 0 else pred for pred in future_predictions]
                pred_dict = {
                    'food_group': food_group,
                    'Q3 2024': future_predictions[0],
                    'Q4 2024': future_predictions[1],
                    'Q1 2025': future_predictions[2],
                    'Q2 2025': future_predictions[3]
                }
                predictions = pd.concat([predictions, pd.DataFrame([pred_dict])], ignore_index=True)

            predictions = predictions.merge(mean_prices, on='food_group', how='left')
            mae = mean_absolute_error(true_values, predicted_values)
            mse = mean_squared_error(true_values, predicted_values)
            rmse = np.sqrt(mse)
            r2 = r2_score(true_values, predicted_values)

            metrics = {
                'Mean Absolute Error (MAE)': mae,
                'Mean Squared Error (MSE)': mse,
                'Root Mean Squared Error (RMSE)': rmse,
                'R-squared (R2)': r2
            }

            return jsonify({
                'predictions': predictions.to_dict(orient='records'),
                'metrics': metrics
            })
        else:
            return jsonify({'error': 'Failed to fetch data from the API'}), 500

    return jsonify({'error': 'Token not found'}), 500

if __name__ == '__main__':
    app.run()