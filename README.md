# Retail Demand Forecasting & Inventory Recommendation

A Streamlit app that forecasts product demand using time series models and converts the forecast into a stock-ordering recommendation (predictive → prescriptive analytics).

Mini project for the **Predictive Analytics** course (Time Series Analysis, Model Validation and Optimization).

## Features

- **Data preparation:** missing value interpolation and outlier clipping
- **EDA:** seasonal decomposition (trend, seasonality, noise) and ADF stationarity test
- **Models:** Seasonal Naive (baseline), Holt-Winters exponential smoothing, SARIMA (order chosen by grid search over p, q using AIC)
- **Validation:** time-based train-test split (last 30 days held out), compared using RMSE, MAE and MAPE
- **Forecast:** 30-day forecast with 95% interval from the best model
- **Prescriptive layer:** safety stock, reorder point and recommended order quantity based on lead time, service level and current stock
- **Ethics and limitations** note inside the app

## Dataset

`train.csv` is a **synthetic dataset** generated for this project. It follows the column format of the Kaggle *Store Item Demand Forecasting Challenge* (`date, store, item, sales`) and contains 5 stores, 10 items and daily data from 2013 to 2017. It includes weekly and yearly seasonality, a trend, a few outliers and missing values.

The app also works with the real Kaggle `train.csv`. Upload it in the sidebar.

## How to run

```bash
pip install -r requirements.txt
streamlit run demand_forecast.py
```

On Windows, if `pip` or `streamlit` is not recognized:

```bash
py -m pip install -r requirements.txt
py -m streamlit run demand_forecast.py
```

Then upload `train.csv` in the sidebar and choose a store and item.

## How the recommendation is calculated

- Safety stock = z × RMSE × √(lead time), where z comes from the chosen service level
- Reorder point = forecast demand over the lead time + safety stock
- Order quantity = max(0, reorder point − current stock)

## Tech stack

Python, pandas, NumPy, statsmodels, SciPy, matplotlib, Streamlit

## Limitations

Forecasts assume past patterns continue. Promotions, shocks and stock-outs hidden in sales data are not modeled. Over-forecasting causes waste and under-forecasting causes lost sales, so large orders should be reviewed by a person.

## Authors

Yash Khatri and Vansh Chhabra
