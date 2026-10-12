"""Retail Demand Forecasting + Inventory Recommendation (single-file project)
Run:  pip install streamlit pandas numpy matplotlib statsmodels scipy
      streamlit run demand_forecast.py
Data: upload Kaggle 'Store Item Demand Forecasting' train.csv (date,store,item,sales)
      or leave empty to use built-in synthetic data.
"""
import warnings
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from itertools import product
from scipy.stats import norm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("📦 Demand Forecasting & Inventory Recommendation")
st.caption("Predictive → Prescriptive analytics | Time Series: Holt-Winters, SARIMA")


@st.cache_data
def synthetic():
    rng = np.random.default_rng(0)
    d = pd.date_range("2021-01-01", "2023-12-31")
    out = []
    for s in (1, 2):
        for i in (1, 2, 3):
            y = (30 + 10 * i + 5 * s + 0.02 * np.arange(len(d))
                 + 8 * np.sin(2 * np.pi * d.dayofyear / 365)
                 + 6 * (d.dayofweek >= 5) + rng.normal(0, 4, len(d)))
            out.append(pd.DataFrame({"date": d, "store": s, "item": i, "sales": np.clip(np.round(np.asarray(y)), 0, None)}))
    return pd.concat(out)


# ---------- Data ----------
f = st.sidebar.file_uploader("Upload CSV (date, store, item, sales)", type="csv")
df = pd.read_csv(f, parse_dates=["date"]) if f else synthetic()
store = st.sidebar.selectbox("Store", sorted(df.store.unique()))
item = st.sidebar.selectbox("Item", sorted(df.item.unique()))
lead = st.sidebar.slider("Supplier lead time (days)", 1, 14, 7)
service = st.sidebar.slider("Service level", 0.80, 0.99, 0.95)
stock = st.sidebar.number_input("Current stock (units)", 0, 100000, 100)
H = 30

s = (df[(df.store == store) & (df.item == item)].set_index("date")["sales"]
     .asfreq("D").interpolate())                                   # missing values
m, sd = s.mean(), s.std()
s = s.clip(m - 3 * sd, m + 3 * sd)                                  # outliers
s = s[-365:]                                                        # last year (fast)

# ---------- EDA ----------
st.header("1. Exploratory analysis")
c1, c2 = st.columns(2)
dec = seasonal_decompose(s, period=7)
fig, ax = plt.subplots(3, 1, figsize=(6, 5), sharex=True)
for a, (n, v) in zip(ax, [("Trend", dec.trend), ("Seasonal", dec.seasonal), ("Noise", dec.resid)]):
    a.plot(v); a.set_ylabel(n)
c1.pyplot(fig)
p_adf = adfuller(s)[1]
c2.metric("ADF p-value", f"{p_adf:.4f}")
c2.write("Stationary ✅" if p_adf < 0.05 else "Non-stationary → differencing (d=1) used in ARIMA ✅")
c2.line_chart(s)

# ---------- Models ----------
st.header("2. Model comparison (last 30 days = test set)")
train, test = s[:-H], s[-H:]


@st.cache_data
def naive(tr, n):  return np.tile(tr[-7:].values, n // 7 + 1)[:n]
@st.cache_data
def hw(tr, n):
    return ExponentialSmoothing(tr, trend="add", seasonal="add", seasonal_periods=7).fit().forecast(n).values
@st.cache_data
def sarima(tr, n):
    best = min(((SARIMAX(tr, order=(p, 1, q), seasonal_order=(1, 1, 1, 7)).fit(disp=False), (p, q))
                for p, q in product([0, 1, 2], [0, 1])), key=lambda x: x[0].aic)  # grid search by AIC
    return best[0].forecast(n).values


models = {"Seasonal Naive": naive, "Holt-Winters": hw, "SARIMA": sarima}
res = {}
for name, fn in models.items():
    p = fn(train, H)
    e = test.values - p
    res[name] = dict(RMSE=np.sqrt((e ** 2).mean()), MAE=np.abs(e).mean(),
                     MAPE=np.mean(np.abs(e / np.maximum(test.values, 1))) * 100)
tbl = pd.DataFrame(res).T.round(2)
best = tbl.RMSE.idxmin()
st.dataframe(tbl, use_container_width=True)
st.success(f"Best model: **{best}** (lowest RMSE)")

# ---------- Forecast ----------
st.header("3. 30-day forecast with 95% interval")
fc = models[best](s, H)
sig = tbl.loc[best, "RMSE"]
idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=H)
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(s[-60:], label="History")
ax.plot(idx, fc, label="Forecast", color="tab:orange")
ax.fill_between(idx, fc - 1.96 * sig, fc + 1.96 * sig, alpha=0.25, color="tab:orange")
ax.legend()
st.pyplot(fig)

# ---------- Prescriptive ----------
st.header("4. Inventory recommendation (prescriptive)")
z = norm.ppf(service)
ss = z * sig * np.sqrt(lead)                      # safety stock from forecast error
lead_demand = fc[:lead].sum()
rop = lead_demand + ss                            # reorder point
order = max(0, int(np.ceil(rop - stock)))
a, b, c = st.columns(3)
a.metric("Safety stock", int(np.ceil(ss)))
b.metric("Reorder point", int(np.ceil(rop)))
c.metric("Order now", order)
if order > 0:
    st.warning(f"Order **{order} units** now: stock {stock} is below reorder point {int(np.ceil(rop))}.")
else:
    st.info("Stock is sufficient for now. No order needed.")

with st.expander("Limitations & ethics"):
    st.write("Forecasts assume past patterns continue (no promotions, shocks, or stock-outs hidden in "
             "sales data). Over-forecasting causes waste and cost; under-forecasting causes lost sales. "
             "Always keep human review for large orders.")
