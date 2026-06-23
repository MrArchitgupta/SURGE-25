# Transfer Learning: Predicting India's GDP Growth from Nifty50 Stock Market Data

## 🎯 Project Overview

Can deep learning models trained on stock market data learn economic patterns useful for predicting GDP growth? This project explores **transfer learning** to answer that question by pre-training a CNN-LSTM neural network on daily Nifty50 stock data, then transferring the learned representations to predict India's annual GDP growth.

**Key Insight:** Stock markets are called the "barometer of the economy" — they encode investor sentiment, corporate earnings, and macroeconomic health in real time. Using transfer learning, we test whether these patterns generalize to GDP prediction.

---

## 📊 What's Inside

### Core Pipeline (4 Phases)

```
┌─────────────────────────────────────────────────────────────┐
│                    FULL PIPELINE                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: PRE-TRAIN on Nifty50 daily/weekly data           │
│  ├─ Input: 60 days of OHLCV (Open, High, Low, Close, Vol) │
│  ├─ Model: Conv1D → MaxPool → LSTM(64)                     │
│  └─ Output: Next-period Close price prediction             │
│                                                             │
│  Phase 2: EXTRACT encoder features                         │
│  ├─ Remove final Dense prediction layer                    │
│  ├─ Aggregate daily/weekly data to yearly statistics       │
│  └─ Extract 64-dimensional LSTM embeddings per year        │
│                                                             │
│  Phase 3: FINE-TUNE on GDP data                            │
│  ├─ Combine: LSTM features + macroeconomic indicators      │
│  ├─ Train 6 models (3 transfer + 3 baseline)               │
│  └─ Evaluate with Leave-One-Out Cross-Validation (LOOCV)   │
│                                                             │
│  Phase 4: ANALYZE results                                  │
│  ├─ Generate predictions vs actual plots                   │
│  ├─ Feature importance analysis                            │
│  └─ Quantify transfer learning impact                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data

| Dataset | Source | Time Range | Frequency | Records |
|---------|--------|-----------|-----------|---------|
| **Nifty50 Weekly** | Yahoo Finance | 2004–2025 | Weekly | ~1,100 observations |
| **Nifty50 Daily** | Yahoo Finance | 2014–2025 | Daily | ~2,800 observations |
| **GDP & Macros** | World Bank/IMF | 1973–2020 | Annual | 48 years |
| **Overlap** | Merged | 2014–2020 | Annual | **~7–12 years** |

**Key Features:**
- **Nifty50:** Open, High, Low, Close, Volume
- **Macroeconomic:** Imports/Exports (% GDP), Inflation, USD-INR Exchange Rate
- **Target:** Annual GDP Growth (%)

### Results

**Best Model: Transfer NN**

| Metric | Value |
|--------|-------|
| **RMSE** | 4.02 pp (percentage points) |
| **MAE** | 2.12 pp |
| **R²** | 0.38 (explains 38% of variance) |
| **MAPE** | 31.5% |

**Transfer Learning Impact:**
- **Neural Network:** +18.5% RMSE improvement (4.94 → 4.02)
- **Gradient Boosting:** -17.8% (transfer hurt due to curse of dimensionality)
- **Ridge Regression:** -7.2% (linear model cannot leverage non-linear LSTM features)

**Key Finding:** Transfer learning significantly improves neural networks, but not tree-based or linear models on this small dataset.

---

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **Deep Learning:** TensorFlow 2.19
- **Data Processing:** Pandas, NumPy
- **ML Models:** scikit-learn (GradientBoostingRegressor, Ridge)
- **Visualization:** Matplotlib
- **CV Strategy:** Leave-One-Out Cross-Validation (essential for 7-sample dataset)

---

## 📁 Repository Structure

```
SURGE-25/
├── README.md                          ← You are here
├── gdp_transfer_learning.py            ← Main pipeline (self-contained)
├── report.md                           ← Detailed technical report
├── nifty50_weekly_20years.csv         ← Weekly stock data (2004-2025)
├── nifty50_data.csv                   ← Additional daily stock data
├── merged_gdp_features.csv            ← GDP + macroeconomic data (1973-2020)
├── trade_india-checkpoint.csv         ← Trade/commerce data
├── Untitled.ipynb                     ← Exploratory notebook
├── main.ipynb                         ← Main analysis notebook
└── results/                           ← Generated outputs
    ├── gdp_prediction_results.png     ← All 6 models' predictions
    ├── transfer_comparison.png        ← Transfer vs baseline deep-dive
    ├── feature_importance.png         ← Feature importance analysis
    ├── nifty_pretrain_predictions.png ← Phase 1 Nifty50 predictions
    └── phase1_training_curve.png      ← CNN-LSTM training dynamics
```

---

## 🚀 Quick Start

### Requirements

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

### Run the Full Pipeline

```bash
python gdp_transfer_learning.py
```

**Output:**
- Console logs showing all 4 phases + results
- 5 PNG visualizations saved to `results/`
- Predictions from 6 models (3 transfer + 3 baseline)

### Expected Runtime
- **Phase 1 (Pre-training):** ~5–10 minutes (depends on your hardware)
- **Phase 2 (Feature engineering):** ~30 seconds
- **Phase 3 (Training):** ~2–3 minutes
- **Phase 4 (Visualization):** ~1 minute

---

## 📈 Model Architecture

### Phase 1: CNN-LSTM Encoder

```
Input: (60 timesteps × 5 features)
   ↓
Conv1D(64 filters, kernel=3, ReLU)      ← Detects local patterns
   ↓
MaxPooling1D(pool=2)                     ← Reduces temporal dim
   ↓
LSTM(64 units, ReLU)  ← THE ENCODER     ← Captures long-term dependencies
   ↓                                       (outputs 64-dim vectors)
Dense(1)              ← THE HEAD         ← Predicts next-period close
   ↓
Output: Scalar price
```

**Why CNN-LSTM?**
- **Conv1D:** Extracts local patterns (support/resistance, momentum)
- **LSTM:** Captures temporal dependencies (trends, seasonality, regimes)

### Phase 3: Fine-tuning Head

**Transfer Model (73 features):**
```
Input: [4 Macro + 5 Nifty Stats + 64 LSTM features]
   ↓
Dense(32, ReLU) + L2(0.01)
   ↓
Dropout(0.3)
   ↓
Dense(16, ReLU) + L2(0.01)
   ↓
Dense(1, Linear)
   ↓
Output: GDP Growth (%)
```

**Baseline Model (9 features):**
```
Input: [4 Macro + 5 Nifty Stats]  ← NO LSTM features
   ↓
Same architecture
```

---

## 🎓 Key Learnings

### Transfer Learning Works for Neural Networks
- **Compatible architecture:** LSTM embeddings → Dense layers is a natural fit
- **Rich representations:** 64-dim LSTM features encode non-linear stock patterns
- **Implicit regularization:** Compressed representation reduces noise vs raw statistics

### Transfer Learning Fails for Tree-based & Linear Models
- **GBR:** 64 LSTM features + 7 data points = overfitting (curse of dimensionality)
- **Ridge:** Linear model cannot leverage non-linear LSTM features; dilutes signal

### The 2020 Anomaly
Stock market (+14.8%) and GDP (-7.25%) decoupled in 2020 due to:
- COVID lockdowns destroyed the real economy
- RBI monetary stimulus + retail investor boom inflated markets
- **Implication:** Transfer learning from stock data works better in normal periods

---

## 📊 Visualizations

### Plot 1: GDP Predictions (All 6 Models)
Three subplots comparing:
- Transfer NN vs Baseline NN
- Transfer GBR vs Baseline GBR
- Transfer Ridge vs Baseline Ridge
- Bar chart: R² and RMSE across all models

### Plot 2: Transfer Learning Impact (Best Model)
- **Residuals:** Actual vs predicted errors over time
- **Absolute errors:** Per-year breakdown (2020 dominates)
- **Metrics table:** Side-by-side comparison with delta

### Plot 3: Feature Importance
- **Transfer model:** Dominated by LSTM features (confirms transfer learning worked)
- **Baseline model:** Relies on macro indicators (Imports_Pct_GDP, USDINR, volatility)

### Plot 4: Nifty50 Pre-training
- Actual vs predicted weekly close prices on test set
- Demonstrates Phase 1 learned meaningful stock patterns

---

## 📝 Detailed Documentation

See **`report.md`** for:
- Complete methodology (2.1K+ words)
- Data descriptions with samples
- Phase-by-phase breakdown with code walkthroughs
- Results interpretation & discussion
- Limitations & future work

---

## 🔍 How to Interpret Results

**RMSE = 4.02 percentage points**
- Model predictions are off by ~4% on average
- E.g., if true GDP growth = 7%, model might predict 3% or 11%
- For a small dataset, this is reasonable

**R² = 0.38**
- Model explains 38% of GDP growth variance
- Remaining 62% due to:
  - Black swan events (COVID, wars)
  - Non-market factors (government policy, external shocks)
  - Data sparsity (only 7 overlapping years)

**Why Transfer NN > Baseline NN?**
- Transfer adds 64 LSTM features encoding 2,800 daily observations
- This gives the neural network richer representations of economic cycles
- Feature importance confirms LSTM features (not macro) drive predictions

---

## ⚠️ Limitations & Caveats

| Limitation | Impact | Workaround |
|------------|--------|-----------|
| **Only 7 overlapping years** | High variance; low generalization | Use historical Nifty data (NSE goes back to 1995) |
| **Annual GDP granularity** | Severely limits sample size | Use quarterly GDP estimates (RBI publishes) |
| **2020 COVID outlier** | Distorts metrics (market/economy decoupled) | Train separate models for crisis vs normal periods |
| **No causal mechanism** | Correlation ≠ causation | Use as ensemble component, not sole predictor |
| **Single country** | Results may not transfer to other markets | Test with SENSEX, other emerging market indices |

---

## 🔮 Future Work

1. **Extend temporal coverage:** Use NSE historical data (1995–present) → 30 years of overlap
2. **Quarterly GDP:** Switch to RBI quarterly estimates (48× more samples)
3. **Ensemble:** Combine transfer learning with traditional econometric models
4. **Causal inference:** Add Granger causality tests to determine if markets *predict* GDP
5. **Multi-market:** Test transfer learning on other economies (Brazil, Mexico, etc.)
6. **Sector-level:** Train separate models on BSE sector indices

---

## 📚 References

- **Transfer Learning:** Yosinski et al., "How transferable are features in deep neural networks?" (NeurIPS 2014)
- **Time Series:** Hochreiter & Schmidhuber, "Long Short-Term Memory" (Neural Computation 1997)
- **CNN-LSTM hybrids:** Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting" (NeurIPS 2015)
- **Small sample ML:** Cawley & Talbot, "On over-fitting in model selection and subsequent selection bias in performance evaluation" (JMLR 2010)

---

## 🤝 Contributing

Found a bug? Have an idea? Issues and PRs welcome!

---

## 📄 License

This project is open source. Feel free to use, modify, and build upon it.

---

## 👤 Author

**Arhit Gupta** (MrArchitgupta)
- Project for SURGE 2025
- June 2026

---

## 🙋 Questions?

Refer to:
1. `report.md` for deep-dive technical details
2. `gdp_transfer_learning.py` for code comments
3. Generated plots in `results/` for visual intuition
4. Jupyter notebooks (`main.ipynb`, `Untitled.ipynb`) for exploratory analysis

---

**Happy forecasting! 📈**
