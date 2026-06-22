"""
============================================================================
 Transfer Learning: Nifty50 CNN-LSTM  ->  GDP Prediction
 (v2 - Using 20-year weekly data for broader coverage)
============================================================================
 Pipeline:
   Phase 1  -  Pre-train a CNN-LSTM on WEEKLY Nifty50 data (2008-2025)
   Phase 2  -  Aggregate weekly stock data to yearly features & merge GDP
   Phase 3  -  Transfer: freeze encoder, extract features, fine-tune on GDP
   Phase 4  -  Visualise results (plots saved to results/)
============================================================================
"""

import os, warnings, sys
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

import tensorflow as tf
from tensorflow.keras import layers, models, Model

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQ_LEN          = 12         # 12 weeks = 1 quarter (weekly data)
EPOCHS_PRETRAIN  = 80
EPOCHS_FINETUNE  = 200
BATCH_SIZE       = 32
RANDOM_SEED      = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==========================================================
# PHASE 1 -- Pre-train CNN-LSTM on Nifty50 WEEKLY data
# ==========================================================
print("=" * 65)
print("  PHASE 1: Pre-training CNN-LSTM on Nifty50 WEEKLY Data")
print("=" * 65)

# --- Load Nifty50 WEEKLY data (20 years) ---
WEEKLY_CSV = 'data/nifty50_weekly_20years.csv'
DAILY_YF   = 'nifty50_yf.csv'

# Load weekly data (skip ticker and empty rows)
df_raw = pd.read_csv(WEEKLY_CSV, skiprows=[1, 2])
df_raw['Date'] = pd.to_datetime(df_raw['Date'])

# Use Price, Close, High, Low as features
# Open is 0 and Volume is NaN for early years, so we skip them
FEATURES = ['Price', 'Close', 'High', 'Low']
df_weekly = df_raw[['Date'] + FEATURES].copy()

for col in FEATURES:
    df_weekly[col] = pd.to_numeric(df_weekly[col], errors='coerce')
df_weekly.dropna(inplace=True)
df_weekly.sort_values('Date', inplace=True)
df_weekly.reset_index(drop=True, inplace=True)

print(f"  Loaded {len(df_weekly)} weekly records  ({df_weekly['Date'].min().date()} to {df_weekly['Date'].max().date()})")
print(f"  Features: {FEATURES}")
print(f"  Window size: {SEQ_LEN} weeks (~1 quarter)")

# Also load daily data for supplementary yearly stats
if os.path.exists(DAILY_YF):
    df_daily_raw = pd.read_csv(DAILY_YF, header=[0, 1], index_col=0)
    df_daily_raw.columns = df_daily_raw.columns.get_level_values(0)
    df_daily_raw.index.name = 'Date'
    df_daily_raw.reset_index(inplace=True)
    df_daily_raw['Date'] = pd.to_datetime(df_daily_raw['Date'])
    DAILY_FEATS = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_daily = df_daily_raw[['Date'] + DAILY_FEATS].copy()
    for c in DAILY_FEATS:
        df_daily[c] = pd.to_numeric(df_daily[c], errors='coerce')
    df_daily.dropna(inplace=True)
    df_daily.sort_values('Date', inplace=True)
    df_daily.reset_index(drop=True, inplace=True)
    print(f"  Also loaded {len(df_daily)} daily records for supplementary features")
else:
    df_daily = None
    print("  (No daily data available for supplementary features)")

# --- Scale weekly data ---
nifty_scaler = MinMaxScaler()
nifty_scaled = nifty_scaler.fit_transform(df_weekly[FEATURES].values)

# --- Sliding-window sequences (weekly) ---
def create_sequences(data, seq_len, target_col_idx):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, target_col_idx])
    return np.array(X), np.array(y)

close_idx = FEATURES.index('Price')  # Weekly close/adj close
X_all, y_all = create_sequences(nifty_scaled, SEQ_LEN, close_idx)
split = int(0.8 * len(X_all))
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]
print(f"  Train: {X_train.shape}   Test: {X_test.shape}")
print(f"  Total sliding windows: {len(X_all)}")

# --- Build CNN-LSTM ---
def build_cnn_lstm(input_shape):
    inp = layers.Input(shape=input_shape, name='ts_input')
    x   = layers.Conv1D(64, 3, activation='relu', padding='same', name='conv1d')(inp)
    x   = layers.MaxPooling1D(2, name='maxpool')(x)
    x   = layers.LSTM(64, name='lstm_encoder')(x)
    out = layers.Dense(1, name='pred_head')(x)
    m   = Model(inp, out, name='nifty_cnn_lstm_weekly')
    m.compile(optimizer='adam', loss='mse')
    return m

pretrained = build_cnn_lstm(X_train.shape[1:])
pretrained.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=7, verbose=1
)

print(f"\n  Training CNN-LSTM on {len(X_train)} weekly windows ...")
history_pretrain = pretrained.fit(
    X_train, y_train,
    epochs=EPOCHS_PRETRAIN,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# --- Evaluate pre-trained model ---
y_pred_nifty = pretrained.predict(X_test, verbose=0).ravel()
nifty_rmse = np.sqrt(mean_squared_error(y_test, y_pred_nifty))
nifty_r2   = r2_score(y_test, y_pred_nifty)
print(f"\n  [OK] Nifty Weekly Pre-train -- RMSE: {nifty_rmse:.6f}   R2: {nifty_r2:.4f}")

# --- Save training curve ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history_pretrain.history['loss'], label='Train Loss')
ax.plot(history_pretrain.history['val_loss'], label='Val Loss')
ax.set(xlabel='Epoch', ylabel='MSE Loss', title='Phase 1 -- Nifty CNN-LSTM Training Curve (Weekly Data)')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'phase1_training_curve.png'), dpi=150)
plt.close(fig)
print(f"  [OK] Saved: {RESULTS_DIR}/phase1_training_curve.png")


# ==========================================================
# PHASE 2 -- Feature Engineering
# ==========================================================
print("\n" + "=" * 65)
print("  PHASE 2: Feature Engineering -- Aggregating to Yearly")
print("=" * 65)

df_weekly['Year'] = df_weekly['Date'].dt.year

# --- Yearly summary statistics from WEEKLY Nifty data ---
yearly_nifty = df_weekly.groupby('Year').agg(
    nifty_close_mean  = ('Price', 'mean'),
    nifty_close2_mean = ('Close', 'mean'),
    nifty_high_max    = ('High',  'max'),
    nifty_low_min     = ('Low',   'min'),
    nifty_volatility  = ('Price', 'std'),
    nifty_annual_ret  = ('Price', lambda s: (s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100 if len(s) > 1 else 0),
    nifty_range       = ('Price', lambda s: s.max() - s.min()),
    nifty_weeks       = ('Price', 'count'),
    nifty_high_low_spread = ('High', lambda s: (s.max() - df_weekly.loc[s.index, 'Low'].min())),
).reset_index()

# If daily data available, add extra daily-derived features
if df_daily is not None:
    df_daily['Year'] = df_daily['Date'].dt.year
    daily_yearly = df_daily.groupby('Year').agg(
        daily_volume_mean = ('Volume', 'mean'),
        daily_volume_std  = ('Volume', 'std'),
        daily_volatility  = ('Close', 'std'),
        daily_trading_days= ('Close', 'count'),
    ).reset_index()
    yearly_nifty = yearly_nifty.merge(daily_yearly, on='Year', how='left')

print(f"  Yearly Nifty features shape: {yearly_nifty.shape}")
print(f"  Years covered by weekly data: {yearly_nifty['Year'].min()} to {yearly_nifty['Year'].max()}")

# --- Build encoder (remove final Dense head) ---
encoder = Model(
    inputs  = pretrained.input,
    outputs = pretrained.get_layer('lstm_encoder').output,
    name    = 'encoder',
)
print(f"  Encoder output dim: {encoder.output_shape[-1]}")

# --- Extract per-year encoder embeddings from WEEKLY data ---
def yearly_encoder_embeddings(df, encoder, scaler, features, seq_len):
    emb = {}
    for year, grp in df.groupby('Year'):
        vals = grp[features].values.astype(np.float64)
        if len(vals) < seq_len + 1:
            continue
        scaled = scaler.transform(vals)
        windows = np.array([scaled[i:i+seq_len] for i in range(len(scaled) - seq_len)])
        feats = encoder.predict(windows, verbose=0)
        emb[year] = feats.mean(axis=0)
    return emb

print("  Extracting encoder embeddings per year (from weekly windows) ...")
emb_dict = yearly_encoder_embeddings(df_weekly, encoder, nifty_scaler, FEATURES, SEQ_LEN)
print(f"  -> Embeddings for {len(emb_dict)} years  (vector dim = {len(list(emb_dict.values())[0])})")

emb_df = pd.DataFrame(emb_dict).T
emb_df.index.name = 'Year'
emb_df.columns = [f'lstm_feat_{i}' for i in range(emb_df.shape[1])]
emb_df.reset_index(inplace=True)

# --- Load GDP data ---
df_gdp = pd.read_csv('merged_gdp_features.csv')
df_gdp['Year'] = pd.to_datetime(df_gdp['Year']).dt.year

# --- Merge GDP + yearly Nifty stats + encoder embeddings ---
merged = (
    df_gdp
    .merge(yearly_nifty, on='Year', how='inner')
    .merge(emb_df,       on='Year', how='inner')
    .sort_values('Year')
    .reset_index(drop=True)
)
print(f"\n  Merged dataset: {merged.shape[0]} rows x {merged.shape[1]} cols")
print(f"  Years: {merged['Year'].min()} to {merged['Year'].max()}")
print(f"  *** {merged.shape[0]} years of overlap (vs 7 in the previous daily-only version) ***")
print()
print(merged[['Year','GDP_USD','GDP_Growth','nifty_annual_ret','nifty_volatility']].to_string(index=False))


# ==========================================================
# PHASE 3 -- Transfer Learning for GDP Prediction
# ==========================================================
print("\n" + "=" * 65)
print("  PHASE 3: Transfer Learning -- GDP Prediction")
print("=" * 65)

TARGET = 'GDP_Growth'
y_gdp  = merged[TARGET].values

MACRO_FEATS  = ['Imports_Pct_GDP', 'Exports_Pct_GDP', 'Inflation', 'USDINR']
NIFTY_FEATS  = [c for c in merged.columns if c.startswith('nifty_') or c.startswith('daily_')]
NIFTY_FEATS  = [c for c in NIFTY_FEATS if c in merged.columns and merged[c].notna().all()]
LSTM_FEATS   = [c for c in merged.columns if c.startswith('lstm_feat_')]

print(f"  Macro features: {len(MACRO_FEATS)}")
print(f"  Nifty stat features: {len(NIFTY_FEATS)}")
print(f"  LSTM encoder features: {len(LSTM_FEATS)}")
print(f"  Target: {TARGET}")
print(f"  Samples: {len(y_gdp)}")

# Feature sets
transfer_cols = MACRO_FEATS + NIFTY_FEATS + LSTM_FEATS
baseline_cols = MACRO_FEATS + NIFTY_FEATS

X_transfer = merged[transfer_cols].values.astype(np.float64)
X_baseline = merged[baseline_cols].values.astype(np.float64)

# Scale
scaler_t = StandardScaler(); X_t = scaler_t.fit_transform(X_transfer)
scaler_b = StandardScaler(); X_b = scaler_b.fit_transform(X_baseline)

# Target scaling (for NN models)
scaler_y = StandardScaler(); y_s = scaler_y.fit_transform(y_gdp.reshape(-1, 1)).ravel()

# -- Helper: Leave-One-Out CV with a Keras NN --
def nn_loocv(X, y_scaled, y_orig, label, y_scaler, epochs=EPOCHS_FINETUNE):
    loo = LeaveOneOut()
    preds = np.zeros(len(y_orig))
    for tr, te in loo.split(X):
        m = models.Sequential([
            layers.Dense(32, activation='relu', input_shape=(X.shape[1],),
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(1),
        ])
        m.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
        m.fit(X[tr], y_scaled[tr], epochs=epochs, batch_size=4, verbose=0)
        preds[te] = m.predict(X[te], verbose=0).ravel()
    preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
    return preds, _metrics(y_orig, preds, label)

# -- Helper: Leave-One-Out CV with sklearn model --
def sk_loocv(X, y, model_cls, label, **kw):
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        m = model_cls(**kw)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds, _metrics(y, preds, label)

def _metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-8, 1, y_true))) * 100
    print(f"  {label:<38s}  RMSE={rmse:7.4f}  MAE={mae:7.4f}  R2={r2:7.4f}  MAPE={mape:5.1f}%")
    return dict(RMSE=rmse, MAE=mae, R2=r2, MAPE=mape)

# -- Run all models --
print()
results = {}

print("  -- Neural Network (Dense) --")
p1, m1 = nn_loocv(X_t, y_s, y_gdp, "Transfer NN  (LSTM feats + macro)", scaler_y)
p2, m2 = nn_loocv(X_b, y_s, y_gdp, "Baseline NN  (macro only)",        scaler_y)
results['Transfer NN'] = (p1, m1)
results['Baseline NN'] = (p2, m2)

print("\n  -- Gradient Boosting Regressor --")
p3, m3 = sk_loocv(X_t, y_gdp, GradientBoostingRegressor,
                   "Transfer GBR (LSTM feats + macro)",
                   n_estimators=100, max_depth=3, random_state=RANDOM_SEED)
p4, m4 = sk_loocv(X_b, y_gdp, GradientBoostingRegressor,
                   "Baseline GBR (macro only)",
                   n_estimators=100, max_depth=3, random_state=RANDOM_SEED)
results['Transfer GBR'] = (p3, m3)
results['Baseline GBR'] = (p4, m4)

print("\n  -- Ridge Regression --")
p5, m5 = sk_loocv(X_t, y_gdp, Ridge, "Transfer Ridge (LSTM feats + macro)", alpha=1.0)
p6, m6 = sk_loocv(X_b, y_gdp, Ridge, "Baseline Ridge (macro only)",        alpha=1.0)
results['Transfer Ridge'] = (p5, m5)
results['Baseline Ridge'] = (p6, m6)


# ==========================================================
# PHASE 4 -- Visualisation & Results
# ==========================================================
print("\n" + "=" * 65)
print("  PHASE 4: Visualisation & Results")
print("=" * 65)

years = merged['Year'].values

# ------------ Plot 1: Predictions Overview ---------------
fig = plt.figure(figsize=(18, 14))
fig.suptitle(f'Transfer Learning: Nifty50 Weekly LSTM -> GDP Growth Prediction\n({len(years)} years of data: {years[0]}-{years[-1]})',
             fontsize=17, fontweight='bold', y=0.98)
gs = GridSpec(2, 2, hspace=0.35, wspace=0.30)

pairs = [
    ('Transfer NN',    'Baseline NN',    'Neural Network',       '#2196F3', '#FF5722'),
    ('Transfer GBR',   'Baseline GBR',   'Gradient Boosting',    '#4CAF50', '#FF9800'),
    ('Transfer Ridge', 'Baseline Ridge', 'Ridge Regression',     '#9C27B0', '#E91E63'),
]

for idx, (tname, bname, title, tcol, bcol) in enumerate(pairs):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    ax.plot(years, y_gdp, 'ko-', lw=2, ms=7, label='Actual', zorder=5)
    ax.plot(years, results[tname][0], 's--', color=tcol, lw=2, ms=6,
            label=f'{tname} (R2={results[tname][1]["R2"]:.3f})')
    ax.plot(years, results[bname][0], '^--', color=bcol, lw=2, ms=6,
            label=f'{bname} (R2={results[bname][1]["R2"]:.3f})')
    ax.set(xlabel='Year', ylabel='GDP Growth (%)', title=title)
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='grey', ls='--', alpha=0.4)

# -- Bar chart in bottom-right cell --
ax = fig.add_subplot(gs[1, 1])
names  = list(results.keys())
r2vals = [results[n][1]['R2']   for n in names]
rmvals = [results[n][1]['RMSE'] for n in names]
clrs   = ['#2196F3','#FF5722','#4CAF50','#FF9800','#9C27B0','#E91E63']
x = np.arange(len(names)); w = 0.35
ax.bar(x - w/2, r2vals, w, color=clrs, alpha=0.85, label='R2')
ax2 = ax.twinx()
ax2.bar(x + w/2, rmvals, w, color=clrs, alpha=0.35, hatch='//', label='RMSE')
ax.set_ylabel('R2 Score'); ax2.set_ylabel('RMSE')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha='right', fontsize=7)
ax.set_title('All Models -- R2  &  RMSE')
ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3, axis='y')

fig.savefig(os.path.join(RESULTS_DIR, 'gdp_prediction_results.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  [OK] Saved: {RESULTS_DIR}/gdp_prediction_results.png")

# ------------ Plot 2: Transfer vs Baseline Deep-dive ---------------
# Pick the best transfer model type for deep-dive
best_tag = min(['NN', 'GBR', 'Ridge'],
               key=lambda t: results[f'Transfer {t}'][1]['RMSE'])
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.suptitle(f'Transfer Learning Impact -- {best_tag} (Best Transfer Model)', fontsize=14, fontweight='bold')

res_t = y_gdp - results[f'Transfer {best_tag}'][0]
res_b = y_gdp - results[f'Baseline {best_tag}'][0]

ax = axes[0]
ax.scatter(years, res_t, c='#4CAF50', s=110, edgecolors='k', zorder=5, label='Transfer')
ax.scatter(years, res_b, c='#FF9800', s=110, edgecolors='k', zorder=5, marker='^', label='Baseline')
ax.axhline(0, color='k', ls='--', alpha=.4)
ax.set(xlabel='Year', ylabel='Residual', title='Prediction Residuals'); ax.legend(); ax.grid(alpha=.3)

ax = axes[1]
xp = np.arange(len(years))
ax.bar(xp - .2, np.abs(res_t), .4, color='#4CAF50', alpha=.8, label='Transfer')
ax.bar(xp + .2, np.abs(res_b), .4, color='#FF9800', alpha=.8, label='Baseline')
ax.set_xticks(xp); ax.set_xticklabels(years, rotation=45)
ax.set(xlabel='Year', ylabel='|Error|', title='Absolute Error by Year'); ax.legend(); ax.grid(alpha=.3, axis='y')

ax = axes[2]; ax.axis('off')
mt = results[f'Transfer {best_tag}'][1]
mb = results[f'Baseline {best_tag}'][1]
def _imp(t, b):
    if b == 0: return '-'
    return f"{(b - t) / abs(b) * 100:+.1f}%"
tbl = [
    ['Metric', f'Transfer\n{best_tag}', f'Baseline\n{best_tag}', 'Delta'],
    ['RMSE',  f"{mt['RMSE']:.4f}",  f"{mb['RMSE']:.4f}",  _imp(mt['RMSE'], mb['RMSE'])],
    ['MAE',   f"{mt['MAE']:.4f}",   f"{mb['MAE']:.4f}",   _imp(mt['MAE'],  mb['MAE'])],
    ['R2',    f"{mt['R2']:.4f}",    f"{mb['R2']:.4f}",    f"{mt['R2'] - mb['R2']:+.4f}"],
    ['MAPE',  f"{mt['MAPE']:.1f}%", f"{mb['MAPE']:.1f}%", _imp(mt['MAPE'], mb['MAPE'])],
]
table = ax.table(cellText=tbl, loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.2, 2.0)
for j in range(4):
    table[0, j].set_facecolor('#1a237e')
    table[0, j].set_text_props(color='white', fontweight='bold')
ax.set_title('Metrics Comparison', fontweight='bold', pad=20)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'transfer_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  [OK] Saved: {RESULTS_DIR}/transfer_comparison.png")

# ------------ Plot 3: Feature Importance ---------------
fig, axes = plt.subplots(1, 2, figsize=(17, 8))
fig.suptitle('Feature Importance -- Gradient Boosting', fontsize=14, fontweight='bold')

for ax, X_data, feat_names, label, color in [
    (axes[0], X_t, transfer_cols, 'Transfer Model (top 20)', '#4CAF50'),
    (axes[1], X_b, baseline_cols, 'Baseline Model',          '#FF9800'),
]:
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=RANDOM_SEED)
    gbr.fit(X_data, y_gdp)
    imp = gbr.feature_importances_
    top = np.argsort(imp)[-20:]
    ax.barh(range(len(top)), imp[top], color=color, alpha=.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(np.array(feat_names)[top], fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title(label)
    ax.grid(alpha=.3, axis='x')

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  [OK] Saved: {RESULTS_DIR}/feature_importance.png")

# ------------ Plot 4: Nifty Pre-training -- Actual vs Predicted ---------------
fig, ax = plt.subplots(figsize=(12, 5))
dummy = np.zeros((len(y_test), len(FEATURES)))
dummy[:, close_idx] = y_test
actual_close = nifty_scaler.inverse_transform(dummy)[:, close_idx]

dummy[:, close_idx] = y_pred_nifty
pred_close = nifty_scaler.inverse_transform(dummy)[:, close_idx]

ax.plot(actual_close, label='Actual Weekly Close', color='#1a237e', lw=1.5)
ax.plot(pred_close,   label='Predicted Weekly Close', color='#e53935', lw=1.5, alpha=.8)
ax.set(xlabel='Test Sample Index (Weeks)', ylabel='Nifty50 Weekly Close Price',
       title=f'Phase 1 -- Nifty50 Weekly CNN-LSTM  (RMSE={nifty_rmse:.4f},  R2={nifty_r2:.4f})')
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'nifty_pretrain_predictions.png'), dpi=150)
plt.close(fig)
print(f"  [OK] Saved: {RESULTS_DIR}/nifty_pretrain_predictions.png")


# ==========================================================
# FINAL SUMMARY
# ==========================================================
print("\n" + "=" * 65)
print("  FINAL RESULTS SUMMARY")
print("=" * 65)

print(f"\n  Data: {len(years)} years ({years[0]}-{years[-1]}), "
      f"pre-trained on {len(X_all)} weekly windows ({df_weekly['Date'].min().year}-{df_weekly['Date'].max().year})")

print(f"\n  {'Model':<38s} {'RMSE':>7s} {'MAE':>7s} {'R2':>7s} {'MAPE':>7s}")
print("  " + "-" * 63)
for name, (_, m) in results.items():
    print(f"  {name:<38s} {m['RMSE']:7.4f} {m['MAE']:7.4f} {m['R2']:7.4f} {m['MAPE']:6.1f}%")

best = min(results.items(), key=lambda kv: kv[1][1]['RMSE'])
print(f"\n  >>> Best Model: {best[0]}  (RMSE={best[1][1]['RMSE']:.4f}  R2={best[1][1]['R2']:.4f})")

print("\n  Transfer-learning improvement (RMSE reduction):")
for tag in ('NN', 'GBR', 'Ridge'):
    t = results[f'Transfer {tag}'][1]['RMSE']
    b = results[f'Baseline {tag}'][1]['RMSE']
    pct = (b - t) / b * 100
    icon = '[+]' if pct > 0 else '[-]'
    print(f"    {icon}  {tag:6s}: {pct:+.1f}%  ({b:.4f} -> {t:.4f})")

print(f"\n  All plots saved to: {os.path.abspath(RESULTS_DIR)}")
print("  Done!\n")
