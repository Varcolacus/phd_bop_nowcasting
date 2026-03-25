"""
Baseline & ML Models for BoP Nowcasting Pilot Study
=====================================================

Implements:
  1. AR(1) baseline — simple autoregressive benchmark
  2. AR(4) baseline — richer autoregressive benchmark
  3. OLS with macro features — traditional econometric approach
  4. LASSO (L1) — sparse linear shrinkage (Tibshirani 1996)
  5. Gradient Boosting (scikit-learn) — ML benchmark
  6. XGBoost — primary ML model
  7. LSTM — recurrent neural network
  8. DFM — Dynamic Factor Model (Stock & Watson, 2002)
  9. Bridge — Bridge equation with BIC variable selection (Baffigi et al., 2004)

All models are evaluated using expanding-window pseudo real-time
out-of-sample forecasting.

Author: PhD Pilot Study
Date: March 2026
"""

import logging
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import statsmodels.api as sm

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Time-Series CV Helper for Hyperparameter Tuning
# ---------------------------------------------------------------------------

def _ts_cv_score(X_train, y_train, model_cls, params, n_splits=3):
    """Evaluate a model config via time-series cross-validation (MSE)."""
    if len(y_train) < n_splits * 5:
        return float('inf')
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        try:
            model = model_cls(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(np.mean((y_val - preds) ** 2))
        except Exception as e:
            import warnings
            warnings.warn(f"TSCV fold failed: {e}", RuntimeWarning)
            scores.append(float('inf'))
    return np.mean(scores)


# ---------------------------------------------------------------------------
# Hyperparameter Grids (module-level constants)
# ---------------------------------------------------------------------------

GB_GRID = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
     "min_samples_leaf": 5, "random_state": 42},
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,
     "min_samples_leaf": 5, "random_state": 42},
    {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05,
     "min_samples_leaf": 5, "random_state": 42},
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
     "min_samples_leaf": 5, "random_state": 42},
]

XGB_GRID = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,
     "min_child_weight": 5, "colsample_bytree": 0.8,
     "reg_alpha": 0.1, "reg_lambda": 1.0,
     "random_state": 42, "verbosity": 0},
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
     "min_child_weight": 5, "colsample_bytree": 0.8,
     "reg_alpha": 0.1, "reg_lambda": 1.0,
     "random_state": 42, "verbosity": 0},
    {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1,
     "min_child_weight": 5, "colsample_bytree": 0.8,
     "reg_alpha": 0.1, "reg_lambda": 1.0,
     "random_state": 42, "verbosity": 0},
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,
     "min_child_weight": 5, "colsample_bytree": 1.0,
     "reg_alpha": 0.0, "reg_lambda": 0.0,
     "random_state": 42, "verbosity": 0},
]


# ---------------------------------------------------------------------------
# Bai-Ng (2002) IC_p2 Factor Selection
# ---------------------------------------------------------------------------

def _select_n_factors_icp2(X, max_k=8):
    """Select number of factors using Bai & Ng (2002) IC_p2 criterion."""
    T, N = X.shape
    max_k = min(max_k, N, T - 2)
    if max_k < 1:
        return 1
    best_k, best_ic = 1, float('inf')
    for k in range(1, max_k + 1):
        pca = PCA(n_components=k)
        F = pca.fit_transform(X)
        X_hat = F @ pca.components_
        V_k = np.mean((X - X_hat) ** 2)
        if V_k <= 0:
            continue
        penalty = k * ((N + T) / (N * T)) * np.log(min(N, T))
        ic = np.log(V_k) + penalty
        if ic < best_ic:
            best_ic = ic
            best_k = k
    return best_k


# ---------------------------------------------------------------------------
# Holm-Bonferroni Multiple Testing Correction
# ---------------------------------------------------------------------------

def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction to an array of p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n <= 1:
        return p.copy()
    order = np.argsort(p)
    adjusted = np.empty(n)
    for i, idx in enumerate(order):
        adjusted[idx] = min(1.0, p[idx] * (n - i))
    # enforce monotonicity
    prev = 0.0
    for idx in order:
        adjusted[idx] = max(adjusted[idx], prev)
        prev = adjusted[idx]
    return adjusted


# ---------------------------------------------------------------------------
# Unit-Root Diagnostics (ADF + KPSS)
# ---------------------------------------------------------------------------

def unit_root_tests(df, cols):
    """Run ADF and KPSS tests on specified columns."""
    from statsmodels.tsa.stattools import adfuller, kpss
    rows = []
    for col in cols:
        series = df[col].dropna()
        if len(series) < 20:
            continue
        try:
            adf_stat, adf_p, *_ = adfuller(series, autolag='AIC')
        except Exception:
            adf_stat, adf_p = np.nan, np.nan
        try:
            kpss_stat, kpss_p, *_ = kpss(series, regression='c', nlags='auto')
        except Exception:
            kpss_stat, kpss_p = np.nan, np.nan
        if adf_p < 0.05 and kpss_p > 0.05:
            conclusion = 'Stationary'
        elif adf_p > 0.05 and kpss_p < 0.05:
            conclusion = 'Unit root'
        else:
            conclusion = 'Inconclusive'
        rows.append({'variable': col, 'adf_stat': round(adf_stat, 3),
                     'adf_pvalue': round(adf_p, 4),
                     'kpss_stat': round(kpss_stat, 3),
                     'kpss_pvalue': round(kpss_p, 4),
                     'conclusion': conclusion})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_modeling_data():
    """Load the prepared modeling dataset."""
    path = DATA_DIR / "modeling_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found at {path}. "
            "Run 01_download_data.py first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """Stores results from a single model's evaluation."""
    model_name: str
    predictions: list = field(default_factory=list)
    actuals: list = field(default_factory=list)
    dates: list = field(default_factory=list)
    rmse: float = None
    mae: float = None
    direction_accuracy: float = None


def ar_forecast(y_train, order=1):
    """
    Fit an AR(p) model and return 1-step-ahead forecast.
    Uses statsmodels OLS for simplicity and robustness.
    """
    if len(y_train) < order + 2:
        return np.nan

    # Build lagged features manually for robustness
    y = np.array(y_train)
    X = np.column_stack([y[order - i - 1: len(y) - i - 1] for i in range(order)])
    y_target = y[order:]

    if len(y_target) < 2:
        return np.nan

    X = sm.add_constant(X)
    try:
        model = sm.OLS(y_target, X).fit()
        # Forecast: use the last 'order' observations
        last_obs = np.array([y[-(i + 1)] for i in range(order)])
        x_new = np.concatenate([[1.0], last_obs])
        return model.predict(x_new)[0]
    except Exception as e:
        logger.warning("ar_forecast failed: %s", e)
        return np.nan


def ols_forecast(X_train, y_train, X_test_row):
    """
    Ridge regression with macro features. 1-step-ahead forecast.
    Uses Ridge (L2 penalty) to avoid overfitting when features > observations.
    """
    from sklearn.linear_model import RidgeCV

    if len(y_train) < 10:
        return np.nan

    try:
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        model.fit(X_train, y_train)
        return model.predict(X_test_row.reshape(1, -1))[0]
    except Exception as e:
        logger.warning("ols_forecast (Ridge) failed: %s", e)
        return np.nan


def lasso_forecast(X_train, y_train, X_test_row):
    """
    LASSO (L1 penalty) regression with CV-selected regularisation.
    Produces sparse coefficient vectors — a natural complement to Ridge.
    Uses 3-fold time-series CV for alpha selection.
    """
    from sklearn.linear_model import LassoCV

    if len(y_train) < 10:
        return np.nan

    try:
        model = LassoCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
                         cv=min(3, len(y_train) // 5),
                         max_iter=5000, random_state=42)
        model.fit(X_train, y_train)
        return model.predict(X_test_row.reshape(1, -1))[0]
    except Exception as e:
        logger.warning("lasso_forecast failed: %s", e)
        return np.nan


def gradient_boosting_forecast(X_train, y_train, X_test_row):
    """
    Scikit-learn Gradient Boosting with time-series CV hyperparameter
    selection.  Falls back to default config when training set is too
    small for 3-fold TSCV.
    """
    if len(y_train) < 10:
        return np.nan

    best_params = GB_GRID[0]
    if len(y_train) >= 20:
        best_score = float('inf')
        for params in GB_GRID:
            score = _ts_cv_score(X_train, y_train,
                                GradientBoostingRegressor, params)
            if score < best_score:
                best_score = score
                best_params = params

    try:
        model = GradientBoostingRegressor(**best_params)
        model.fit(X_train, y_train)
        return model.predict(X_test_row.reshape(1, -1))[0]
    except Exception as e:
        logger.warning("gradient_boosting_forecast failed: %s", e)
        return np.nan


def xgboost_forecast(X_train, y_train, X_test_row):
    """
    XGBoost with time-series CV hyperparameter selection.
    """
    if len(y_train) < 10:
        return np.nan

    best_params = XGB_GRID[0]
    if len(y_train) >= 20:
        best_score = float('inf')
        for params in XGB_GRID:
            score = _ts_cv_score(X_train, y_train,
                                xgb.XGBRegressor, params)
            if score < best_score:
                best_score = score
                best_params = params

    try:
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        return model.predict(X_test_row.reshape(1, -1))[0]
    except Exception as e:
        logger.warning("xgboost_forecast failed: %s", e)
        return np.nan


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------

class _LSTMNet(nn.Module if HAS_TORCH else object):
    """Single-layer LSTM with dropout for 1-step-ahead regression."""
    def __init__(self, input_size, hidden_size=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


def lstm_forecast(X_train, y_train, X_test_row, lookback=4, hidden_size=32,
                  epochs=100, lr=0.005):
    """
    LSTM 1-step-ahead forecast with validation-based early stopping
    and learning-rate scheduling.  Uses last 20 % of the training
    window as hold-out validation for early stopping.
    """
    if not HAS_TORCH:
        return np.nan
    if len(y_train) < lookback + 10:
        return np.nan

    torch.manual_seed(42)

    try:
        # Build sequences
        X_seq, y_seq = [], []
        for i in range(lookback, len(y_train)):
            X_seq.append(X_train[i - lookback:i])
            y_seq.append(y_train[i])
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)

        # Train / validation split (last 20 %)
        n_seq = len(y_seq)
        n_val = max(2, int(0.2 * n_seq))
        X_tr, X_val = X_seq[:-n_val], X_seq[-n_val:]
        y_tr, y_val = y_seq[:-n_val], y_seq[-n_val:]

        X_t = torch.from_numpy(X_tr)
        y_t = torch.from_numpy(y_tr)
        X_v = torch.from_numpy(X_val)
        y_v = torch.from_numpy(y_val)

        model = _LSTMNet(X_train.shape[1], hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
        loss_fn = nn.MSELoss()

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

            # Validation-based early stopping
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_v), y_v).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break

        # Predict: use last `lookback` rows of training features
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            x_new = np.vstack([X_train[-(lookback - 1):], X_test_row.reshape(1, -1)])
            x_new = torch.from_numpy(x_new.astype(np.float32)).unsqueeze(0)
            return model(x_new).item()
    except Exception as e:
        logger.warning("lstm_forecast failed: %s", e)
        return np.nan


# ---------------------------------------------------------------------------
# GRU Model (simpler alternative to LSTM)
# ---------------------------------------------------------------------------

class _GRUNet(nn.Module if HAS_TORCH else object):
    """GRU network — fewer parameters than LSTM, often works better in small samples."""
    def __init__(self, input_size, hidden_size=32, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=dropout, num_layers=2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def gru_forecast(X_train, y_train, X_test_row, lookback=4, hidden_size=32,
                 epochs=100, lr=0.005):
    """GRU 1-step-ahead forecast with validation-based early stopping
    and learning-rate scheduling."""
    if not HAS_TORCH:
        return np.nan
    if len(y_train) < lookback + 10:
        return np.nan

    torch.manual_seed(42)

    try:
        X_seq, y_seq = [], []
        for i in range(lookback, len(y_train)):
            X_seq.append(X_train[i - lookback:i])
            y_seq.append(y_train[i])
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)

        # Train / validation split (last 20 %)
        n_seq = len(y_seq)
        n_val = max(2, int(0.2 * n_seq))
        X_tr, X_val = X_seq[:-n_val], X_seq[-n_val:]
        y_tr, y_val = y_seq[:-n_val], y_seq[-n_val:]

        X_t = torch.from_numpy(X_tr)
        y_t = torch.from_numpy(y_tr)
        X_v = torch.from_numpy(X_val)
        y_v = torch.from_numpy(y_val)

        model = _GRUNet(X_train.shape[1], hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
        loss_fn = nn.MSELoss()

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_v), y_v).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            x_new = np.vstack([X_train[-(lookback - 1):], X_test_row.reshape(1, -1)])
            x_new = torch.from_numpy(x_new.astype(np.float32)).unsqueeze(0)
            return model(x_new).item()
    except Exception as e:
        logger.warning("gru_forecast failed: %s", e)
        return np.nan


# ---------------------------------------------------------------------------
# Dynamic Factor Model (Stock & Watson, 2002)
# ---------------------------------------------------------------------------

def dfm_forecast(X_train, y_train, X_test_row, n_factors=None):
    """
    DFM-style nowcasting: extract principal components from the feature
    matrix, then regress the target on the extracted factors.
    Follows Stock & Watson (2002) two-step approach.

    Number of factors is selected automatically via Bai & Ng (2002)
    IC_p2 criterion when n_factors is None.
    """
    if len(y_train) < 15:
        return np.nan

    try:
        if n_factors is None:
            k = _select_n_factors_icp2(X_train)
        else:
            k = n_factors
        k = min(k, X_train.shape[1], len(y_train) - 2)
        if k < 1:
            return np.nan

        pca = PCA(n_components=k)
        F_train = pca.fit_transform(X_train)
        F_test = pca.transform(X_test_row.reshape(1, -1))

        F_train_c = sm.add_constant(F_train)
        # Manually add constant for single-row test data
        F_test_c = np.hstack([[1.0], F_test.flatten()]).reshape(1, -1)

        model = sm.OLS(y_train, F_train_c).fit()
        return model.predict(F_test_c)[0]
    except Exception as e:
        logger.warning("dfm_forecast failed: %s", e)
        return np.nan


# ---------------------------------------------------------------------------
# Bridge Equation (Baffigi et al., 2004)
# ---------------------------------------------------------------------------

def bridge_forecast(X_train, y_train, X_test_row, max_vars=5):
    """
    Bridge equation with forward stepwise variable selection by BIC.
    Standard central-bank nowcasting approach: select a parsimonious set
    of high-frequency indicators via information criteria, then OLS.
    """
    if len(y_train) < 15:
        return np.nan

    try:
        n_features = X_train.shape[1]
        max_k = min(max_vars, n_features, len(y_train) // 5)

        if max_k < 1:
            return np.nan

        selected = []
        remaining = list(range(n_features))
        best_bic = np.inf

        for _ in range(max_k):
            best_new_bic = np.inf
            best_new_var = None

            for var in remaining:
                candidate = selected + [var]
                X_cand = sm.add_constant(X_train[:, candidate])
                try:
                    m = sm.OLS(y_train, X_cand).fit()
                    if m.bic < best_new_bic:
                        best_new_bic = m.bic
                        best_new_var = var
                except Exception as e:
                    logger.debug("bridge variable selection skip: %s", e)
                    continue

            if best_new_var is not None and best_new_bic < best_bic:
                selected.append(best_new_var)
                remaining.remove(best_new_var)
                best_bic = best_new_bic
            else:
                break

        if not selected:
            return np.nan

        X_sel_train = sm.add_constant(X_train[:, selected])
        # Manually add constant for single-row test data
        X_sel_test = np.hstack([[1.0], X_test_row[selected]]).reshape(1, -1)

        model = sm.OLS(y_train, X_sel_train).fit()
        return model.predict(X_sel_test)[0]
    except Exception as e:
        logger.warning("bridge_forecast failed: %s", e)
        return np.nan


# ---------------------------------------------------------------------------
# Expanding Window Evaluation
# ---------------------------------------------------------------------------

def expanding_window_evaluation(df, target_col, feature_cols,
                                 min_train_size=40, step=1):
    """
    Pseudo real-time expanding window out-of-sample evaluation.

    Parameters:
    -----------
    df : DataFrame with date, target, and feature columns
    target_col : str, name of the target variable (BoP component)
    feature_cols : list of str, names of macro feature columns
    min_train_size : int, minimum number of observations before
                     starting out-of-sample evaluation
    step : int, step size for expanding window

    Returns:
    --------
    dict of ForecastResult objects, one per model
    """
    print(f"\nRunning expanding window evaluation...")
    print(f"   Target: {target_col}")
    print(f"   Features: {len(feature_cols)} variables")
    print(f"   Min training size: {min_train_size} quarters")

    # Prepare data — drop rows where target or features are NaN
    cols_needed = [target_col] + feature_cols
    df_clean = df[["date"] + cols_needed].dropna().reset_index(drop=True)

    if len(df_clean) < min_train_size + 5:
        print(f"   [WARN] Not enough data ({len(df_clean)} obs). Need {min_train_size + 5}.")
        return {}

    y = df_clean[target_col].values
    X = df_clean[feature_cols].values
    dates = df_clean["date"].values

    n = len(y)
    n_test = n - min_train_size

    print(f"   Total observations: {n}")
    print(f"   Out-of-sample periods: {n_test}")

    # Initialize results for each model
    models = {
        "AR(1)": ForecastResult("AR(1)"),
        "AR(4)": ForecastResult("AR(4)"),
        "Ridge": ForecastResult("Ridge"),
        "LASSO": ForecastResult("LASSO"),
        "DFM": ForecastResult("DFM"),
        "Bridge": ForecastResult("Bridge"),
        "GradientBoosting": ForecastResult("GradientBoosting"),
        "XGBoost": ForecastResult("XGBoost"),
    }
    if HAS_TORCH:
        models["LSTM"] = ForecastResult("LSTM")
        models["GRU"] = ForecastResult("GRU")

    # Scale features for ML models
    scaler = StandardScaler()

    for t in range(min_train_size, n, step):
        # Training data: everything up to time t
        y_train = y[:t]
        X_train = X[:t]

        # Test observation: time t
        y_test = y[t]
        X_test = X[t]
        test_date = dates[t]

        # Scale features (fit on training data only)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test.reshape(1, -1)).flatten()

        # --- AR(1) ---
        pred_ar1 = ar_forecast(y_train, order=1)
        models["AR(1)"].predictions.append(pred_ar1)
        models["AR(1)"].actuals.append(y_test)
        models["AR(1)"].dates.append(test_date)

        # --- AR(4) ---
        pred_ar4 = ar_forecast(y_train, order=4)
        models["AR(4)"].predictions.append(pred_ar4)
        models["AR(4)"].actuals.append(y_test)
        models["AR(4)"].dates.append(test_date)

        # --- Ridge ---
        pred_ols = ols_forecast(X_train_scaled, y_train, X_test_scaled)
        models["Ridge"].predictions.append(pred_ols)
        models["Ridge"].actuals.append(y_test)
        models["Ridge"].dates.append(test_date)

        # --- LASSO ---
        pred_lasso = lasso_forecast(X_train_scaled, y_train, X_test_scaled)
        models["LASSO"].predictions.append(pred_lasso)
        models["LASSO"].actuals.append(y_test)
        models["LASSO"].dates.append(test_date)

        # --- DFM ---
        pred_dfm = dfm_forecast(X_train_scaled, y_train, X_test_scaled)
        models["DFM"].predictions.append(pred_dfm)
        models["DFM"].actuals.append(y_test)
        models["DFM"].dates.append(test_date)

        # --- Bridge ---
        pred_bridge = bridge_forecast(X_train_scaled, y_train, X_test_scaled)
        models["Bridge"].predictions.append(pred_bridge)
        models["Bridge"].actuals.append(y_test)
        models["Bridge"].dates.append(test_date)

        # --- Gradient Boosting ---
        pred_gb = gradient_boosting_forecast(X_train_scaled, y_train, X_test_scaled)
        models["GradientBoosting"].predictions.append(pred_gb)
        models["GradientBoosting"].actuals.append(y_test)
        models["GradientBoosting"].dates.append(test_date)

        # --- XGBoost ---
        pred_xgb = xgboost_forecast(X_train_scaled, y_train, X_test_scaled)
        models["XGBoost"].predictions.append(pred_xgb)
        models["XGBoost"].actuals.append(y_test)
        models["XGBoost"].dates.append(test_date)

        # --- LSTM ---
        if HAS_TORCH and "LSTM" in models:
            pred_lstm = lstm_forecast(X_train_scaled, y_train, X_test_scaled)
            models["LSTM"].predictions.append(pred_lstm)
            models["LSTM"].actuals.append(y_test)
            models["LSTM"].dates.append(test_date)

        # --- GRU ---
        if HAS_TORCH and "GRU" in models:
            pred_gru = gru_forecast(X_train_scaled, y_train, X_test_scaled)
            models["GRU"].predictions.append(pred_gru)
            models["GRU"].actuals.append(y_test)
            models["GRU"].dates.append(test_date)

    # Compute metrics
    for name, res in models.items():
        preds = np.array(res.predictions, dtype=float)
        acts = np.array(res.actuals, dtype=float)

        # Remove NaN predictions
        valid = ~np.isnan(preds) & ~np.isnan(acts)
        if valid.sum() < 2:
            continue

        preds_v = preds[valid]
        acts_v = acts[valid]

        res.rmse = np.sqrt(mean_squared_error(acts_v, preds_v))
        res.mae = mean_absolute_error(acts_v, preds_v)

        # Direction accuracy: did the model predict the correct sign of change?
        if len(acts_v) > 1:
            actual_dir = np.diff(acts_v) > 0
            pred_dir = np.diff(preds_v) > 0
            res.direction_accuracy = np.mean(actual_dir == pred_dir) * 100

    return models


# ---------------------------------------------------------------------------
# Diebold-Mariano Test
# ---------------------------------------------------------------------------

def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano test for equal predictive accuracy (two-sided).

    H0: Both forecasts have equal accuracy
    H1: The two forecasts have different accuracy

    Parameters:
    -----------
    e1 : array, forecast errors from model 1
    e2 : array, forecast errors from model 2
    h : int, forecast horizon

    Returns:
    --------
    DM statistic and two-sided p-value
    """
    from scipy import stats

    d = e1 ** 2 - e2 ** 2  # Loss differential (squared errors)
    n = len(d)

    if n < 5:
        return np.nan, np.nan

    d_mean = np.mean(d)

    # Newey-West type variance estimator
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = gamma_0

    for k in range(1, h):
        if k < n:
            gamma_k = np.cov(d[k:], d[:-k])[0, 1]
            gamma_sum += 2 * gamma_k

    var_d = gamma_sum / n

    if var_d <= 0:
        return np.nan, np.nan

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))

    return dm_stat, p_value


# ---------------------------------------------------------------------------
# Bootstrap Confidence Intervals (Künsch, 1989 block bootstrap)
# ---------------------------------------------------------------------------

def block_bootstrap_rmse_ci(models, benchmark="AR(1)", n_boot=1000,
                             block_length=4, alpha=0.05, seed=42):
    """
    Compute block bootstrap confidence intervals for RMSE improvements
    vs. a benchmark model (Künsch, 1989).

    Parameters:
    -----------
    models : dict of ForecastResult objects
    benchmark : str, name of the benchmark model
    n_boot : int, number of bootstrap replications
    block_length : int, block length for temporal dependence
    alpha : float, significance level (two-sided)
    seed : int, random seed

    Returns:
    --------
    DataFrame with model, pct_improvement, ci_lower, ci_upper
    """
    rng = np.random.RandomState(seed)

    bench_res = models.get(benchmark)
    if bench_res is None or bench_res.rmse is None:
        return pd.DataFrame()

    bench_errors = np.array(bench_res.actuals) - np.array(bench_res.predictions)
    bench_valid = ~np.isnan(bench_errors)
    n = bench_valid.sum()

    if n < block_length * 2:
        return pd.DataFrame()

    rows = []
    for name, res in models.items():
        if name == benchmark or res.rmse is None:
            continue

        model_errors = np.array(res.actuals) - np.array(res.predictions)
        model_valid = ~np.isnan(model_errors)
        both_valid = bench_valid & model_valid

        e_bench = bench_errors[both_valid]
        e_model = model_errors[both_valid]
        n_valid = len(e_bench)

        if n_valid < block_length * 2:
            continue

        # Block bootstrap
        n_blocks = int(np.ceil(n_valid / block_length))
        boot_ratios = []

        for _ in range(n_boot):
            # Sample block starting indices
            starts = rng.randint(0, n_valid - block_length + 1, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n_valid]

            boot_bench = e_bench[idx]
            boot_model = e_model[idx]

            rmse_bench = np.sqrt(np.mean(boot_bench ** 2))
            rmse_model = np.sqrt(np.mean(boot_model ** 2))

            if rmse_bench > 0:
                boot_ratios.append((rmse_model - rmse_bench) / rmse_bench * 100)

        if boot_ratios:
            boot_ratios = np.array(boot_ratios)
            ci_lo = np.percentile(boot_ratios, 100 * alpha / 2)
            ci_hi = np.percentile(boot_ratios, 100 * (1 - alpha / 2))
            pct_imp = (res.rmse - bench_res.rmse) / bench_res.rmse * 100

            rows.append({
                "model": name,
                "pct_improvement": round(pct_imp, 1),
                "ci_lower": round(ci_lo, 1),
                "ci_upper": round(ci_hi, 1),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Conformal Prediction Intervals (Chernozhukov et al. 2021 / Lei et al. 2018)
# ---------------------------------------------------------------------------

def conformal_prediction_intervals(models, alpha=0.10):
    """
    Split-conformal prediction intervals for each model.

    Uses the first 50 % of OOS residuals as calibration set to compute
    the conformity quantile, then reports coverage and interval width
    on the remaining 50 %.

    Parameters:
    -----------
    models : dict of ForecastResult objects
    alpha  : float, miscoverage level (default 0.10 → 90 % intervals)

    Returns:
    --------
    DataFrame with model, nominal_coverage, empirical_coverage,
    mean_width, median_width
    """
    rows = []
    for name, res in models.items():
        preds = np.array(res.predictions, dtype=float)
        acts = np.array(res.actuals, dtype=float)
        valid = ~np.isnan(preds) & ~np.isnan(acts)
        preds_v = preds[valid]
        acts_v = acts[valid]
        n = len(preds_v)
        if n < 10:
            continue

        # Split: first half = calibration, second half = evaluation
        n_cal = n // 2
        residuals_cal = np.abs(acts_v[:n_cal] - preds_v[:n_cal])
        # Conformal quantile: ceil((n_cal+1)(1-alpha))/n_cal percentile
        q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
        q_hat = np.quantile(residuals_cal, q_level)

        # Evaluation on second half
        preds_eval = preds_v[n_cal:]
        acts_eval = acts_v[n_cal:]
        lower = preds_eval - q_hat
        upper = preds_eval + q_hat
        covered = (acts_eval >= lower) & (acts_eval <= upper)

        rows.append({
            "model": name,
            "nominal_coverage": round(1 - alpha, 2),
            "empirical_coverage": round(np.mean(covered), 3),
            "mean_width": round(2 * q_hat, 1),
            "median_width": round(2 * q_hat, 1),
            "n_cal": n_cal,
            "n_eval": len(acts_eval),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Clark-West (2007) Test for Nested Models
# ---------------------------------------------------------------------------

def clark_west_test(e1, e2, y_actual, y_pred_restricted):
    """
    Clark & West (2007) test for comparing nested forecasting models.

    Adjusts the MSPE comparison for the noise in the larger model's
    parameter estimates under the null that the restrictions are true.

    Parameters:
        e1: forecast errors from the unrestricted (larger) model
        e2: forecast errors from the restricted (smaller/benchmark) model
        y_actual: actual values
        y_pred_restricted: predictions from the restricted model

    Returns:
        CW test statistic, p-value (one-sided)
    """
    from scipy import stats

    n = len(e1)
    if n < 10:
        return np.nan, np.nan

    # CW adjustment: f_t = e_restricted^2 - e_unrestricted^2 + (ŷ_restricted - ŷ_unrestricted)^2
    # Here e1=unrestricted, e2=restricted, so (ŷ_restricted - ŷ_unrestricted) = e1 - e2
    f_t = e2 ** 2 - e1 ** 2 + (e2 - e1) ** 2

    f_mean = np.mean(f_t)
    f_var = np.var(f_t, ddof=1) / n

    if f_var <= 0:
        return np.nan, np.nan

    cw_stat = f_mean / np.sqrt(f_var)
    p_value = 1 - stats.norm.cdf(cw_stat)  # one-sided

    return cw_stat, p_value


# ---------------------------------------------------------------------------
# R² Out-of-Sample (Campbell & Thompson, 2008)
# ---------------------------------------------------------------------------

def r2_oos(actuals, predictions, benchmark_predictions):
    """
    Out-of-sample R² relative to a benchmark (Campbell & Thompson 2008).

    R²_OOS = 1 - Σ(y - ŷ_model)² / Σ(y - ŷ_bench)²

    Positive values mean the model beats the benchmark.
    """
    a = np.array(actuals, dtype=float)
    p = np.array(predictions, dtype=float)
    b = np.array(benchmark_predictions, dtype=float)

    valid = ~np.isnan(a) & ~np.isnan(p) & ~np.isnan(b)
    if valid.sum() < 2:
        return np.nan

    sse_model = np.sum((a[valid] - p[valid]) ** 2)
    sse_bench = np.sum((a[valid] - b[valid]) ** 2)

    if sse_bench == 0:
        return np.nan

    return 1 - sse_model / sse_bench


# ---------------------------------------------------------------------------
# Forecast Combination
# ---------------------------------------------------------------------------

def forecast_combination(models, method="inverse_rmse", exclude=None):
    """
    Combine forecasts from multiple models.

    Parameters:
        models: dict of ForecastResult objects
        method: 'equal' or 'inverse_rmse'
        exclude: list of model names to exclude (e.g., benchmark)

    Returns:
        ForecastResult with combined predictions
    """
    if exclude is None:
        exclude = []

    eligible = {k: v for k, v in models.items()
                if k not in exclude and v.rmse is not None and len(v.predictions) > 0}

    if not eligible:
        return ForecastResult("Combination")

    # Align predictions by index (all should have same length from expanding window)
    n = min(len(v.predictions) for v in eligible.values())
    names = list(eligible.keys())

    combo = ForecastResult("Combination")
    for i in range(n):
        if method == "equal":
            weights_i = {k: 1.0 / len(names) for k in names}
        elif method == "inverse_rmse" and i >= 1:
            # Expanding weights: only use RMSE accumulated through t-1
            inv_rmse_i = {}
            for k, v in eligible.items():
                past_preds = np.array(v.predictions[:i], dtype=float)
                past_acts = np.array(v.actuals[:i], dtype=float)
                valid = ~np.isnan(past_preds) & ~np.isnan(past_acts)
                if valid.sum() >= 1:
                    rmse_k = np.sqrt(np.mean((past_acts[valid] - past_preds[valid]) ** 2))
                    inv_rmse_i[k] = 1.0 / max(rmse_k, 1e-12)
            if inv_rmse_i:
                total = sum(inv_rmse_i.values())
                weights_i = {k: inv_rmse_i.get(k, 0) / total for k in names}
            else:
                weights_i = {k: 1.0 / len(names) for k in names}
        else:
            weights_i = {k: 1.0 / len(names) for k in names}

        pred = sum(weights_i[k] * float(eligible[k].predictions[i]) for k in names
                   if not np.isnan(float(eligible[k].predictions[i])))
        combo.predictions.append(pred)
        combo.actuals.append(eligible[names[0]].actuals[i])
        combo.dates.append(eligible[names[0]].dates[i])

    preds = np.array(combo.predictions, dtype=float)
    acts = np.array(combo.actuals, dtype=float)
    valid = ~np.isnan(preds) & ~np.isnan(acts)
    if valid.sum() >= 2:
        combo.rmse = np.sqrt(mean_squared_error(acts[valid], preds[valid]))
        combo.mae = mean_absolute_error(acts[valid], preds[valid])
        if valid.sum() > 1:
            actual_dir = np.diff(acts[valid]) > 0
            pred_dir = np.diff(preds[valid]) > 0
            combo.direction_accuracy = np.mean(actual_dir == pred_dir) * 100

    return combo


# ---------------------------------------------------------------------------
# Model Confidence Set (Hansen, Lunde & Nason, 2011)
# ---------------------------------------------------------------------------

def model_confidence_set(models, alpha=0.10, n_boot=1000, block_length=4, seed=42):
    """
    Model Confidence Set (Hansen, Lunde & Nason 2011).

    Iteratively eliminates the worst model until the null of equal
    predictive ability cannot be rejected at level alpha.

    Returns:
        list of model names in the superior set, list of eliminated models
    """
    rng = np.random.RandomState(seed)

    eligible = {k: v for k, v in models.items()
                if v.rmse is not None and len(v.predictions) > 0}
    if len(eligible) < 2:
        return list(eligible.keys()), []

    # Align all predictions
    n = min(len(v.predictions) for v in eligible.values())
    names = list(eligible.keys())

    # Build loss matrix (squared errors)
    losses = {}
    for k, v in eligible.items():
        preds = np.array(v.predictions[:n], dtype=float)
        acts = np.array(v.actuals[:n], dtype=float)
        losses[k] = (acts - preds) ** 2

    eliminated = []
    surviving = list(names)

    while len(surviving) > 1:
        m = len(surviving)
        # Pairwise loss differentials
        d_bar = np.zeros((m, m))
        d_series = {}
        for i in range(m):
            for j in range(i + 1, m):
                d_ij = losses[surviving[i]] - losses[surviving[j]]
                valid = ~np.isnan(d_ij)
                if valid.sum() < 10:
                    continue
                d_clean = d_ij[valid]
                d_bar[i, j] = np.mean(d_clean)
                d_bar[j, i] = -d_bar[i, j]
                d_series[(i, j)] = d_clean

        # T_R statistic: max of standardized pairwise means
        t_stats = []
        for i in range(m):
            for j in range(i + 1, m):
                if (i, j) not in d_series:
                    continue
                d_clean = d_series[(i, j)]
                se = np.std(d_clean, ddof=1) / np.sqrt(len(d_clean))
                if se > 1e-10:
                    t_stats.append(abs(np.mean(d_clean)) / se)

        if not t_stats:
            break

        T_R = max(t_stats)

        # Bootstrap distribution of T_R under the null
        n_valid = min(len(d) for d in d_series.values()) if d_series else n
        n_blocks = int(np.ceil(n_valid / block_length))
        boot_T = []

        for _ in range(n_boot):
            starts = rng.randint(0, max(1, n_valid - block_length + 1), size=n_blocks)
            idx = np.concatenate([np.arange(s, min(s + block_length, n_valid))
                                  for s in starts])[:n_valid]

            boot_t_stats = []
            for i in range(m):
                for j in range(i + 1, m):
                    if (i, j) not in d_series:
                        continue
                    d_clean = d_series[(i, j)]
                    d_boot = d_clean[idx[:len(d_clean)]] if len(idx) >= len(d_clean) else d_clean
                    d_centered = d_boot - np.mean(d_clean)  # center under null
                    se = np.std(d_centered, ddof=1) / np.sqrt(len(d_centered))
                    if se > 1e-10:
                        boot_t_stats.append(abs(np.mean(d_centered)) / se)

            if boot_t_stats:
                boot_T.append(max(boot_t_stats))

        if not boot_T:
            break

        p_value = np.mean(np.array(boot_T) >= T_R)

        if p_value < alpha:
            # Reject null: eliminate worst model (highest mean loss)
            mean_losses = {surviving[i]: np.nanmean(losses[surviving[i]])
                           for i in range(m)}
            worst = max(mean_losses, key=mean_losses.get)
            eliminated.append(worst)
            surviving.remove(worst)
        else:
            break

    return surviving, eliminated


# ---------------------------------------------------------------------------
# Results Summary
# ---------------------------------------------------------------------------

def print_results(models, benchmark="AR(1)"):
    """
    Print a summary table of model performance.
    """
    print("\n" + "=" * 70)
    print("  RESULTS: OUT-OF-SAMPLE FORECAST ACCURACY")
    print("=" * 70)

    print(f"\n  {'Model':<20} {'RMSE':>10} {'MAE':>10} {'Dir. Acc.':>10} {'vs AR(1)':>10}")
    print("  " + "-" * 60)

    benchmark_rmse = models.get(benchmark, ForecastResult(benchmark)).rmse

    for name, res in models.items():
        rmse_str = f"{res.rmse:.2f}" if res.rmse else "N/A"
        mae_str = f"{res.mae:.2f}" if res.mae else "N/A"
        dir_str = f"{res.direction_accuracy:.1f}%" if res.direction_accuracy else "N/A"

        # Relative RMSE vs benchmark
        if res.rmse and benchmark_rmse:
            rel = ((res.rmse - benchmark_rmse) / benchmark_rmse) * 100
            rel_str = f"{rel:+.1f}%"
        else:
            rel_str = "—"

        print(f"  {name:<20} {rmse_str:>10} {mae_str:>10} {dir_str:>10} {rel_str:>10}")

    # Diebold-Mariano tests vs AR(1)
    print(f"\n  Diebold-Mariano Test vs {benchmark}:")
    print(f"  {'Model':<20} {'DM Stat':>10} {'p-value':>10} {'Significant':>12}")
    print("  " + "-" * 52)

    bench_res = models.get(benchmark)
    if bench_res:
        bench_errors = np.array(bench_res.actuals) - np.array(bench_res.predictions)
        bench_valid = ~np.isnan(bench_errors)

        for name, res in models.items():
            if name == benchmark:
                continue

            model_errors = np.array(res.actuals) - np.array(res.predictions)
            model_valid = ~np.isnan(model_errors)

            # Align valid observations
            both_valid = bench_valid & model_valid
            if both_valid.sum() < 5:
                continue

            dm_stat, p_val = diebold_mariano_test(
                model_errors[both_valid],
                bench_errors[both_valid]
            )

            if not np.isnan(dm_stat):
                sig = "Yes *" if p_val < 0.10 else ("Yes **" if p_val < 0.05 else "No")
                print(f"  {name:<20} {dm_stat:>10.3f} {p_val:>10.4f} {sig:>12}")

    print("\n" + "=" * 70)


def save_results(models):
    """Save forecast results to CSV."""
    rows = []
    for name, res in models.items():
        for i in range(len(res.predictions)):
            rows.append({
                "model": name,
                "date": res.dates[i],
                "actual": res.actuals[i],
                "predicted": res.predictions[i],
            })

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "forecast_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nForecast results saved to {out_path}")

    # Summary metrics
    summary_rows = []
    for name, res in models.items():
        summary_rows.append({
            "model": name,
            "rmse": res.rmse,
            "mae": res.mae,
            "direction_accuracy": res.direction_accuracy,
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "model_comparison.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Model comparison saved to {summary_path}")


# ---------------------------------------------------------------------------
# SHAP Feature Importance
# ---------------------------------------------------------------------------

def shap_feature_importance(df, target_col, feature_cols, min_train_size=40):
    """
    Compute SHAP values for XGBoost trained on the full training window.
    Saves a bar plot (mean |SHAP|) and a beeswarm plot to outputs/.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols_needed = [target_col] + feature_cols
    df_clean = df[["date"] + cols_needed].dropna().reset_index(drop=True)

    y = df_clean[target_col].values
    X = df_clean[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train XGBoost on the full dataset (for explanation, not forecasting)
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0,
    )
    model.fit(X_scaled, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Mean |SHAP| bar chart
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.3)))
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_n = min(15, len(feature_cols))
    top_idx = order[:top_n]

    ax.barh(range(top_n), mean_abs[top_idx][::-1],
            color="#E91E63", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_cols[i] for i in top_idx][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("XGBoost Feature Importance (SHAP)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved shap_importance.png")

    # Beeswarm / summary plot
    fig, ax = plt.subplots(figsize=(10, max(4, len(feature_cols) * 0.3)))
    shap.summary_plot(shap_values, X_scaled, feature_names=feature_cols,
                      show=False, max_display=top_n)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved shap_summary.png")

    # Print top features
    print(f"\n  Top {top_n} features by mean |SHAP value|:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    {rank:>2}. {feature_cols[idx]:<25} {mean_abs[idx]:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full pilot study pipeline."""
    print("=" * 70)
    print("  BOP NOWCASTING PILOT STUDY")
    print("  AI vs. Econometric Baselines — Expanding Window Evaluation")
    print("=" * 70)

    # Load data
    df = load_modeling_data()
    print(f"\nLoaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Period: {df['date'].min()} to {df['date'].max()}")

    # Identify target and features
    # Target: BoP current account (or trade balance)
    # Strategy: use the first BoP-related column as target
    potential_targets = [c for c in df.columns
                        if "bop" in c.lower() and "yoy" not in c and "qoq" not in c]

    if not potential_targets:
        # Fallback: use the first numeric column that's not a date
        potential_targets = [c for c in df.columns
                            if c != "date" and df[c].dtype in [np.float64, np.int64]
                            and "yoy" not in c and "qoq" not in c]

    if not potential_targets:
        print("[ERROR] No suitable target variable found.")
        return

    target_col = potential_targets[0]

    # Features: all other numeric columns EXCEPT other BoP components
    # AND contemporaneous target transformations to prevent data leakage.
    # bop_ca_qoq = bop_ca[t] - bop_ca[t-1] contains the target at time t.
    bop_leakage = ["bop_goods", "bop_services"]
    target_contemp = [f"{target_col}_yoy", f"{target_col}_qoq"]
    feature_cols = [c for c in df.columns
                    if c != "date" and c != target_col
                    and c not in target_contemp
                    and df[c].dtype in [np.float64, np.int64, float]
                    and not any(c.startswith(prefix) for prefix in bop_leakage)]

    # Also add lagged target as a feature (if not already present)
    for lag_name, lag_val in [(f"{target_col}_lag1", 1), (f"{target_col}_lag4", 4)]:
        if lag_name not in df.columns:
            df[lag_name] = df[target_col].shift(lag_val)
        if lag_name not in feature_cols:
            feature_cols.append(lag_name)

    print(f"\n   Target variable: {target_col}")
    print(f"   Feature variables: {len(feature_cols)}")
    for f in feature_cols[:10]:
        print(f"     - {f}")
    if len(feature_cols) > 10:
        print(f"     ... and {len(feature_cols) - 10} more")

    # Run expanding window evaluation
    models = expanding_window_evaluation(
        df, target_col, feature_cols,
        min_train_size=40,
        step=1
    )

    if not models:
        print("[ERROR] Evaluation failed. Not enough data.")
        return

    # Print results
    print_results(models)

    # Save results
    save_results(models)

    # --- Bootstrap Confidence Intervals ---
    print("\n" + "=" * 70)
    print("  BOOTSTRAP CONFIDENCE INTERVALS (Block Bootstrap, B=1000)")
    print("=" * 70)
    try:
        boot_df = block_bootstrap_rmse_ci(models, benchmark="AR(1)",
                                           n_boot=1000, block_length=4)
        if not boot_df.empty:
            print(f"\n  {'Model':<20} {'% vs AR(1)':>12} {'95% CI':>20}")
            print("  " + "-" * 54)
            for _, row in boot_df.iterrows():
                ci_str = f"[{row['ci_lower']:+.1f}, {row['ci_upper']:+.1f}]"
                print(f"  {row['model']:<20} {row['pct_improvement']:>+10.1f}% {ci_str:>20}")
            boot_df.to_csv(OUTPUT_DIR / "bootstrap_ci.csv", index=False)
            print(f"\n  Saved bootstrap_ci.csv")
    except Exception as e:
        print(f"  [WARN] Bootstrap CI failed: {e}")

    # --- SHAP Feature Importance ---
    if HAS_SHAP:
        print("\n" + "=" * 70)
        print("  SHAP FEATURE IMPORTANCE (XGBoost)")
        print("=" * 70)
        try:
            shap_feature_importance(df, target_col, feature_cols)
        except Exception as e:
            print(f"  [WARN] SHAP analysis failed: {e}")
    else:
        print("\n  [INFO] Install 'shap' for feature importance analysis.")

    print("\n[OK] Pilot study complete!")


if __name__ == "__main__":
    main()
