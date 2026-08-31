from dataclasses import dataclass
from typing import Dict

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

@dataclass
class CointegrationResult:
    is_cointegrated: bool
    hedge_ratio: float
    intercept: float
    adf_statistic: float
    adf_pvalue: float
    critical_values: Dict[str, float]
    spread: pd.Series

class EngleGrangerTester:

    def __init__(self, significance: float = 0.05):
        self.significance = significance

    def test(self, series_a: pd.Series, series_b: pd.Series) -> CointegrationResult:
        series_a = series_a.dropna()
        series_b = series_b.dropna()
        series_a, series_b = series_a.align(series_b, join="inner")

        if len(series_a) < 30:
            raise ValueError("There needs to be at least 30 overlapping observations to test cointegration")

        x= sm.add_constant(series_b.values)
        model = sm.OLS(series_a.values, x).fit()
        alpha, beta = model.params[0], model.params[1]
        residuals = model.resid

        adf_stat, p_value, _, _, crit_values, _ = adfuller(residuals, autolag="AIC", result_object=False)
        spread = series_a - beta * series_b - alpha

        return CointegrationResult(is_cointegrated=p_value < self.significance,
                                   hedge_ratio = float(beta),
                                   intercept = float(alpha),
                                   adf_statistic = float(adf_stat),
                                   adf_pvalue = float(p_value),
                                   critical_values = {k: float(v) for k, v in crit_values.items()},
                                   spread = spread,
                                   )

    @staticmethod
    def verify_with_builtin(series_a: pd.Series, series_b: pd.Series) -> dict:
        score, p_value, crit_values = coint(series_a, series_b)
        return {"score": float(score), "p_value": float(p_value), "critical_values": crit_values}

def rolling_hedge_ration(series_a: pd.Series, series_b: pd.Series, window: int = 90) -> pd.Series:
    betas = pd.Series(index=series_a.index, dtype=float)
    for i in range(window, len(series_a)):
        a_win = series_a.iloc[i - window:i]
        b_win = series_b.iloc[i - window:i]
        x = sum.add_constant(b_win.values)
        model = sm.OLS(a_win.values, x).fit()
        betas.iloc[i] = model.params[i]
    return betas