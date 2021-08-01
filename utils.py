import pandas as pd
import numpy as np


def get_portfolio_volatility(weights: pd.Series or np.ndarray, covariance_matrix: pd.DataFrame) -> float:
    assert len(weights) == len(covariance_matrix)
    assert sum(weights) == 1

    variance = weights.dot(covariance_matrix).dot(weights)
    std = np.sqrt(variance)
    return std


def get_portfolio_volatility_delta(
        prev_weights: pd.Series or np.ndarray,
        after_weights: pd.Series or np.ndarray,
        covariance_matrix: pd.DataFrame,
) -> float:
    assert len(prev_weights) == len(after_weights) == len(covariance_matrix)

    prev_vol = get_portfolio_volatility(prev_weights, covariance_matrix)
    after_vol = get_portfolio_volatility(after_weights, covariance_matrix)
    volatility_delta = after_vol - prev_vol
    return volatility_delta


def generate_multivariate(returns_df: pd.DataFrame, counts: int) -> pd.DataFrame:
    assets_counts = len(returns_df.columns)

    cov = returns_df.cov()
    returns_mean = returns_df.mean()

    Z = np.random.normal(size=(assets_counts, counts))
    L = np.linalg.cholesky(cov)

    generated_returns = np.full((counts, assets_counts), returns_mean).T + np.dot(L, Z)
    return pd.DataFrame(generated_returns.T, columns=returns_df.columns)


def generate_correlated_returns(returns_mean: pd.Series,
                                rho: float,
                                counts: int,
                                returns_std: pd.Series = 1) -> pd.DataFrame:
    assert len(returns_mean) == 2
    assets_counts = len(returns_mean)

    Z = np.random.normal(size=(2, counts))
    cov = np.array([
        [1, rho],
        [rho, 1],
    ])
    L = np.linalg.cholesky(cov)

    generated_returns = np.full((counts, assets_counts), returns_mean) + pd.DataFrame(np.dot(L, Z)).T * returns_std
    return pd.DataFrame(generated_returns)


def generate_multiple_correlated_returns(returns_mean: pd.Series,
                                         rho_matrix: np.ndarray,
                                         counts: int,
                                         returns_std: pd.Series = 1) -> pd.DataFrame:
    assert np.array_equal(rho_matrix, rho_matrix.T)
    assets_counts = len(returns_mean)

    Z = np.random.normal(size=(assets_counts, counts))
    L = np.linalg.cholesky(rho_matrix)

    generated_returns = np.full((counts, assets_counts), returns_mean) + pd.DataFrame(np.dot(L, Z)).T * returns_std
    return pd.DataFrame(generated_returns)


if __name__ == "__main__":
    close_price_df = pd.read_csv("./data/sample_stock_close_price.csv", index_col=0, parse_dates=True)
    close_price_df.columns = ['Samsung Electronics', 'SK Hynix', 'KAKAO', 'NAVER', 'KODEX Inverse']

    port_weights = np.array([0.2, 0.5, 0.3, 0, 0])
    port_weights_2 = np.array([0.1, 0.2, 0.1, 0.3, 0.3])
    stock_covariance = close_price_df.pct_change().cov()
    print(stock_covariance)

    portfolio_volatility = get_portfolio_volatility(port_weights, stock_covariance)
    print(portfolio_volatility)

    vol_delta = get_portfolio_volatility_delta(port_weights, port_weights_2, stock_covariance)
    print(vol_delta)

    port_weights_3 = np.array([0, 0, 0, 0, 1])
    portfolio_volatility = get_portfolio_volatility(port_weights_3, stock_covariance)
    print(portfolio_volatility)

    print(close_price_df.pct_change().corr())

    res = generate_multivariate(close_price_df.pct_change(), 1000)
    print(res.corr())

    res = generate_correlated_returns(pd.Series([0.01, 0.02]), 0.7, 1000)
    print(res.corr())

    res = generate_multiple_correlated_returns(pd.Series([0, 0, 0, 0, 0]), rho_matrix=np.array([
        [1.000000, 0.507238, 0.149265, 0.212039, -0.732306],
        [0.507238, 1.000000, 0.177784, 0.162638, -0.571248],
        [0.149265, 0.177784, 1.000000, 0.319981, -0.294286],
        [0.212039, 0.162638, 0.319981, 1.000000, -0.338091],
        [-0.732306, -0.571248, -0.294286, -0.338091, 1.000000],
    ]), counts=10000, returns_std=pd.Series([0.017, 0.024, 0.035, 0.02, 0.02]))
    print(res.describe())
    print(res.corr())
