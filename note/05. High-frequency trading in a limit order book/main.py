"""
"High-frequency trading in a limit order book" - Avellaneda and Stoikov, 2006
"""
import pandas as pd
import numpy as np
import price_series_generator
import plot_utils
from model import MarketImpactModel
from strategy import InventoryStrategy, SymmetricStrategy

if __name__ == "__main__":
    PATHS = 1000
    START_PRICE = 100
    SIGMA = 2
    STEPS = 200
    dt = (1 / STEPS)
    GAMMA = 0.1
    K = 1.5
    A = 140

    optimal_spread = 2 * np.log(1 + GAMMA / K) / GAMMA
    spread = optimal_spread / 2

    price_array = price_series_generator.generate_price_series(
        initial_value=START_PRICE,
        sigma=SIGMA,
        paths=PATHS,
        steps=STEPS,
    )

    market_impact_model = MarketImpactModel(A=A, K=K, dt=dt)

    optimal_bid_ask_price_df = pd.DataFrame(columns=['bid', 'ask', 'price'], dtype=float)
    final_profit_df = pd.DataFrame(columns=['inventory', 'symmetric'], dtype=float)
    for path_i, single_price_series in enumerate(price_array):
        inventory_strategy = InventoryStrategy(spread, market_impact_model, gamma=GAMMA, sigma=SIGMA, steps=STEPS)
        symmetric_strategy = SymmetricStrategy(spread, market_impact_model)
        final_price = 0
        for step_i, price in enumerate(single_price_series):
            # Inventory Strategy ===========================================
            opt_bid, opt_ask = inventory_strategy.step(price, step_i)
            # ==============================================================

            # Symmetric Strategy ===========================================
            symmetric_strategy.step(price)
            # ==============================================================

            if path_i == 0:
                optimal_bid_ask_price_df.loc[len(optimal_bid_ask_price_df)] = [opt_bid, opt_ask, price]
            final_price = price

        inventory_strategy_profit = inventory_strategy.get_final_profit(final_price)
        symmetric_strategy_profit = symmetric_strategy.get_final_profit(final_price)
        final_profit_df.loc[len(final_profit_df)] = [inventory_strategy_profit, symmetric_strategy_profit]

    plot_utils.plot_compare_price(optimal_bid_ask_price_df)    # 첫번째만 출력
    plot_utils.plot_compare_histogram(final_profit_df)
