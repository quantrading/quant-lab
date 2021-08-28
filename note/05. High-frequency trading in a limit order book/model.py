import numpy as np


def get_reservation_price(
        price: float,
        inventory: float,
        gamma: float,
        sigma: float,
        time_to_end: float
) -> float:
    reservation_price = price - inventory * gamma * (sigma ** 2) * time_to_end
    return reservation_price


class MarketMaker:
    def __init__(self):
        self.cash = 0
        self.__inventory = 0

    def get_inventory(self):
        return self.__inventory

    def add_inventory(self):
        self.__inventory += 1

    def subtract_inventory(self):
        self.__inventory -= 1

    def ask_touch(self, price: float, spread: float):
        self.subtract_inventory()
        self.cash += (price + spread)

    def bid_touch(self, price: float, spread: float):
        self.add_inventory()
        self.cash -= (price - spread)

    def get_final_profit(self, final_price) -> float:
        profit = self.cash
        profit += self.__inventory * final_price
        return profit


class MarketImpactModel:
    def __init__(self, A: float, K: float, dt: float):
        self.K = K
        self.A = A
        self.dt = dt

    def bid_ask_spread_touch(self, bid_spread: float, ask_spread: float) -> tuple[bool, bool]:
        bid_touch = True if bid_spread < 0 else self.__spread_touch(bid_spread)
        ask_touch = True if ask_spread < 0 else self.__spread_touch(ask_spread)
        return bid_touch, ask_touch

    def __spread_touch(self, spread: float) -> bool:
        touch_prob = self.A * np.exp(-self.K * spread) * self.dt
        touch_prob = max(0, min(touch_prob, 1))
        touch = np.random.choice([True, False], p=[touch_prob, 1 - touch_prob])
        return touch
