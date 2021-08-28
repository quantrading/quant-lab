from model import MarketImpactModel, MarketMaker, get_reservation_price


class InventoryStrategy:
    def __init__(self, spread: float, market_impact_model: MarketImpactModel, gamma: float, sigma: float, steps: int):
        self.market_maker = MarketMaker()
        self.spread = spread
        self.market_impact_model = market_impact_model
        self.gamma = gamma
        self.sigma = sigma
        self.steps = steps

    def step(self, price: float, step_i: int) -> tuple[float, float]:
        reservation_price = get_reservation_price(
            price=price,
            inventory=self.get_inventory(),
            gamma=self.gamma,
            sigma=self.sigma,
            time_to_end=1 - step_i / self.steps
        )
        bid_spread = self.spread - (reservation_price - price)
        ask_spread = self.spread + (reservation_price - price)

        bid_touch, ask_touch = self.market_impact_model.bid_ask_spread_touch(bid_spread, ask_spread)

        if bid_touch:
            self.market_maker.bid_touch(price, bid_spread)
        if ask_touch:
            self.market_maker.ask_touch(price, ask_spread)

        bid_price = reservation_price - self.spread
        ask_price = reservation_price + self.spread
        return bid_price, ask_price

    def get_inventory(self) -> float:
        return self.market_maker.get_inventory()

    def get_final_profit(self, price: float) -> float:
        return self.market_maker.get_final_profit(price)


class SymmetricStrategy:
    def __init__(self, spread: float, market_impact_model: MarketImpactModel):
        self.market_maker = MarketMaker()
        self.spread = spread
        self.market_impact_model = market_impact_model

    def step(self, price: float):
        bid_touch, ask_touch = self.market_impact_model.bid_ask_spread_touch(self.spread, self.spread)

        if bid_touch:
            self.market_maker.bid_touch(price, self.spread)
        if ask_touch:
            self.market_maker.ask_touch(price, self.spread)

    def get_final_profit(self, price: float) -> float:
        return self.market_maker.get_final_profit(price)
