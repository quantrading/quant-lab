import numpy as np


def generate_binary_white_noise(sigma: float, steps: int, paths=1):
    white_noise = sigma * np.random.choice([1, -1], (paths, steps))
    return white_noise


def generate_price_series(
        initial_value: float,
        sigma: float,
        paths: int,
        steps: int,
) -> np.ndarray:
    dt = 1 / steps

    white_noise = generate_binary_white_noise(sigma=sigma * np.sqrt(dt), steps=steps, paths=paths)
    price_array = initial_value + np.cumsum(white_noise, axis=1)
    price_array = np.insert(price_array, 0, initial_value, axis=1)

    return price_array
