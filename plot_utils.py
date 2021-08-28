import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(data):
    sns.displot(data=data, kde=True)
    plt.show()


def plot_price(data):
    plt.plot(data)
    plt.show()


def plot_compare_histogram(df: pd.DataFrame):
    sns.displot(df)
    plt.show()


def plot_compare_price(df: pd.DataFrame):
    df.plot()
    plt.show()


def plot_final_price_distribution(data: np.ndarray):
    plot_distribution(data)


def plot_multi_price_paths(data: np.ndarray):
    plot_price(data)
