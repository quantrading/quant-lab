"""
Likelihood Grid Search 를 통한 정규분포 모수 추정 (MLE)
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(1 / 2) * ((x - mu) / sigma) ** 2)


def get_likelihood_by_product(x_array: np.ndarray, mu: float, sigma: float) -> float:
    probability_list = []
    for x in x_array:
        probability = normal_pdf(x, mu, sigma)
        probability_list.append(probability)
    likelihood = np.product(probability_list)
    return likelihood


def get_likelihood_by_log_sum(x_array: np.ndarray, mu: float, sigma: float) -> float:
    log_probability_list = []
    for x in x_array:
        probability = normal_pdf(x, mu, sigma)
        log_probability_list.append(np.log(probability))
    likelihood = np.sum(log_probability_list)
    return likelihood


if __name__ == "__main__":
    p_mu = 50
    p_sigma = 13
    normal_dist_samples = np.random.normal(p_mu, p_sigma, 1000)

    # Parameter setting for Grid Search
    min_mu, max_mu = 40, 60
    min_sigma, max_sigma = 5, 20

    mu_list = np.unique(np.linspace(min_mu, max_mu, 10).round(2))
    sigma_list = np.unique(np.linspace(min_sigma, max_sigma, 10).round(1))

    likelihood_df = pd.DataFrame(columns=['mu', 'sigma', 'likelihood'])
    for mu in mu_list:
        for sigma in sigma_list:
            likelihood = get_likelihood_by_log_sum(normal_dist_samples, mu, sigma)
            likelihood_df.loc[len(likelihood_df)] = [mu, sigma, likelihood]

    likelihood_df = likelihood_df.replace([-np.inf, np.inf], np.nan)

    mle = likelihood_df.loc[likelihood_df['likelihood'].argmax()]
    print(mle)

    likelihood_df['normalized_likelihood'] = (likelihood_df['likelihood'] - likelihood_df['likelihood'].mean()) / likelihood_df['likelihood'].std()

    heatmap_df = likelihood_df.pivot('mu', 'sigma', 'normalized_likelihood')

    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(heatmap_df, annot=True, linewidths=.5, ax=ax)
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(likelihood_df['mu'], likelihood_df['sigma'], likelihood_df['normalized_likelihood'], linewidth=0.2)
    plt.show()

    fig = go.Figure(data=[go.Surface(z=heatmap_df.values)])
    fig.show()
