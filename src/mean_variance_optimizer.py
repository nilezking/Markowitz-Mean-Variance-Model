import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import optimize as sc
import seaborn as sns
import matplotlib.pyplot as plt

#plt.style.use('ggplot')
sns.set_style("darkgrid")
plt.rc("figure", figsize=(10, 8))
plt.rc("savefig", dpi=90)

def download(tickers: list, years: int):
    """
    Downloads adjusted closing prices for a list of tickers over a specified time period.

    Parameters:
        tickers (list): List of stock tickers.
        years (int): Number of years of historical data to download.

    Returns:
        pd.DataFrame: DataFrame containing adjusted closing prices of the tickers.
    """
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days= 365 * years)

    data = yf.download(tickers, start= start_date, end= end_date, interval= '1d')['Adj Close']
    return data

def return_statistics(data: pd.DataFrame):
    """
    Calculates log returns, mean returns, covariance matrix, and correlation matrix of the stock data.

    Parameters:
        data (pd.DataFrame): DataFrame containing adjusted closing prices.

    Returns:
        tuple: Mean returns (pd.Series), covariance matrix (pd.DataFrame), correlation matrix (pd.DataFrame).
    """
    log_returns = np.log(data/data.shift(1)).dropna()
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    corr_matrix = log_returns.corr()
    return mean_returns, cov_matrix, corr_matrix

def negative_sharpe(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame, rfr: float, lambda_coeff):
    """
    Computes the negative Sharpe ratio for a given set of weights.

    Parameters:
        weights (np.ndarray): Asset weights in the portfolio.
        mean_returns (pd.Series): Mean returns of assets.
        cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
        rfr (float): Risk-free rate.
        lambda_coeff (float): Regularization parameter for L2 penalty.

    Returns:
        float: Negative Sharpe ratio of the portfolio.
    """
    portfolio_mean_return = annualized_portfolio_return(weights, mean_returns)
    portfolio_stdev = annualized_portfolio_stdev(weights, cov_matrix, lambda_coeff)
    return -(portfolio_mean_return - rfr)/portfolio_stdev

def annualized_portfolio_return(weights: np.ndarray, mean_returns: pd.Series):
    """
    Calculates the annualized portfolio return.

    Parameters:
        weights (np.ndarray): Asset weights in the portfolio.
        mean_returns (pd.Series): Mean returns of assets.

    Returns:
        float: Annualized portfolio return.
    """
    portfolio_mean_return = np.sum(mean_returns*weights) * 252
    return portfolio_mean_return

def annualized_portfolio_stdev(weights: np.ndarray, cov_matrix: pd.DataFrame, lambda_coeff):
    """
    Calculates the annualized portfolio standard deviation with L2 regularization.

    Parameters:
        weights (np.ndarray): Asset weights in the portfolio.
        cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
        lambda_coeff (float): Regularization parameter for L2 penalty.

    Returns:
        float: Annualized portfolio standard deviation.
    """
    portfolio_stdev = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
    l2_penalty = lambda_coeff * np.sum(weights**2)
    return portfolio_stdev + l2_penalty

def max_sharpe(mean_returns: pd.Series, cov_matrix: pd.DataFrame, bound: tuple, rfr: float, method: str, lambda_coeff):
    """
    Optimizes portfolio weights to achieve the maximum Sharpe ratio.

    Parameters:
        mean_returns (pd.Series): Mean returns of assets.
        cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
        bound (tuple): Bounds for weights of assets.
        rfr (float): Risk-free rate.
        method (str): Optimization method.
        lambda_coeff (float): Regularization parameter for L2 penalty.

    Returns:
        tuple: Optimization result, annualized portfolio return, annualized portfolio standard deviation, allocation DataFrame.
    """
    n_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, rfr, lambda_coeff)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(bound for asset in range(n_assets))
    result = sc.minimize(negative_sharpe, n_assets*[1./n_assets], args=args, method=method, bounds=bounds, constraints=constraints)

    max_sharpe_returns = annualized_portfolio_return(result['x'], mean_returns)
    max_sharpe_stdev = annualized_portfolio_stdev(result['x'], cov_matrix, lambda_coeff)
    max_sharpe_allocation = pd.DataFrame(result['x'], index= mean_returns.index, columns= ['ALLOCATION'])
    max_sharpe_allocation.ALLOCATION = [round(i*100, 0) for i in max_sharpe_allocation.ALLOCATION]

    return result, max_sharpe_returns, max_sharpe_stdev, max_sharpe_allocation

def minimum_variance(mean_returns: pd.Series, cov_matrix: pd.DataFrame, bound: tuple, rfr: float, method: str, lambda_coeff):
    """
    Optimizes portfolio weights to minimize portfolio variance.

    Parameters:
        mean_returns (pd.Series): Mean returns of assets.
        cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
        bound (tuple): Bounds for weights of assets.
        rfr (float): Risk-free rate.
        method (str): Optimization method.
        lambda_coeff (float): Regularization parameter for L2 penalty.

    Returns:
        tuple: Optimization result, annualized portfolio return, annualized portfolio standard deviation, allocation DataFrame, Sharpe ratio.
    """
    n_assets = len(mean_returns)
    args = (cov_matrix, lambda_coeff)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(bound for asset in range(n_assets))
    result = sc.minimize(annualized_portfolio_stdev, n_assets*[1./n_assets], args=args, method=method, bounds=bounds, constraints=constraints)

    minimum_variance_returns = annualized_portfolio_return(result['x'], mean_returns)
    minimum_variance_stdev = annualized_portfolio_stdev(result['x'], cov_matrix, lambda_coeff)
    minimum_variance_allocation = pd.DataFrame(result['x'], index= mean_returns.index, columns= ['ALLOCATION'])
    minimum_variance_allocation.ALLOCATION = [round(i*100, 0) for i in minimum_variance_allocation.ALLOCATION]
    minimum_variance_sharpe = (minimum_variance_returns - rfr) / minimum_variance_stdev

    return result, minimum_variance_returns, minimum_variance_stdev, minimum_variance_allocation, minimum_variance_sharpe

def efficient_frontier(mean_returns: pd.Series, cov_matrix: np.ndarray, bound: tuple, mv, ms, method: str, lambda_coeff):
    """
    Computes the efficient frontier by finding optimal portfolios for a range of target returns.

    Parameters:
        mean_returns (pd.Series): Mean returns of assets.
        cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
        bound (tuple): Bounds for weights of assets.
        mv (float): Minimum variance portfolio return.
        ms (float): Maximum Sharpe ratio portfolio return.
        method (str): Optimization method.
        lambda_coeff (float): Regularization parameter for L2 penalty.

    Returns:
        tuple: Efficient frontier standard deviations, target returns, Sharpe ratios, and optimization results.
    """
    return_target = np.linspace(mv, ms, 100)
    efficient = []
    sharpe = []
    results= []
    for target in return_target:
        n_assets = len(mean_returns)
        args = (cov_matrix, lambda_coeff)
        constraints = ({'type':'eq', 'fun': lambda x: annualized_portfolio_return(x, mean_returns) - target}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple(bound for asset in range(n_assets))
        result = sc.minimize(annualized_portfolio_stdev, n_assets*[1./n_assets], args=args, method=method, bounds=bounds, constraints= constraints)
        efficient.append(result.fun)
        sharpe.append((target - 0.03)/result.fun)
        results.append(result)
    return efficient, return_target, sharpe, results

def summary(result_max_sharpe, result_minimum_variance, result_efficient_frontier, minimum_variance_sharpe, 
            minimum_variance_returns, minimum_variance_stdev, max_sharpe_returns, max_sharpe_stdev, method, 
            max_sharpe_allocation, minimum_variance_allocation, stock_data, constaints, lambda_coeff, amount):
    """
        Calculates and prints the results of portfolio optimization, including the maximum Sharpe ratio, minimum volatility, 
        and efficient frontier. It also generates relevant plots such as the efficient frontier and a correlation heatmap.

        Parameters:
            result_max_sharpe (sc.OptimizeResult): Result of the optimization to maximize the Sharpe ratio.
            result_minimum_variance (sc.OptimizeResult): Result of the optimization to minimize portfolio volatility.
            result_efficient_frontier (list): List of optimization results for the efficient frontier.
            minimum_variance_sharpe (float): Sharpe ratio of the minimum variance portfolio.
            minimum_variance_returns (float): Expected annual return of the minimum variance portfolio.
            minimum_variance_stdev (float): Standard deviation (volatility) of the minimum variance portfolio.
            max_sharpe_returns (float): Expected annual return of the maximum Sharpe ratio portfolio.
            max_sharpe_stdev (float): Standard deviation (volatility) of the maximum Sharpe ratio portfolio.
            method (str): Optimization method used (e.g., 'SLSQP').
            max_sharpe_allocation (pd.Series): Allocation of assets in the maximum Sharpe ratio portfolio.
            minimum_variance_allocation (pd.Series): Allocation of assets in the minimum volatility portfolio.
            stock_data (pd.DataFrame): DataFrame containing stock data for the tickers.
            constraints (tuple): Tuple defining weight constraints for the portfolio (e.g., (0, 1)).
            lambda_coeff (float): L2 regularization coefficient.
            amount (float): Total amount of money to be allocated in the portfolio.

        Returns:
            None: This function prints optimization results, allocations, and generates plots.
    """
    ef_vals = []
    ef_nits = 0
    for result in result_efficient_frontier:
        ef_vals.append(result.success)
        ef_nits += result.nit
    status_efficient_frontier = all(i for i in ef_vals)

    if status_efficient_frontier:
        status_efficient_frontier = 'Success'
    else:
        status_efficient_frontier = 'Failure'

    if result_max_sharpe.success:
        status_max_sharpe = 'Success'
    else:
        status_max_sharpe = 'Failure'

    if result_minimum_variance.success:
        status_minimum_variance = 'Success'
    else:
        status_minimum_variance = 'Failure'

    def portfolio_discrete_allocation(allocations, data, amount):
        currency_amount = (allocations / 100 * amount)
        stock_price = data.iloc[-1]
        discrete_allocation = pd.concat([currency_amount, stock_price], axis= 1)
        discrete_allocation = round(discrete_allocation.iloc[:, 0] / discrete_allocation.iloc[:, 1], 0)
        allocations = pd.concat([currency_amount, discrete_allocation], axis= 1)
        
        return allocations

    allocations_MaxSR = pd.concat([max_sharpe_allocation, portfolio_discrete_allocation(max_sharpe_allocation, stock_data, amount)], axis=1)
    allocations_MinVol = pd.concat([minimum_variance_allocation, portfolio_discrete_allocation(minimum_variance_allocation, stock_data, amount)], axis=1)
    allocations_MaxSR.columns = ['ALLOCATION (%)', 'AMOUNT (€)', 'DISCRETE ALLOCATION (pcs.)']
    allocations_MinVol.columns = ['ALLOCATION (%)', 'AMOUNT (€)', 'DISCRETE ALLOCATION (pcs.)']

    print('\n')
    print('Optimization Results'.center(79))
    print(79* "=")
    print('Max Sharpe optimization status:'.ljust(69) + str(status_max_sharpe))
    print('Min Volatility optimization status:'.ljust(69) + str(status_minimum_variance))
    print('Efficient frontier optimization status:'.ljust(69) + str(status_efficient_frontier))
    print('')
    print('L2 Regularisation lambda coefficient:'.ljust(69) + str(lambda_coeff))
    print('Weight constraints:'.ljust(69) + str(constaints))
    print('')
    print('Iterations (MaxSR):'.ljust(69) + str(result_max_sharpe.nit))
    print('Iterations (MinVol):'.ljust(69) + str(result_minimum_variance.nit))
    print('Iterations (EF):'.ljust(69) + str(ef_nits)) 
    print('')
    print('Optimization method:'.ljust(69) + method)
    print('Date:'.ljust(69) + str(dt.datetime.now().date()))
    print('Time:'.ljust(69) + str(dt.datetime.now().time().replace(microsecond=0)))
    print('')
    print(79*'=')
    print('Max Sharpe Allocations'.center(79))
    print(79*'-')
    print(allocations_MaxSR)
    print('')
    print('Annualized portfolio return:'.ljust(69) + str(round(max_sharpe_returns, 2)))
    print('Annualized portfolio standard deviation:'.ljust(69) + str(round(max_sharpe_stdev, 2)))
    print('Annualized portfolio Sharpe ratio:'.ljust(69) + str(round(-result_max_sharpe.fun, 2)))
    print('')
    print(79*'=')
    print('Min Volatility Allocations'.center(79))
    print(79*'-')
    print(allocations_MinVol)
    print('')
    print('Annualized portfolio return:'.ljust(69) + str(round(minimum_variance_returns, 2)))
    print('Annualized portfolio standard deviation:'.ljust(69) + str(round(minimum_variance_stdev, 2)))
    print('Annualized portfolio Sharpe ratio:'.ljust(69) + str(round(minimum_variance_sharpe, 2)))
    print('')
    print(79*'=')
    print('Message (MaxSR): ' + result_max_sharpe.message)
    print('Message (MinVol): ' + result_minimum_variance.message)
    print(79*'-')

def plot(x: list, y: np.ndarray, z: list, minimum_variance_returns: float, minimum_variance_stdev: float, 
         max_sharpe_returns: float, max_sharpe_stdev: float, result_max_sharpe: sc.OptimizeResult, corr_matrix: pd.DataFrame):
    """
    Plots the efficient frontier, maximum Sharpe ratio, minimum volatility, and a correlation heatmap for a given set 
    of portfolio returns, volatilities, and other relevant data.

    Parameters:
        x (list): List of portfolio volatilities.
        y (np.ndarray): Array of portfolio returns.
        z (list): List of Sharpe ratios for the portfolios.
        minimum_variance_returns (float): Expected return of the minimum volatility portfolio.
        minimum_variance_stdev (float): Standard deviation of the minimum volatility portfolio.
        max_sharpe_returns (float): Expected return of the maximum Sharpe ratio portfolio.
        max_sharpe_stdev (float): Standard deviation of the maximum Sharpe ratio portfolio.
        result_max_sharpe (sc.OptimizeResult): Result of the maximum Sharpe ratio optimization.
        corr_matrix (pd.DataFrame): Correlation matrix of stock returns.

    Returns:
        fig (matplotlib.figure.Figure): The generated figure containing the plots.
    """
    scatter = plt.scatter(x, y, c= z, marker= 'o', cmap='viridis', label='Efficient frontier')

    max_sharpe_ratio = round(-result_max_sharpe.fun, 2)
    min_vol_sharpe_ratio = round((minimum_variance_returns-0.03)/minimum_variance_stdev, 2)

    plt.scatter(max_sharpe_stdev, max_sharpe_returns, marker='o', color='r', s=200, label=f'Maximum Sharpe ratio: {max_sharpe_ratio}')
    plt.scatter(minimum_variance_stdev, minimum_variance_returns, marker='o', color='g', s=200, label=f'Minimum volatility: {min_vol_sharpe_ratio}')

    plt.xlabel('Volatility')
    plt.ylabel('Annual returns')
    plt.legend(labelspacing=0.8, loc= 'lower right')
    plt.title('Efficient frontier', fontdict={'fontsize':18}, pad=12)
    plt.colorbar(scatter, label='Sharpe ratio')

    fig = plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='rocket_r', square=True, vmin=-1, vmax=1, mask= mask, cbar= True)
    ax.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    return fig

def main(tickers: list, years: int):
    """
    Main function that performs portfolio optimization, generates results, and plots. It first downloads stock data, 
    calculates portfolio returns and covariance matrix, and then optimizes the portfolio for minimum variance, 
    maximum Sharpe ratio, and efficient frontier. Finally, it generates and prints the results, and plots the efficient 
    frontier and correlation heatmap.

    Parameters:
        tickers (list): List of stock tickers to analyze.
        years (int): Number of years of historical data to download.

    Returns:
        plotting (matplotlib.figure.Figure): The figure containing the generated plots (efficient frontier and correlation heatmap).
    """
    constraints = (0,1)
    rfr = 0.03
    lambda_coeff = 0.1
    method = 'SLSQP'
    amount = 10000
    
    data = download(tickers, years)
    rets, cov_matrix, corr_matrix = return_statistics(data)
    result_minimum_variance, minimum_variance_returns, minimum_variance_stdev, minimum_variance_allocation, minimum_variance_sharpe = minimum_variance(rets, cov_matrix, constraints, rfr, method, lambda_coeff)
    result_max_sharpe, max_sharpe_returns, max_sharpe_stdev, max_sharpe_allocation = max_sharpe(rets, cov_matrix, constraints, rfr, method, lambda_coeff)
    efficient, return_target, sharpe, result_efficient_frontier = efficient_frontier(rets, cov_matrix, constraints, minimum_variance_returns, max_sharpe_returns, method, lambda_coeff)
    plotting = plot(efficient, return_target, sharpe, minimum_variance_returns, minimum_variance_stdev, max_sharpe_returns, max_sharpe_stdev, result_max_sharpe, corr_matrix)
    summary(result_max_sharpe, result_minimum_variance, result_efficient_frontier, minimum_variance_sharpe, minimum_variance_returns, minimum_variance_stdev, max_sharpe_returns, max_sharpe_stdev, method, max_sharpe_allocation, minimum_variance_allocation, data, constraints, lambda_coeff, amount)
    return plotting

r = main(["SAMPO.HE", "ENENTO.HE", "KCR.HE", "GOFORE.HE", "NESTE.HE", "OMASP.HE", "QTCOM.HE", "REG1V.HE", "VALMT.HE", "ICP1V.HE", "METSO.HE", "KNEBV.HE", "NDA-FI.HE"], 5)
plt.show()