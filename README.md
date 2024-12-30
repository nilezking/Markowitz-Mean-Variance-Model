# Portfolio Optimization

![Portfolio Chart](https://raw.githubusercontent.com/nilezking/Markowitz-Mean-Variance-Model/main/Figure_1.png)

## Overview

This project implements portfolio optimization techniques including maximizing the Sharpe ratio, minimizing volatility, and generating the efficient frontier. It also includes visualization tools to plot the efficient frontier, maximum Sharpe ratio, minimum volatility, and correlation heatmap of asset returns. The portfolio optimization is performed using historical stock data and custom constraints, with the option to apply L2 regularization. Optimization process rely on numerical methods. Purpose of the project is to assist the portfolio manager in asset allocation.

## Features

- **Maximum Sharpe Ratio Optimization:** Optimizes the portfolio allocation to maximize the Sharpe ratio.
- **Minimum Volatility Optimization:** Optimizes the portfolio allocation to minimize volatility.
- **Efficient Frontier:** Generates the efficient frontier based on the minimum variance and maximum Sharpe ratio portfolios.
- **Discrete Allocation:** Converts portfolio allocations into discrete stock units based on the total amount.
- **Visualization:** Plots the efficient frontier, maximum Sharpe ratio, minimum volatility, and a correlation heatmap.
- **Flexible Constraints:** Supports portfolio weight constraints and L2 regularization to punish extreme weights.

## Installation

Clone this repository:

    git clone https://github.com/nilezking/Markowitz-Mean-Variance-Model.git

Navigate to the project directory:

    cd Markowitz-Mean-Variance-Model

Install the required dependencies on your virtual environment:

    pip install -r requirements.txt

## Usage

To run the portfolio optimization and generate the results, call the main() function with a list of stock tickers and the number of years of historical data you'd like to download.

    from mean_variance_optimizer import main

Define the list of stock tickers and the number of years of historical data

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    years = 5

Run the main function to perform the optimization and plotting

    main(tickers, years)

### Parameters

    tickers (list): A list of stock tickers to include in the optimization (e.g., ['AAPL', 'MSFT', 'GOOGL']).
    years (int): The number of years of historical data to download (e.g., 5).

### Functions

**summary()**

Generates a summary report of the optimization results, including the status of each optimization, portfolio allocations, and performance metrics (e.g., annualized return, standard deviation, Sharpe ratio).

**plot()**

Generates and displays a plot of the efficient frontier, the maximum Sharpe ratio portfolio, minimum volatility portfolio, and a correlation heatmap of the assets.

**main()**

Performs the portfolio optimization using the maximum Sharpe ratio and minimum volatility strategies. It also generates the efficient frontier and calls the summary() and plot() functions to display the results.

## Requirements

    Python 3
    numpy
    pandas
    matplotlib
    seaborn
    scipy
    yfinance

To install the dependencies, run:

    pip install -r requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project uses historical stock data downloaded via the yfinance package.
Portfolio optimization methods were implemented based on Markowitz's modern portfolio theory.