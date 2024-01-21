from typing import Dict, List
import pandas as pd
from env_stocktrading.env_portfolio_allocation import SimplePortfolioAllocationBaseEnv
from preprocessing.custom_technical_indicators import TechnicalIndicator
from preprocessing.process_yh_finance import YHFinanceProcessor

def download_and_clean_data(yfp: YHFinanceProcessor, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Download and clean stock data for multiple stocks over a given date range.
    """
    # Download and concatenate data for all tickers
    all_data = dict()
    for ticker in tickers:
        df_raw = yfp.download_data([ticker], start_date, end_date, "1d")
        cleaned_data = yfp.clean_data(df_raw)
        all_data[ticker] = cleaned_data

    return all_data

def augment_data(yfp: YHFinanceProcessor, df_stocks: dict, indicators: List[str], custom_technical_indicators: List[TechnicalIndicator] = [], vix: bool = True):
    """
    Augment data for each stock with technical indicators and optionally VIX.

    Parameters:
    - yfp (YHFinanceProcessor): Yahoo Finance data processor.
    - df_stocks (dict): Dictionary of DataFrames, each representing stock data.
    - indicators (List[str]): List of standard technical indicators.
    - custom_technical_indicators (List[TechnicalIndicator]): List of custom technical indicators.
    - vix (bool): Flag to include VIX data.

    Returns:
    - dict: Dictionary of augmented DataFrames for each stock.
    """
    augmented_stocks = {}
    for ticker, df in df_stocks.items():
        # Create technical indicators array for each stock
        data_with_technical_indicators = yfp.add_technical_indicators(df, indicators, custom_technical_indicators)
        # Extract technical indicators columns from data, by excluding "timestamp", 'open', 'high', 'low', 'close', 'volume', "tic"
        tech_array_columns = [col for col in data_with_technical_indicators.columns if col not in ["timestamp", 'open', 'high', 'low', 'close', 'volume', "tic"]]

        for i, tech_indicator in enumerate(tech_array_columns):
            # Add technical indicators to the data
            df[f"tech_{i}"] = data_with_technical_indicators[tech_indicator].values

        if vix:
            df = yfp.add_vix(df)

        augmented_stocks[ticker] = df.dropna().reset_index(drop=True)

    return augmented_stocks

def save_data(df: pd.DataFrame, filename: str):
    """
    Save a DataFrame to a CSV file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - filename (str): The name of the file to save the data in.

    This function saves the provided DataFrame into a CSV file with the given filename.
    """
    df.to_csv(filename)

def create_environment(stock_data: Dict[str, pd.DataFrame], gamma: float = 0.99):
    """
    Create and return a portfolio allocation environment with data including technical indicators.

    Parameters:
    - stock_data (Dict[str, pd.DataFrame]): Dictionary with stock tickers as keys and corresponding
      DataFrame (including technical indicators) as values.
    - gamma (float): Discount factor for the environment.

    Returns:
    - SimplePortfolioAllocationBaseEnv: An instance of the portfolio allocation environment.
    """
    processed_stock_data = {}
    for ticker, data in stock_data.items():
        # Assuming the data includes technical indicators as part of its columns
        processed_stock_data[ticker] = {
            'close': data['close'].values,
            'open': data['open'].values,
            'high': data['high'].values,
            'low': data['low'].values,
            # Extract technical indicators from the data
            'tech_array': data.filter(regex='tech_').values
        }

    return SimplePortfolioAllocationBaseEnv(
        stock_data=processed_stock_data,
        initial_account_value=1000000,
        is_training_mode=True,
        discount_factor=gamma
    )
