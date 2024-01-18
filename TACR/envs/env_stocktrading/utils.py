from typing import List

import pandas as pd
from TACR.envs.env_stocktrading.env_stocktrading import SimpleOneStockStockTradingBaseEnv
from preprocessing.custom_technical_indicators import TechnicalIndicator
from preprocessing.process_yh_finance import YHFinanceProcessor


def download_and_clean_data(yfp: YHFinanceProcessor, tickers: List[str], start_date: str, end_date: str):
    """
    Download and clean stock data for a given range of dates.

    Parameters:
    - yfp (YHFinanceProcessor): The Yahoo Finance data processor.
    - tickers (List[str]): List of stock tickers to download data for.
    - start_date (str): Start date for the data download.
    - end_date (str): End date for the data download.

    Returns:
    - pd.DataFrame: Cleaned DataFrame with stock data.

    This function downloads stock data using Yahoo Finance for the specified tickers and date range.
    It then cleans the data using the provided Yahoo Finance Processor.
    """
    df_raw = yfp.download_data(tickers, start_date, end_date, "1d")
    return yfp.clean_data(df_raw)


def augment_data(yfp: YHFinanceProcessor, df: pd.DataFrame, indicators: List[str], custom_technical_indicators: List[TechnicalIndicator] = []):
    """
    Augment the stock data with technical indicators and turbulence index.

    Parameters:
    - yfp (YHFinanceProcessor): The Yahoo Finance data processor.
    - df (pd.DataFrame): The DataFrame with stock data.
    - indicators (List[str]): List of standard technical indicators to include.
    - custom_technical_indicators (List[TechnicalIndicator], optional): List of custom technical indicators to include.

    Returns:
    - pd.DataFrame: DataFrame with additional technical indicators and turbulence index.

    This function enhances the provided DataFrame with standard and custom technical indicators,
    and also adds a turbulence index for each data point.
    """
    df = yfp.add_technical_indicators(df, indicators, custom_technical_indicators)
    df = yfp.add_turbulence(df, 252) # Assuming 252 trading days in a year
    df = yfp.add_vix(df) # Add VIX index

    # Drop rows with NaN values and reset the index
    df = df.dropna().reset_index(drop=True)
    
    return df


def save_data(df: pd.DataFrame, filename: str):
    """
    Save a DataFrame to a CSV file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - filename (str): The name of the file to save the data in.

    This function saves the provided DataFrame into a CSV file with the given filename.
    """
    df.to_csv(filename)


def create_environment(yfp: YHFinanceProcessor, dataset: pd.DataFrame, indicators: List[str], custom_technical_indicators: List[TechnicalIndicator] = []):
    """
    Create and return a stock trading environment.

    Parameters:
    - yfp (YHFinanceProcessor): The Yahoo Finance data processor.
    - dataset (pd.DataFrame): The DataFrame containing stock data.
    - indicators (List[str]): List of standard technical indicators to include.
    - custom_technical_indicators (List[TechnicalIndicator], optional): List of custom technical indicators to include.

    Returns:
    - SimpleOneStockStockTradingBaseEnv: An instance of the stock trading environment.

    This function creates a trading environment using the provided dataset and technical indicators.
    The environment can be used for training reinforcement learning models.
    """
    # Convert the DataFrame to arrays for price, technical indicators, and turbulence index
    price_array, tech_array, turbulence_array = yfp.df_to_array(dataset, indicators, custom_technical_indicators, True) 
    return SimpleOneStockStockTradingBaseEnv(price_array=price_array, tech_array=tech_array, is_training_mode=True)
