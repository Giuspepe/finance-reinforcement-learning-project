from typing import List

import pandas as pd
from TACR.envs.env_stocktrading.env_stocktrading import SimpleOneStockStockTradingBaseEnv
from preprocessing.custom_technical_indicators import TechnicalIndicator
from preprocessing.process_yh_finance import YHFinanceProcessor


def download_and_clean_data(yfp: YHFinanceProcessor, tickers: List[str], start_date: str, end_date: str):
    """
    Downloads and cleans the stock data.
    """
    df_raw = yfp.download_data(tickers, start_date, end_date, "1d")
    return yfp.clean_data(df_raw)


def augment_data(yfp: YHFinanceProcessor, df: pd.DataFrame, indicators: List[str], custom_technical_indicators: List[TechnicalIndicator] = []):
    """
    Adds technical indicators and turbulence index to the data.
    """
    df = yfp.add_technical_indicators(df, indicators, custom_technical_indicators)
    df = yfp.add_turbulence(df, 252)
    df = yfp.add_vix(df)

    # Drop rows with NaN values
    df = df.dropna().reset_index(drop=True)
    
    return df


def save_data(df: pd.DataFrame, filename: str):
    """
    Saves the DataFrame to a CSV file.
    """
    df.to_csv(filename)


def create_environment(yfp: YHFinanceProcessor, dataset: pd.DataFrame, indicators: List[str], custom_technical_indicators: List[TechnicalIndicator] = []):
    """
    Creates and returns the trading environment instance.
    """
    price_array, tech_array, turbulence_array = yfp.df_to_array(dataset, indicators, custom_technical_indicators, True) 
    return SimpleOneStockStockTradingBaseEnv(price_array=price_array, tech_array=tech_array, is_training_mode=True)

