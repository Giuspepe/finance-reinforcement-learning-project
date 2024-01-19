from datetime import timedelta
from typing import List
from pandas.tseries.offsets import BDay
import datetime
import re

import exchange_calendars as tc
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
import pytz
import stockstats as ss
import yfinance as yf

from preprocessing.custom_technical_indicators import TechnicalIndicator

# Code adapted from FinRL library: https://github.com/AI4Finance-LLC/FinRL


class YHFinanceProcessor:
    def __init__(self):
        pass

    def download_data(
        self,
        ticker_list: "list[str]",
        start_date: str,
        end_date: str,
        interval: str,
    ) -> pd.DataFrame:
        """
        Downloads historical stock price data for the given ticker(s) within the specified date range and interval.

        Args:
            ticker_list (list[str]): List of ticker symbols.
            start_date (str): Start date in the format 'YYYY-MM-DD'.
            end_date (str): End date in the format 'YYYY-MM-DD'.
            interval (str): Interval for the data (e.g., '1d' for daily, '1h' for hourly).

        Returns:
            pd.DataFrame: DataFrame containing the downloaded stock price data.

        Raises:
            ValueError: If the interval format is invalid.
        """
        # Check if the interval is valid
        if not self.is_valid_interval(interval):
            raise ValueError("Invalid interval format")

        # Initialize the start and end date and interval
        self.start = start_date
        self.end = end_date
        self.interval = interval

        # Download the data into a dataframe
        start_date = pd.Timestamp(start_date, tz=pytz.UTC)
        end_date = pd.Timestamp(end_date, tz=pytz.UTC)
        delta = timedelta(days=1)
        data = pd.DataFrame()
        for ticker in ticker_list:
            print(f"Downloading data for {ticker}...")
            while start_date <= end_date:
                print(f"Downloading data for {ticker} on {start_date}...")
                temp_df = yf.download(
                    ticker,
                    start=start_date,
                    end=start_date + delta,
                    interval=interval,
                )
                temp_df["ticker"] = ticker
                data = pd.concat([data, temp_df])
                start_date += delta

            # Reset the start date and end date
            start_date = pd.Timestamp(self.start, tz=pytz.UTC)
            end_date = pd.Timestamp(self.end, tz=pytz.UTC)

        data = data.reset_index().drop(columns=["Adj Close"])
        data.columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tic",
        ]

        return data

    def get_trading_days(self, start_date: str, end_date: str) -> "list[str]":
        """
        Get the trading days between the start and end date.

        Args:
            start_date (str): Start date in the format 'YYYY-MM-DD'.
            end_date (str): End date in the format 'YYYY-MM-DD'.

        Returns:
            list[str]: List of trading days between the start and end date.
        """
        # Get the NYSE calendar
        nyse = tc.get_calendar("NYSE")
        # Get the trading days between the start and end date
        df = nyse.sessions_in_range(
            pd.Timestamp(start_date, tz=pytz.UTC), pd.Timestamp(end_date, tz=pytz.UTC)
        )
        # Convert the trading days to a list of strings in the format 'YYYY-MM-DD'
        trading_days = [str(day)[:10] for day in df]
        return trading_days

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the downloaded stock price data.

        Args:
            data (pd.DataFrame): DataFrame containing the downloaded stock price data.

        Returns:
            pd.DataFrame: DataFrame containing the cleaned stock price data.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        # Validate interval
        if self.interval not in ["1d", "1m"]:
            raise ValueError("Invalid interval")

        tickers = data["tic"].unique()
        times = self._generate_time_index()
        cleaned_data = []

        for ticker in tickers:
            ticker_data = data[data["tic"] == ticker].copy()
            ticker_df = ticker_data.set_index("timestamp").reindex(times)
            ticker_df["tic"] = ticker  # Add ticker column

            # Fill missing values for specified columns
            for column in ["open", "high", "low", "close"]:
                ticker_df[column].fillna(method="ffill", inplace=True)

            ticker_df["volume"].fillna(0, inplace=True)
            ticker_df.fillna(0, inplace=True)  # For remaining NaN values
            cleaned_data.append(ticker_df)

        # Rename index to 'timestamp' before concatenating
        cleaned_data = [
            df.reset_index().rename(columns={"index": "timestamp"})
            for df in cleaned_data
        ]

        return pd.concat(cleaned_data)

    def _generate_time_index(self):
        if self.interval == "1d":
            return pd.date_range(start=self.start, end=self.end, freq=BDay())
        elif self.interval == "1m":
            # Generate minute-by-minute time index for each trading day
            trading_days = pd.date_range(start=self.start, end=self.end, freq=BDay())
            return pd.DatetimeIndex(
                [
                    dt + pd.Timedelta(minutes=i)
                    for dt in trading_days
                    for i in range(390)  # 390 minutes in a trading day
                ]
            )
        

    def add_technical_indicators(
        self, data: pd.DataFrame, tech_indicator_list: "list[str]", custom_technical_indicators: List[TechnicalIndicator] = []
    ) -> pd.DataFrame:
        """
        Add technical indicators to the given DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame containing stock data.
            tech_indicator_list (list[str]): A list of technical indicators to be added.

        Returns:
            pd.DataFrame: The DataFrame with the added technical indicators.
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "timestamp"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    # Retrieve the indicator values for each ticker
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["timestamp"] = df[df.tic == unique_ticker[i]][
                        "timestamp"
                    ].to_list()
                    # Concatenate the indicator values for all tickers
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], ignore_index=True
                    )
                except Exception as e:
                    print(e)
            # Merge the indicator values with the original DataFrame
            df = df.merge(
                indicator_df[["tic", "timestamp", indicator]],
                on=["tic", "timestamp"],
                how="left",
            )


        df = df.sort_values(by=["timestamp", "tic"])

        # Add custom technical indicators
        df_technical_indicators = pd.DataFrame()
        for indicator in custom_technical_indicators:
            indicator.add_column(df_before=df, df_after=df_technical_indicators)
        df = df.join(other=df_technical_indicators, how="inner")
        return df

    def add_vix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add VIX (Volatility Index) data to the dataframe.

        Args:
            data (pd.DataFrame): DataFrame containing stock price data.

        Returns:
            pd.DataFrame: DataFrame with VIX data added.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        vix_df = self.download_data(["VIXY"], self.start, self.end, self.interval)
        cleaned_vix = self.clean_data(vix_df)

        # Select timestamp and VIXY close price, then rename column
        vix = cleaned_vix[["timestamp", "close"]].rename(columns={"close": "VIXY"})

        # Merge VIX data with original dataframe
        merged_df = data.merge(vix, on="timestamp", how="left")
        return merged_df.sort_values(["timestamp", "tic"]).reset_index(drop=True)

    @staticmethod
    def is_valid_interval(time_interval: str) -> bool:
        """
        Check if the given time interval is valid.

        Args:
            time_interval (str): The time interval to be checked.

        Returns:
            bool: True if the time interval is valid, False otherwise.
        """
        # Define a regular expression pattern for a valid interval format
        pattern = r"^[0-9]+[mhdwkmo]$"

        # Use re.match to check if the input interval matches the pattern
        return bool(re.match(pattern, time_interval))

    def calculate_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        """
        Calculate the turbulence index for stock prices.

        Args:
            data (pd.DataFrame): DataFrame containing stock price data.
            time_period (int): Time period for calculating rolling window covariance.

        Returns:
            pd.DataFrame: DataFrame with turbulence index.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        df_price_pivot = data.pivot(
            index="timestamp", columns="tic", values="close"
        ).pct_change()
        unique_dates = df_price_pivot.index

        # Initialize turbulence index list
        turbulence_index = [0] * len(unique_dates)
        rolling_covariances = df_price_pivot.rolling(
            window=time_period, min_periods=time_period
        ).cov()

        for i in range(time_period, len(unique_dates)):
            current_date = unique_dates[i]
            current_return = df_price_pivot.loc[current_date]

            cov_matrix = (
                rolling_covariances.loc[(current_date, slice(None)), :]
                .unstack(level=1)  # Convert to DataFrame with MultiIndex
                .dropna(axis=1, how="all")
                .dropna(axis=0, how="all")
            )
            if not cov_matrix.empty:
                common_tickers = cov_matrix.columns.get_level_values(1)
                current_return = current_return[common_tickers]
                mean_return = df_price_pivot[common_tickers].loc[:current_date].mean()
                deviation = current_return - mean_return
                turbulence = deviation.dot(np.linalg.pinv(cov_matrix)).dot(deviation.T)
                turbulence_index[i] = max(0, turbulence)  # Ensure non-negative values

        return pd.DataFrame({"timestamp": unique_dates, "turbulence": turbulence_index})

    def add_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        """
        Add turbulence index to the stock data.

        Args:
            data (pd.DataFrame): DataFrame containing stock price data.
            time_period (int): Time period for calculating the turbulence index.

        Returns:
            pd.DataFrame: DataFrame with turbulence index added.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        # Calculate turbulence index
        turbulence_index = self.calculate_turbulence(data, time_period)

        # Merge the turbulence index with the original data
        merged_data = data.merge(turbulence_index, on="timestamp", how="left")
        return merged_data.sort_values(["timestamp", "tic"]).reset_index(drop=True)

    def df_to_array(
        self, df: pd.DataFrame, tech_indicator_list: "list[str]", custom_technical_indicators: List[TechnicalIndicator], use_vix: bool
    ) -> "list[np.ndarray]":
        """
        Convert DataFrame to arrays for price, technical indicators, and either VIX or turbulence.

        Args:
            df (pd.DataFrame): DataFrame containing stock price data.
            tech_indicator_list (list[str]): List of technical indicators to include.
            use_vix (bool): Flag to determine whether to use VIX or turbulence.

        Returns:
            list[np.ndarray]: List containing arrays for price, technical indicators, and VIX/turbulence.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        unique_tickers = df["tic"].unique()
        close_arrays = []
        open_arrays = []
        high_arrays = []
        low_arrays = []
        tech_arrays = []
        turb_arrays = []

        for tic in unique_tickers:
            ticker_data = df[df["tic"] == tic]
            close_arrays.append(ticker_data[["close"]].values)
            open_arrays.append(ticker_data[["open"]].values)
            high_arrays.append(ticker_data[["high"]].values)
            low_arrays.append(ticker_data[["low"]].values)
            # Get array of custom technical indicators by looking at the columns of the dataframe and selecting the ones that start with
            # the prefix "CT_"
            custom_tech_arrays = ticker_data[[col for col in ticker_data.columns if col.startswith("CT_")]].values
            # Get array of technical indicators by looking at the columns of the dataframe and selecting the ones that are in the list
            # of technical indicators
            tech_arrays.append(ticker_data[tech_indicator_list].values)
            tech_arrays.append(custom_tech_arrays)
            if use_vix:
                turb_arrays.append(ticker_data["VIXY"].values)

        # Use numpy.hstack to concatenate arrays horizontally
        close_arrays = np.hstack(close_arrays)
        open_arrays = np.hstack(open_arrays)
        high_arrays = np.hstack(high_arrays)
        low_arrays = np.hstack(low_arrays)
        tech_array = np.hstack(tech_arrays)
        if use_vix:
            turbulence_array = np.hstack(turb_arrays)
        else:
            turbulence_array = np.array([])

        return close_arrays, open_arrays, high_arrays, low_arrays, tech_array, turbulence_array

    def fetch_latest_data(
        self, ticker_list, time_interval, tech_indicator_list, limit=100
    ):
        """
        Fetch the latest stock data for a list of tickers.

        Args:
            ticker_list (list[str]): List of ticker symbols.
            time_interval (str): Time interval for fetching data.
            tech_indicator_list (list[str]): List of technical indicators to add.
            limit (int): Number of data points to fetch.

        Returns:
            Tuple: Contains arrays for latest price, technical indicators, and turbulence.
        """

        # Check if the interval is valid
        if not self.is_valid_interval(time_interval):
            raise ValueError("Invalid interval format")

        # Get start and end dates
        end_datetime = datetime.datetime.now()
        start_datetime = end_datetime - datetime.timedelta(minutes=limit + 1)

        # Fetch data for each ticker
        data_dfs = []
        for tic in ticker_list:
            barset = yf.download(
                tic, start=start_datetime, end=end_datetime, interval=time_interval
            )
            barset["tic"] = tic
            data_dfs.append(barset)

        # Concatenate and clean data
        data_df = pd.concat(data_dfs).reset_index().drop(columns=["Adj Close"])
        data_df.columns = ["timestamp", "open", "high", "low", "close", "volume", "tic"]
        data_df = self.clean_data(data_df)  # Clean data

        # Add technical indicators
        df_with_indicators = self.add_technical_indicator(data_df, tech_indicator_list)

        # Convert to arrays for price, technical indicators, and VIX
        close_array, open_array, high_array, low_array, tech_array, _ = self.df_to_array(
            df_with_indicators, tech_indicator_list, if_vix=True
        )

        # Fetch latest turbulence data
        latest_turb = yf.download(
            "VIXY", start=start_datetime, end=end_datetime, interval=time_interval
        )["Close"].values

        return close_array[-1], open_array[-1], high_array[-1], low_array[-1], tech_array[-1], latest_turb