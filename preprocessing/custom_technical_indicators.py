from abc import abstractmethod
import numpy as np

import pandas as pd
import pandas_ta as ta


class TechnicalIndicator:
    def __init__(self, symbol: str):
        """
        Initialize a TechnicalIndicator object.

        Parameters:
        - symbol (str): A symbol representing the technical indicator.

        This is an abstract base class for creating technical indicators. Each subclass should implement
        its own methods for adding the indicator column to a DataFrame and generating a string identifier.
        """
        self.symbol = symbol

    @abstractmethod
    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        """
        Add a new column to the DataFrame based on this technical indicator.

        Parameters:
        - df_before (pd.DataFrame): DataFrame before applying the indicator.
        - df_after (pd.DataFrame): DataFrame where the new indicator column will be added.

        This method should be implemented by subclasses to calculate and add the technical indicator to df_after.
        """
        pass

    @abstractmethod
    def string_identifier(self) -> str:
        """
        Generate a string identifier for this technical indicator.

        Returns:
        - str: A string identifier for the technical indicator.

        This method should be implemented by subclasses to return a unique string identifier for the indicator.
        """
        pass

    @staticmethod
    def return_free_column_from_symbol_in_df(df: pd.DataFrame, symbol: str):
        """
        Generate a unique column name for a given symbol in a DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame to check for existing columns.
        - symbol (str): The symbol to base the column name on.

        Returns:
        - str: A unique column name for the given symbol in the DataFrame.

        This static method generates a unique column name for the DataFrame, ensuring no naming conflicts with existing columns.
        """
        index = 0
        while "CT_" + symbol + f"_{index}" in df.columns:
            index = index + 1
        free_symbol_in_df = "CT_" + symbol + f"_{index}"
        return free_symbol_in_df


class RSI(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="RSI")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = ta.rsi(df_before["close"], length=self.length)

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(rsi_l={self.length})"
        return identifier


class Z_SCORE(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="Z_SCORE")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = ta.zscore(df_before["close"], length=self.length)

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(z_score_l={self.length})"
        return identifier


class PCT_RETURN(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="PCT_RETURN")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = ta.percent_return(close=df_before["close"], length=self.length)

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(pct_r_l={self.length})"
        return identifier


class DRAWDOWN(TechnicalIndicator):
    def __init__(self):
        super().__init__(symbol="DRAWDOWN")

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = ta.drawdown(close=df_before["close"])["DD_PCT"]

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}"
        return identifier


class RVI_PCT_CHANGE(TechnicalIndicator):
    def __init__(self, length: int, rvi_pct_change_length: int):
        super().__init__(symbol="RVI_PCT_CHANGE")
        self.length = length
        self.rvi_pct_change_length = rvi_pct_change_length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        rvi = ta.rvi(
            close=df_before["close"],
            high=df_before["high"],
            low=df_before["low"],
            length=self.length,
        )
        rvi_pct_change = rvi.pct_change(periods=self.rvi_pct_change_length)
        rvi_pct_change = rvi_pct_change.replace(to_replace=np.inf, value=0)
        rvi_pct_change = rvi_pct_change.replace(to_replace=np.NINF, value=0)
        rvi_pct_change = rvi_pct_change.replace(to_replace=np.nan, value=0)
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = rvi_pct_change

    def string_identifier(self) -> str:
        identifier = (
            f"{self.symbol}(rvi_l={self.length}, pct_c_l={self.rvi_pct_change_length})"
        )
        return identifier


class RVI(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="RVI")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = ta.rvi(
            close=df_before["close"],
            high=df_before["high"],
            low=df_before["low"],
            length=self.length,
        )

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(rvi_l={self.length})"
        return identifier


class OBV(TechnicalIndicator):
    def __init__(self):
        super().__init__(symbol="OBV")

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = ta.obv(close=df_before["close"], volume=df_before["volume"])


class OBV_PCT_CHANGE(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="OBV_PCT_CHANGE")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        obv = ta.obv(close=df_before["close"], volume=df_before["volume"])
        obv_pct_change = obv.pct_change(periods=self.length)
        obv_pct_change = obv_pct_change.replace(to_replace=np.inf, value=0)
        obv_pct_change = obv_pct_change.replace(to_replace=np.NINF, value=0)
        obv_pct_change = obv_pct_change.replace(to_replace=np.nan, value=0)
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = obv_pct_change

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(pct_l={self.length})"
        return identifier


class STOCKRSI(TechnicalIndicator):
    def __init__(self, stockrsi_length: int, rsi_length: int):
        super().__init__(symbol="STOCKRSI")
        self.stockrsi_length = stockrsi_length
        self.rsi_length = rsi_length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        stockrsi = ta.stochrsi(
            close=df_before["close"],
            length=self.stockrsi_length,
            rsi_length=self.rsi_length,
        )
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol + "_slow"
            )
        ] = stockrsi[
            f"STOCHRSIk_{str(self.stockrsi_length)}_{str(self.rsi_length)}_3_3"
        ]
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol + "_fast"
            )
        ] = stockrsi[
            f"STOCHRSId_{str(self.stockrsi_length)}_{str(self.rsi_length)}_3_3"
        ]

    def string_identifier(self) -> str:
        identifier = (
            f"{self.symbol}(stockrsi_l={self.stockrsi_length}, rsi_l={self.rsi_length})"
        )
        return identifier


class STOCKRSI_PCT_CHANGE(TechnicalIndicator):
    def __init__(self, stockrsi_length: int, rsi_length: int, pct_change_length: int):
        super().__init__(symbol="STOCKRSI_PCT_CHANGE")
        self.stockrsi_length = stockrsi_length
        self.rsi_length = rsi_length
        self.pct_change_length = pct_change_length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        stockrsi = ta.stochrsi(
            close=df_before["close"],
            length=self.stockrsi_length,
            rsi_length=self.rsi_length,
        )
        stockrsi_pct_change = stockrsi.pct_change(periods=self.pct_change_length)
        stockrsi_pct_change = stockrsi_pct_change.replace(to_replace=np.inf, value=0)
        stockrsi_pct_change = stockrsi_pct_change.replace(to_replace=np.NINF, value=0)
        stockrsi_pct_change = stockrsi_pct_change.replace(to_replace=np.nan, value=0)
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol + "_slow"
            )
        ] = stockrsi_pct_change[
            f"STOCHRSIk_{str(self.stockrsi_length)}_{str(self.rsi_length)}_3_3"
        ]
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol + "_fast"
            )
        ] = stockrsi_pct_change[
            f"STOCHRSId_{str(self.stockrsi_length)}_{str(self.rsi_length)}_3_3"
        ]

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(stockrsi_l={self.stockrsi_length}, rsi_l={self.rsi_length}, pct_l={self.pct_change_length})"
        return identifier


class EFFRATIO(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="EFFRATIO")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = ta.er(close=df_before["close"], length=self.length)

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(eff_l={self.length})"
        return identifier


class TSI(TechnicalIndicator):
    def __init__(
        self, tsi_fast_length: int, tsi_slow_length: int, tsi_signal_length: int
    ):
        super().__init__(symbol="TSI")
        self.tsi_fast_length = tsi_fast_length
        self.tsi_slow_length = tsi_slow_length
        self.tsi_signal_length = tsi_signal_length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        tsi = ta.tsi(
            close=df_before["close"],
            fast=self.tsi_fast_length,
            slow=self.tsi_slow_length,
            signal=self.tsi_signal_length,
        )
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol + "_v"
            )
        ] = tsi[
            f"TSI_{str(self.tsi_fast_length)}_{str(self.tsi_slow_length)}_{str(self.tsi_signal_length)}"
        ]
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol + "_signal"
            )
        ] = tsi[
            f"TSIs_{str(self.tsi_fast_length)}_{str(self.tsi_slow_length)}_{str(self.tsi_signal_length)}"
        ]

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(tsi_f={self.tsi_fast_length}, tsi_slow={self.tsi_slow_length}, tsi_signal={self.tsi_signal_length})"
        return identifier


class TSI_PCT_CHANGE(TechnicalIndicator):
    def __init__(
        self,
        tsi_fast_length: int,
        tsi_slow_length: int,
        tsi_signal_length: int,
        pct_change_length: int,
    ):
        super().__init__(symbol="TSI_PCT_CHANGE")
        self.tsi_fast_length = tsi_fast_length
        self.tsi_slow_length = tsi_slow_length
        self.tsi_signal_length = tsi_signal_length
        self.pct_change_length = pct_change_length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        tsi = ta.tsi(
            close=df_before["close"],
            fast=self.tsi_fast_length,
            slow=self.tsi_slow_length,
            signal=self.tsi_signal_length,
        )
        tsi_pct_change = tsi.pct_change(periods=self.pct_change_length)
        tsi_pct_change = tsi_pct_change.replace(to_replace=np.inf, value=0)
        tsi_pct_change = tsi_pct_change.replace(to_replace=np.NINF, value=0)
        tsi_pct_change = tsi_pct_change.replace(to_replace=np.nan, value=0)

        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol + "_v"
            )
        ] = tsi_pct_change[
            f"TSI_{str(self.tsi_fast_length)}_{str(self.tsi_slow_length)}_{str(self.tsi_signal_length)}"
        ]
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol + "_signal"
            )
        ] = tsi_pct_change[
            f"TSIs_{str(self.tsi_fast_length)}_{str(self.tsi_slow_length)}_{str(self.tsi_signal_length)}"
        ]

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(tsi_f={self.tsi_fast_length}, tsi_slow={self.tsi_slow_length}, tsi_signal={self.tsi_signal_length}, pct_l={self.pct_change_length})"
        return identifier


class EMA(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="EMA")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = ta.ema(close=df_before["close"], length=self.length)

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(ema_l={self.length})"
        return identifier


class EMA_PCT_CHANGE(TechnicalIndicator):
    def __init__(self, length: int, pct_change_length):
        super().__init__(symbol="EMA_PCT_CHANGE")
        self.length = length
        self.pct_change_length = pct_change_length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        ema = ta.ema(close=df_before["close"], length=self.length)
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = ema.pct_change(periods=self.pct_change_length)

    def string_identifier(self) -> str:
        identifier = (
            f"{self.symbol}(ema_l={self.length}, pct_l={self.pct_change_length})"
        )
        return identifier


class VWMA(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="VWMA")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = ta.vwma(
            close=df_before["close"], volume=df_before["volume"], length=self.length
        )

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(vwma_l={self.length})"
        return identifier


class VWMA_PCT_CHANGE(TechnicalIndicator):
    def __init__(self, length: int, pct_change_length):
        super().__init__(symbol="VWMA_PCT_CHANGE")
        self.length = length
        self.pct_change_length = pct_change_length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        vwma = ta.vwma(
            close=df_before["close"], volume=df_before["volume"], length=self.length
        )
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = vwma.pct_change(periods=self.pct_change_length)

    def string_identifier(self) -> str:
        identifier = (
            f"{self.symbol}(vwma_l={self.length}, pct_l={self.pct_change_length})"
        )
        return identifier


class CHOP(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="CHOP")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        chop = ta.chop(
            high=df_before["high"],
            low=df_before["low"],
            close=df_before["close"],
            length=self.length,
        )
        chop = chop.replace(to_replace=np.inf, value=50)
        chop = chop.replace(to_replace=np.NINF, value=50)
        chop = chop.replace(to_replace=np.nan, value=50)
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = chop

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(chop_l={self.length})"
        return identifier


class EBSW(TechnicalIndicator):
    def __init__(self):
        super().__init__(symbol="EBSW")

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = ta.ebsw(close=df_before["close"])


class DM(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="DM")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        dm = ta.dm(high=df_before["high"], low=df_before["low"], length=self.length)
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol + "P"
            )
        ] = dm[f"DMP_{self.length}"]
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol + "N"
            )
        ] = dm[f"DMN_{self.length}"]

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(dm_l={self.length})"
        return identifier


# class QQE(TechnicalIndicator):
#     def __init__(self, length: int):
#         super().__init__(symbol="QQE")
#         self.length = length

#     def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
#         qqe = ta.qqe(close=df_before["close"], length=self.length)

#         for column in qqe:
#             df_after[TechnicalIndicator.return_free_column_from_symbol_in_df(df=df_after, symbol=self.symbol+"_"+column)] = qqe[column]

class VORTEX(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="VORTEX")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        vortex = ta.vortex(
            high=df_before["high"],
            low=df_before["low"],
            close=df_before["close"],
            length=self.length,
        )
        for column in vortex:
            df_after[
                TechnicalIndicator.return_free_column_from_symbol_in_df(
                    df=df_after, symbol=self.symbol + "_" + column
                )
            ] = vortex[column]

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(length={self.length})"
        return identifier


class RVI(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="RVI")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        rvi = ta.rvi(
            high=df_before["high"],
            low=df_before["low"],
            close=df_before["close"],
            length=self.length,
        )
        for column in rvi:
            df_after[
                TechnicalIndicator.return_free_column_from_symbol_in_df(
                    df=df_after, symbol=self.symbol + "_" + column
                )
            ] = rvi[column]

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(length={self.length})"
        return identifier


class ADX(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="ADX")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        adx = ta.adx(
            high=df_before["high"],
            low=df_before["low"],
            close=df_before["close"],
            length=self.length,
        )
        adx = adx[[f"ADX_{self.length}"]]
        for column in adx:
            df_after[
                TechnicalIndicator.return_free_column_from_symbol_in_df(
                    df=df_after, symbol=self.symbol + "_" + column
                )
            ] = adx[column]

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(length={self.length})"
        return identifier


class TRIX(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="TRIX")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        trix = ta.trix(close=df_before["close"], length=self.length)
        trix = trix[[f"TRIX_{self.length}_9"]]
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = trix

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(length={self.length})"
        return identifier


class TRIX_PCT_CHANGE(TechnicalIndicator):
    def __init__(self, length: int, pct_change_length: int):
        super().__init__(symbol="TRIX_PCT_CHANGE")
        self.length = length
        self.pct_change_length = pct_change_length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        trix = ta.trix(close=df_before["close"], length=self.length)
        trix = trix[[f"TRIX_{self.length}_9"]]
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = trix.pct_change(periods=self.pct_change_length)

    def string_identifier(self) -> str:
        identifier = (
            f"{self.symbol}(length={self.length}, pct_l={self.pct_change_length})"
        )
        return identifier


class BINARY_SMA_RISING(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="BINARY_SMA_RISING")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        sma = ta.sma(close=df_before["close"], length=self.length)
        sma_pct_change = sma.pct_change(periods=2)
        binary_signal = sma_pct_change.apply(lambda x: 0 if x < 0 else 1)
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = binary_signal

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(length={self.length})"
        return identifier


class BINARY_SMA_HIGHER_THAN_SECOND_SMA(TechnicalIndicator):
    def __init__(self, length_1: int, length_2: int):
        super().__init__(symbol="BINARY_SMA_HIGHER_THAN_SECOND_SMA")
        self.length_1 = length_1
        self.length_2 = length_2

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        sma_1 = ta.sma(close=df_before["close"], length=self.length_1)
        sma_2 = ta.sma(close=df_before["close"], length=self.length_2)
        sma_diff = sma_1 - sma_2

        binary_signal = sma_diff.apply(lambda x: 0 if x < 0 else 1)
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = binary_signal

    def string_identifier(self) -> str:
        identifier = (
            f"{self.symbol}(length_1={self.length_1}, length_2={self.length_2})"
        )
        return identifier


class RSI_CATEGORICAL(TechnicalIndicator):
    def __init__(self, length: int):
        super().__init__(symbol="RSI_CATEGORICAL")
        self.length = length

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        rsi: pd.Series = ta.rsi(close=df_before["close"], length=self.length)

        categorical_signal = rsi.apply(lambda x: -1 if x < 30 else (1 if x > 70 else 0))
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = categorical_signal

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}(length={self.length})"
        return identifier


class DIST_PERCENTUAL_ENTRE_SMA(TechnicalIndicator):
    def __init__(self, length_1: int, length_2: int):
        super().__init__(symbol="DIST_PERCENTUAL_ENTRE_SMA")
        self.length_1 = length_1
        self.length_2 = length_2

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        sma_1 = ta.sma(close=df_before["close"], length=self.length_1)
        sma_2 = ta.sma(close=df_before["close"], length=self.length_2)
        sma_dist = (sma_1 - sma_2) / sma_1
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = sma_dist

    def string_identifier(self) -> str:
        identifier = (
            f"{self.symbol}(length_1={self.length_1}, length_2={self.length_2})"
        )
        return identifier


class CANDLES_PFR(TechnicalIndicator):
    def __init__(self):
        super().__init__(symbol="CANDLES_PFR")

    def add_column(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        min_2cd_ant = df_before["low"].rolling(2, closed="left").min()
        close_ant = df_before["close"].shift(1)

        PFR_low = np.where(df_before["low"] < min_2cd_ant, 1, 0)
        PFR_close = np.where(df_before["close"] > close_ant, 1, 0)
        PFR_sinal_compra = np.where((PFR_low + PFR_close) == 2, 1, 0)

        df_pfr_sinal_compra = pd.DataFrame(data=PFR_sinal_compra, index=df_before.index)
        df_after[
            TechnicalIndicator.return_free_column_from_symbol_in_df(
                df=df_after, symbol=self.symbol
            )
        ] = df_pfr_sinal_compra

    def string_identifier(self) -> str:
        identifier = f"{self.symbol}"
        return identifier
