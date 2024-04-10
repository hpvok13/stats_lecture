import os
from bisect import bisect_left
from warnings import warn

import pandas as pd
import torch
import yfinance as yf
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(
        self,
        symbols_file,
        seq_len,
        dir="data",
        period="60d",
        interval="2m",
        train=True,
        download=True,
        columns=["Datetime", "Open", "High", "Low", "Close", "Volume"],
    ):
        super().__init__()
        self.symbols = pd.read_csv(symbols_file)["Symbol"].to_list()
        if train:
            self.symbols = self.symbols[: int(len(self.symbols) * 0.8)]
        else:
            self.symbols = self.symbols[int(len(self.symbols) * 0.8) :]
        self.seq_len = seq_len
        self.dir = dir
        self.columns = columns
        if download:
            self._download(self.symbols, dir, period, interval, seq_len, columns)
        else:
            cum_sum = 0
            self.cum_len_tables = []
            kept_symbols = []
            for symbol in self.symbols:
                try:
                    len_table = pd.read_csv(os.path.join(dir, f"{symbol}.csv")).shape[0]
                except Exception as e:
                    warn(f"{e} exception when processing symbol {symbol}")
                    continue
                assert len_table >= self.seq_len
                kept_symbols.append(symbol)
                cum_sum += len_table - self.seq_len + 1
                self.cum_len_tables.append(cum_sum)
            self.symbols = kept_symbols

    def _download(self, symbols, dir, period, interval, seq_len, columns):
        cum_sum = 0
        self.cum_len_tables = []
        kept_symbols = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                table = ticker.history(
                    period=period, interval=interval, actions=False
                ).reset_index()[columns]
            except Exception as e:
                warn(f"{e} exception when processing symbol {symbol}")
                continue
            if table.shape[0] < seq_len:
                continue
            kept_symbols.append(symbol)
            cum_sum += table.shape[0] - seq_len + 1
            self.cum_len_tables.append(cum_sum)
            table.to_csv(os.path.join(dir, f"{symbol}.csv"))
        self.symbols = kept_symbols

    def __len__(self):
        return self.cum_len_tables[-1]

    def __getitem__(self, idx):
        table_idx = bisect_left(self.cum_len_tables, idx)
        assert table_idx < len(self.cum_len_tables)
        row_idx = idx - self.cum_len_tables[table_idx - 1] if table_idx > 0 else idx
        symbol = self.symbols[table_idx]
        table = pd.read_csv(
            os.path.join(self.dir, f"{symbol}.csv"),
            skiprows=row_idx,
            nrows=self.seq_len,
            header=0,
            names=self.columns,
        )
        table = table.drop(columns="Datetime")
        tensor = torch.tensor(table.values, dtype=torch.float32)
        return tensor
