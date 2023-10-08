# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/100_utils.ipynb.

# %% auto 0
__all__ = ['get_memory_usage', 'download_file', 'check_fgm', 'col_renamer', 'df2ts', 'sat_get_fgm_from_df', 'juno_get_state',
           'calc_vec_mag', 'calc_time_diff']

# %% ../nbs/100_utils.ipynb 3
import os
import requests

import polars as pl
import pandas as pd
import xarray as xr

import pandas
import numpy as np
from xarray_einstats import linalg

from datetime import timedelta

from loguru import logger
from multipledispatch import dispatch

from xarray import DataArray
from typing import Union
from typing import Any, Collection

# %% ../nbs/100_utils.ipynb 4
from humanize import naturalsize

# %% ../nbs/100_utils.ipynb 5
def get_memory_usage(data):
    datatype = type(data)
    match datatype:
        case pl.DataFrame:
            size = data.estimated_size()
        case pd.DataFrame:
            size = data.memory_usage().sum()
        case xr.DataArray:
            size = data.nbytes

    logger.info(f"{naturalsize(size)} ({datatype.__name__})")
    return size

# %% ../nbs/100_utils.ipynb 6
def download_file(url, local_dir="./", file_name=None):
    """
    Download a file from a URL and save it locally.

    Returns:
    file_path (str): Path to the downloaded file.
    """
    if file_name is None:
        file_name = url.split("/")[-1]

    file_path = os.path.join(local_dir, file_name)
    dir = os.path.dirname(file_path)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    if not os.path.exists(file_path):
        logger.debug(f"Downloading from {url}")
        response = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(response.content)
    return file_path

def check_fgm(vec):
    # check if time is monotonic increasing
    logger.info("Check if time is monotonic increasing")
    assert vec.time.to_series().is_monotonic_increasing
    # check available time difference
    logger.info(
        f"Available time delta: {vec.time.diff(dim='time').to_series().unique()}"
    )
    # data_array.time.diff(dim="time").plot(yscale="log")


def col_renamer(lbl: str):
    if lbl.startswith("BX"):
        return "BX"
    if lbl.startswith("BY"):
        return "BY"
    if lbl.startswith("BZ"):
        return "BZ"
    return lbl


def df2ts(
    df: Union[pandas.DataFrame, pl.DataFrame, pl.LazyFrame], cols, attrs, name=None
):
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Expected column {col} not found in the dataframe.")

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # Prepare data
    data = df[cols]

    # Prepare coordinates
    time = df.index if isinstance(df, pandas.DataFrame) else df["time"]

    # Create the DataArray
    coords = {"time": time, "v_dim": cols}

    return xr.DataArray(data, coords=coords, attrs=attrs, name=name)


def sat_get_fgm_from_df(df: Union[pandas.DataFrame, pl.DataFrame, pl.LazyFrame]):
    attrs = {"coordinate_system": "se", "units": "nT"}

    return df2ts(df, cols=["BX", "BY", "BZ"], attrs=attrs, name="sat_fgm")


def juno_get_state(df: Union[pandas.DataFrame, pl.DataFrame, pl.LazyFrame]):
    attrs = {"coordinate_system": "se", "units": "km"}
    return df2ts(df, cols=["X", "Y", "Z"], attrs=attrs, name="sat_state")


def calc_vec_mag(vec) -> DataArray:
    return linalg.norm(vec, dims="v_dim")

# %% ../nbs/100_utils.ipynb 7
@dispatch(pl.DataFrame)
def calc_time_diff(data: pl.DataFrame): 
    return data.get_column('time').diff(null_behavior="drop").unique().sort()

@dispatch(pl.LazyFrame)
def calc_time_diff(
    data: pl.LazyFrame
) -> pl.Series: 
    return calc_time_diff(data.collect())
