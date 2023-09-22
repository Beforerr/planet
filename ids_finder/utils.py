# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/100_utils.ipynb.

# %% auto 0
__all__ = ['download_file', 'check_fgm', 'col_renamer', 'df2ts', 'sat_get_fgm_from_df', 'juno_get_state', 'calc_vec_mag',
           'calc_vec_mean_mag', 'calc_vec_std', 'calc_vec_relative_diff', 'pl_format_time', 'pl_norm', 'pl_dvec',
           'compute_std', 'compute_combinded_std', 'compute_index_std', 'compute_index_diff', 'compute_indices']

# %% ../nbs/100_utils.ipynb 1
import os
import requests

import polars as pl
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


# %% ../nbs/100_utils.ipynb 2
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

# %% ../nbs/100_utils.ipynb 3
def calc_vec_mean_mag(vec: DataArray):
    return linalg.norm(vec, dims="v_dim").mean(dim="time")


def calc_vec_std(vec: DataArray):
    """
    Computes the standard deviation of a vector.
    """
    return linalg.norm(vec.std(dim="time"), dims="v_dim")


def calc_vec_relative_diff(vec: DataArray):
    """
    Computes the relative difference between the last and first elements of a vector.
    """
    dvec = vec.isel(time=-1) - vec.isel(time=0)
    return linalg.norm(dvec, dims="v_dim") / linalg.norm(vec, dims="v_dim").mean(
        dim="time"
    )

def _expand_selectors(items: Any, *more_items: Any) -> list[Any]:
    """
    See `_expand_selectors` in `polars`.
    """
    expanded: list[Any] = []
    for item in (
        *(
            items
            if isinstance(items, Collection) and not isinstance(items, str)
            else [items]
        ),
        *more_items,
    ):
        expanded.append(item)
    return expanded


# some helper functions
def pl_format_time(tau):
    return [
        pl.col("time").alias("tstart"),
        (pl.col("time") + tau).dt.cast_time_unit("ns").alias("tstop"),
        (pl.col("time") + tau / 2).dt.cast_time_unit("ns"),
    ]


def pl_norm(columns, *more_columns) -> pl.Expr:
    """
    Computes the square root of the sum of squares for the given columns.

    Args:
    *columns (str): Names of the columns.

    Returns:
    pl.Expr: Expression representing the square root of the sum of squares.
    """
    all_columns = _expand_selectors(columns, *more_columns)
    squares = [pl.col(column).pow(2) for column in all_columns]

    return sum(squares).sqrt()


def pl_dvec(columns, *more_columns):
    all_columns = _expand_selectors(columns, *more_columns)
    return [
        (pl.col(column).first() - pl.col(column).last()).alias(f"d{column}_vec")
        for column in all_columns
    ]



# %% ../nbs/100_utils.ipynb 4
def compute_std(df: pl.DataFrame, tau) -> pl.DataFrame:
    b_cols = ["BX", "BY", "BZ"]
    b_std_cols = [col_name + "_std" for col_name in b_cols]

    std_df = (
        df.group_by_dynamic("time", every=tau / 2, period=tau)
        .agg(
            pl.col(b_cols).std(ddof=0).map_alias(lambda col_name: col_name + "_std"),
        )
        .with_columns(
            pl_norm(b_std_cols).alias("B_std"),
        )
        .drop(b_std_cols)
    )
    return std_df


def compute_combinded_std(df: pl.DataFrame, tau) -> pl.DataFrame:
    b_cols = ["BX", "BY", "BZ"]
    b_combined_std_cols = [col_name + "_combined_std" for col_name in b_cols]
    offsets = [0 * tau, tau / 2]
    combined_std_dfs = []
    for offset in offsets:
        truncated_df = df.select(
            (pl.col("time") - offset).dt.truncate(tau, offset=offset).alias("time"),
            pl.col(b_cols),
        )

        prev_df = truncated_df.select(
            (pl.col("time") + tau).dt.cast_time_unit("ns"),
            pl.col(b_cols),
        )

        next_df = truncated_df.select(
            (pl.col("time") - tau).dt.cast_time_unit("ns"),
            pl.col(b_cols),
        )

        temp_combined_std_df = (
            pl.concat([prev_df, next_df])
            .group_by("time")
            .agg(
                pl.col(b_cols)
                .std(ddof=0)
                .map_alias(lambda col_name: col_name + "_combined_std"),
            )
            .with_columns(pl_norm(b_combined_std_cols).alias("B_combined_std"))
            .drop(b_combined_std_cols)
            .sort("time")
        )

        combined_std_dfs.append(temp_combined_std_df)

    combined_std_df = pl.concat(combined_std_dfs)
    return combined_std_df

# %% ../nbs/100_utils.ipynb 5
@dispatch(xr.DataArray, object)
def compute_index_std(data: DataArray, tau):
    """
    Examples
    --------
    >>> i1 = index_std(juno_fgm_b, tau)
    """

    # NOTE: large tau values will speed up the computation

    # Resample the data based on the provided time interval.
    grouped_data = data.resample(time=pandas.Timedelta(tau, unit="s"))

    # Compute the standard deviation for all groups
    vec_stds = linalg.norm(grouped_data.std(dim="time"), dims="v_dim")
    # vec_stds = grouped_data.map(calc_vec_std) # NOTE: This is way much slower (like 30x slower)

    offset = pandas.Timedelta(tau / 2, unit="s")
    vec_stds["time"] = vec_stds["time"] + offset

    vec_stds_next = vec_stds.assign_coords(
        {"time": vec_stds["time"] - pandas.Timedelta(tau, unit="s")}
    )
    vec_stds_previous = vec_stds.assign_coords(
        {"time": vec_stds["time"] + pandas.Timedelta(tau, unit="s")}
    )
    return np.minimum(vec_stds / vec_stds_next, vec_stds / vec_stds_previous)

@dispatch(pl.LazyFrame, object)
def compute_index_std(df: pl.LazyFrame, tau, join_strategy="inner"):  # noqa: F811
    """
    Compute the standard deviation index based on the given DataFrame and tau value.

    Parameters
    ----------
    - df (pl.LazyFrame): The input DataFrame.
    - tau (int): The time interval value.

    Returns
    -------
    - pl.LazyFrame: DataFrame with calculated 'index_std' column.

    Examples
    --------
    >>> index_std_df = compute_index_std_pl(df, tau)
    >>> index_std_df

    Notes
    -----
    Simply shift to calculate index_std would not work correctly if data is missing, like `std_next = pl.col("B_std").shift(-2)`.

    """

    if isinstance(tau, (int, float)):
        tau = timedelta(seconds=tau)

    if "B_std" in df.columns:
        std_df = df
    else:
        # Compute standard deviations
        std_df = compute_std(df, tau)

    # Calculate the standard deviation index
    prev_std_df = std_df.select(
        (pl.col("time") + tau).dt.cast_time_unit("ns"),
        (pl.col("B_std")).alias("B_std_prev"),
    )

    next_std_df = std_df.select(
        (pl.col("time") - tau).dt.cast_time_unit("ns"),
        (pl.col("B_std")).alias("B_std_next"),
    )

    index_std_df = (
        std_df.join(prev_std_df, on="time", how=join_strategy)
        .join(next_std_df, on="time", how=join_strategy)
        .with_columns(
            (pl.col("B_std") / (pl.max_horizontal("B_std_prev", "B_std_next"))).alias(
                "index_std"
            )
        )
    )
    return index_std_df

# %% ../nbs/100_utils.ipynb 6
def compute_index_diff(df, tau):
    b_cols = ["BX", "BY", "BZ"]
    db_cols = ["d" + col_name + "_vec" for col_name in b_cols]

    index_diff = (
        df.with_columns(pl_norm(b_cols).alias("B"))
        .group_by_dynamic("time", every=tau / 2, period=tau)
        .agg(
            pl.count(),
            pl.col("B").mean().alias("B_mean"),
            *pl_dvec(b_cols),
        )
        .with_columns(
            pl_norm(db_cols).alias("dB_vec"),
        )
        .with_columns(
            (pl.col("dB_vec") / pl.col("B_mean")).alias("index_diff"),
        )
    )

    return index_diff


@dispatch(pl.LazyFrame, timedelta)
def compute_indices(
    df: pl.LazyFrame, 
    tau
) -> pl.LazyFrame:
    join_strategy = "inner"
    std_df = compute_std(df, tau)
    combined_std_df = compute_combinded_std(df, tau)

    index_std = compute_index_std(std_df, tau)
    index_diff = compute_index_diff(df, tau)

    indices = (
        index_std.join(index_diff, on="time")
        .join(combined_std_df, on="time", how=join_strategy)
        .with_columns(
            pl.sum_horizontal("B_std_prev", "B_std_next").alias("B_added_std"),
        )
        .with_columns(
            (pl.col("B_std") / (pl.max_horizontal("B_std_prev", "B_std_next"))).alias(
                "index_std"
            ),
            (pl.col("B_combined_std") / pl.col("B_added_std")).alias(
                "index_fluctuation"
            ),
        )
    )

    return indices


@dispatch(pl.DataFrame, timedelta)
def compute_indices(    # noqa: F811
    df: pl.DataFrame, 
    tau: timedelta,
) -> pl.LazyFrame | pl.DataFrame:  
    """
    Compute all index based on the given DataFrame and tau value.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    tau : datetime.timedelta
        Time interval value.

    Returns
    -------
    tuple : 
        Tuple containing DataFrame results for fluctuation index, 
        standard deviation index, and 'index_num'.

    Examples
    --------
    >>> indices = compute_indices(df, tau)

    Notes
    -----
    - Simply shift to calculate index_std would not work correctly if data is missing, 
        like `std_next = pl.col("B_std").shift(-2)`.
    - Drop null though may lose some IDs (using the default `join_strategy`). 
        Because we could not tell if it is a real ID or just a partial wave 
        from incomplete data without previous or/and next std. 
        Hopefully we can pick up the lost ones with smaller tau.
    - TODO: Can be optimized further, but this is already fast enough.
        - TEST: if `join` can be improved by shift after filling the missing values.
        - TEST: if `list` in `polars` really fast?
    """
    return compute_indices(df.lazy(), tau).collect()


