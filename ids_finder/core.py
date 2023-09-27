# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_ids_finder.ipynb.

# %% auto 0
__all__ = ['THRESHOLD_RATIO', 'BnOverB_RD_lower_threshold', 'dBOverB_RD_upper_threshold', 'BnOverB_TD_upper_threshold',
           'dBOverB_TD_lower_threshold', 'BnOverB_ED_upper_threshold', 'dBOverB_ED_upper_threshold',
           'BnOverB_ND_lower_threshold', 'dBOverB_ND_lower_threshold', 'get_candidate_data', 'get_candidates',
           'time_stamp', 'plot_basic', 'format_candidate_title', 'plot_candidate', 'plot_candidates', 'calc_duration',
           'calc_d_duration', 'find_start_end_times', 'get_time_from_condition', 'calc_candidate_duration',
           'calc_candidate_d_duration', 'calibrate_candidate_duration', 'calc_classification_index', 'classify_id',
           'calc_rotation_angle', 'calc_candidate_rotation_angle', 'get_candidate_location', 'get_ID_filter_condition',
           'calc_candidate_classification_index', 'convert_to_dataframe', 'IDsPipeline', 'process_candidates',
           'CandidateID']

# %% ../nbs/00_ids_finder.ipynb 3
#| code-summary: import all the packages needed for the project
from fastcore.utils import *
from fastcore.test import *
from .utils import *
import polars as pl
import xarray as xr

try:
    import modin.pandas as pd
    import modin.pandas as mpd
    from modin.config import ProgressBar
    ProgressBar.enable()
except ImportError:
    import pandas as pd

import pandas
    
import numpy as np
from xarray_einstats import linalg

from datetime import timedelta

from loguru import logger

import pytplot

import pdpipe as pdp
from multipledispatch import dispatch


from xarray.core.dataarray import DataArray

# %% ../nbs/00_ids_finder.ipynb 10
@dispatch(object, xr.DataArray)
def get_candidate_data(candidate, data, coord:str=None, neighbor:int=0) -> xr.DataArray:
    duration = candidate['tstop'] - candidate['tstart']
    offset = neighbor*duration
    temp_tstart = candidate['tstart'] - offset
    temp_tstop = candidate['tstop'] + offset
    
    return data.sel(time=slice(temp_tstart,  temp_tstop))

@dispatch(object, pl.DataFrame)
def get_candidate_data(candidate, data, coord:str=None, neighbor:int=0) -> xr.DataArray:
    """
    Notes
    -----
    much slower than `get_candidate_data_xr`
    """
    duration = candidate['tstart'] - candidate['tstop']
    offset = neighbor*duration
    temp_tstart = candidate['tstart'] - offset
    temp_tstop = candidate['tstop'] + offset
    
    temp_data = data.filter(
        pl.col("time").is_between(temp_tstart, temp_tstop)
    )
    
    return df2ts(temp_data, ["BX", "BY", "BZ"], attrs={"coordinate_system": coord, "units": "nT"})

def get_candidates(candidates: pd.DataFrame, candidate_type=None, num:int=4):
    
    if candidate_type is not None:
        _candidates = candidates[candidates['type'] == candidate_type]
    else:
        _candidates = candidates
    
    # Sample a specific number of candidates if num is provided and it's less than the total number
    if num < len(_candidates):
        logger.info(f"Sampling {num} {candidate_type} candidates out of {len(_candidates)}")
        return _candidates.sample(num)
    else:
        return _candidates

# %% ../nbs/00_ids_finder.ipynb 12
from zoneinfo import ZoneInfo
from pyspedas.cotrans.minvar_matrix_make import minvar_matrix_make
from pyspedas import tvector_rotate
from pytplot import timebar, store_data, tplot, split_vec, join_vec, tplot_options, options, highlight, degap

# %% ../nbs/00_ids_finder.ipynb 13
def time_stamp(ts):
    "Return POSIX timestamp as float."
    return pd.Timestamp(ts, tz="UTC").timestamp()

def plot_basic(
    data: xr.DataArray, 
    tstart: pd.Timestamp, 
    tstop: pd.Timestamp,
    tau: timedelta, 
    mva_tstart=None, mva_tstop=None, neighbor: int = 1
):
    if mva_tstart is None:
        mva_tstart = tstart
    if mva_tstop is None:
        mva_tstop = tstop

    mva_b = data.sel(time=slice(mva_tstart, mva_tstop))
    store_data("fgm", data={"x": mva_b.time, "y": mva_b})
    minvar_matrix_make("fgm")  # get the MVA matrix

    temp_tstart = tstart - neighbor * tau
    temp_tstop = tstop + neighbor * tau

    temp_b = data.sel(time=slice(temp_tstart, temp_tstop))
    store_data("fgm", data={"x": temp_b.time, "y": temp_b})
    temp_btotal = calc_vec_mag(temp_b)
    store_data("fgm_btotal", data={"x": temp_btotal.time, "y": temp_btotal})

    tvector_rotate("fgm_mva_mat", "fgm")
    split_vec("fgm_rot")
    pytplot.data_quants["fgm_btotal"]["time"] = pytplot.data_quants["fgm_rot"][
        "time"
    ]  # NOTE: whenever using `get_data`, we may lose precision in the time values. This is a workaround.
    join_vec(
        [
            "fgm_rot_x",
            "fgm_rot_y",
            "fgm_rot_z",
            "fgm_btotal",
        ],
        new_tvar="fgm_all",
    )

    options("fgm", "legend_names", [r"$B_x$", r"$B_y$", r"$B_z$"])
    options("fgm_all", "legend_names", [r"$B_l$", r"$B_m$", r"$B_n$", r"$B_{total}$"])
    options("fgm_all", "ysubtitle", "[nT LMN]")
    tstart_ts = time_stamp(tstart)
    tstop_ts = time_stamp(tstop)
    # .replace(tzinfo=ZoneInfo('UTC')).timestamp()
    highlight(["fgm", "fgm_all"], [tstart_ts, tstop_ts])
    
    degap("fgm")
    degap("fgm_all")

def format_candidate_title(candidate: pandas.Series):
    format_float = lambda x: rf"$\bf {x:.2f} $" if isinstance(x, (float, int)) else rf"$\bf {x} $"

    base_line = rf'$\bf {candidate.get("type", "N/A")} $ candidate (time: {candidate.get("time", "N/A")}) with index '
    index_line = rf'i1: {format_float(candidate.get("index_std", "N/A"))}, i2: {format_float(candidate.get("index_fluctuation", "N/A"))}, i3: {format_float(candidate.get("index_diff", "N/A"))}'
    info_line = rf'$B_n/B$: {format_float(candidate.get("BnOverB", "N/A"))}, $dB/B$: {format_float(candidate.get("dBOverB", "N/A"))}, $(dB/B)_{{max}}$: {format_float(candidate.get("dBOverB_max", "N/A"))},  $Q_{{mva}}$: {format_float(candidate.get("Q_mva", "N/A"))}'
    title = rf"""{base_line}
    {index_line}
    {info_line}"""
    return title


def plot_candidate(candidate: pandas.Series, sat_fgm: xr.DataArray, tau: timedelta):
    if pd.notnull(candidate.get("d_tstart")) and pd.notnull(candidate.get("d_tstop")):
        plot_basic(
            sat_fgm,
            candidate["tstart"],
            candidate["tstop"],
            tau,
            candidate["d_tstart"],
            candidate["d_tstop"],
        )
    else:
        plot_basic(sat_fgm, candidate["tstart"], candidate["tstop"], tau)

    tplot_options("title", format_candidate_title(candidate))

    if "d_time" in candidate.keys():
        d_time_ts = time_stamp(candidate["d_time"])
        timebar(d_time_ts, color="red")
    if "d_tstart" in candidate.keys() and not pd.isnull(candidate["d_tstart"]):
        d_start_ts = time_stamp(candidate["d_tstart"])
        timebar(d_start_ts)
    if "d_tstop" in candidate.keys() and not pd.isnull(candidate["d_tstop"]):
        d_stop_ts = time_stamp(candidate["d_tstop"])
        timebar(d_stop_ts)

    # tplot(['fgm','fgm_all'])
    tplot("fgm_all")


def plot_candidates(
    candidates: pandas.DataFrame, candidate_type=None, num=4, plot_func=plot_candidate
):
    """Plot a set of candidates.

    Parameters:
    - candidates (pd.DataFrame): DataFrame containing the candidates.
    - candidate_type (str, optional): Filter candidates based on a specific type.
    - num (int): Number of candidates to plot, selected randomly.
    - plot_func (callable): Function used to plot an individual candidate.

    """

    # Filter by candidate_type if provided
    
    candidates = get_candidates(candidates, candidate_type, num)

    # Plot each candidate using the provided plotting function
    for _, candidate in candidates.iterrows():
        plot_func(candidate)

# %% ../nbs/00_ids_finder.ipynb 17
THRESHOLD_RATIO  = 1/4

from typing import Tuple

def calc_duration(vec: xr.DataArray, threshold_ratio=THRESHOLD_RATIO) -> pandas.Series:
    # NOTE: gradient calculated at the edge is not reliable.
    vec_diff = vec.differentiate("time", datetime_unit="s").isel(time=slice(1,-1))
    vec_diff_mag = linalg.norm(vec_diff, dims='v_dim')

    # Determine d_star based on trend
    if vec_diff_mag.isnull().all():
        raise ValueError("The differentiated vector magnitude contains only NaN values. Cannot compute duration.")
    
    d_star_index = vec_diff_mag.argmax(dim="time")
    d_star = vec_diff_mag[d_star_index]
    d_time = vec_diff_mag.time[d_star_index]
    
    threshold = d_star * threshold_ratio

    start_time, end_time = find_start_end_times(vec_diff_mag, d_time, threshold)

    dict = {
        'd_star': d_star.item(),
        'd_time': d_time.values,
        'threshold': threshold.item(),
        'd_tstart': start_time,
        'd_tstop': end_time,
    }

    return pandas.Series(dict)

def calc_d_duration(vec: xr.DataArray, d_time, threshold) -> pd.Series:
    vec_diff = vec.differentiate("time", datetime_unit="s")
    vec_diff_mag = linalg.norm(vec_diff, dims='v_dim')

    start_time, end_time = find_start_end_times(vec_diff_mag, d_time, threshold)

    return pandas.Series({
        'd_tstart': start_time,
        'd_tstop': end_time,
    })
 
def find_start_end_times(vec_diff_mag: xr.DataArray, d_time, threshold) -> Tuple[pd.Timestamp, pd.Timestamp]:
    # Determine start time
    pre_vec_mag = vec_diff_mag.sel(time=slice(None, d_time))
    start_time = get_time_from_condition(pre_vec_mag, threshold, "last_below")

    # Determine stop time
    post_vec_mag = vec_diff_mag.sel(time=slice(d_time, None))
    end_time = get_time_from_condition(post_vec_mag, threshold, "first_below")

    return start_time, end_time


def get_time_from_condition(vec: xr.DataArray, threshold, condition_type) -> pd.Timestamp:
    if condition_type == "first_below":
        condition = vec < threshold
        index_choice = 0
    elif condition_type == "last_below":
        condition = vec < threshold
        index_choice = -1
    else:
        raise ValueError(f"Unknown condition_type: {condition_type}")

    where_result = np.where(condition)[0]

    if len(where_result) > 0:
        return vec.time[where_result[index_choice]].values
    return None

# %% ../nbs/00_ids_finder.ipynb 18
def calc_candidate_duration(candidate: pd.Series, data) -> pd.Series:
    try:
        candidate_data = get_candidate_data(candidate, data)
        return calc_duration(candidate_data)
    except Exception as e:
        # logger.debug(f"Error for candidate {candidate} at {candidate['time']}: {str(e)}") # can not be serialized
        print(f"Error for candidate {candidate} at {candidate['time']}: {str(e)}")
        raise e

def calc_candidate_d_duration(candidate, data) -> pd.Series:
    try:
        if pd.isnull(candidate['d_tstart']) or pd.isnull(candidate['d_tstop']):
            candidate_data = get_candidate_data(candidate, data, neighbor=1)
            d_time = candidate['d_time']
            threshold = candidate['threshold']
            return calc_d_duration(candidate_data, d_time, threshold)
        else:
            return pandas.Series({
                'd_tstart': candidate['d_tstart'],
                'd_tstop': candidate['d_tstop'],
            })
    except Exception as e:
        # logger.debug(f"Error for candidate {candidate} at {candidate['time']}: {str(e)}")
        print(f"Error for candidate {candidate} at {candidate['time']}: {str(e)}")
        raise e

# %% ../nbs/00_ids_finder.ipynb 19
def calibrate_candidate_duration(
    candidate: pd.Series, data:xr.DataArray, data_resolution, ratio = 3/4
):
    """
    Calibrates the candidate duration. 
    - If only one of 'd_tstart' or 'd_tstop' is provided, calculates the missing one based on the provided one and 'd_time'.
    - Then if this is not enough points between 'd_tstart' and 'd_tstop', returns None for both.
    
    
    Parameters
    ----------
    - candidate (pd.Series): The input candidate with potential missing 'd_tstart' or 'd_tstop'.
    
    Returns
    -------
    - pd.Series: The calibrated candidate.
    """
    
    start_notnull = pd.notnull(candidate['d_tstart'])
    stop_notnull = pd.notnull(candidate['d_tstop']) 
    
    match start_notnull, stop_notnull:
        case (True, True):
            d_tstart = candidate['d_tstart']
            d_tstop = candidate['d_tstop']
        case (True, False):
            d_tstart = candidate['d_tstart']
            d_tstop = candidate['d_time'] -  candidate['d_tstart'] + candidate['d_time']
        case (False, True):
            d_tstart = candidate['d_time'] -  candidate['d_tstop'] + candidate['d_time']
            d_tstop = candidate['d_tstop']
        case (False, False):
            return pandas.Series({
                'd_tstart': None,
                'd_tstop': None,
            })
    
    duration = d_tstop - d_tstart
    num_of_points_between = data.time.sel(time=slice(d_tstart, d_tstop)).count().item()
    
    if num_of_points_between <= (duration/data_resolution) * ratio:
        d_tstart = None
        d_tstop = None
    
    return pandas.Series({
        'd_tstart': d_tstart,
        'd_tstop': d_tstop,
    })

# %% ../nbs/00_ids_finder.ipynb 22
BnOverB_RD_lower_threshold = 0.4
dBOverB_RD_upper_threshold = 0.2

BnOverB_TD_upper_threshold = 0.2
dBOverB_TD_lower_threshold = dBOverB_RD_upper_threshold

BnOverB_ED_upper_threshold = BnOverB_RD_lower_threshold
dBOverB_ED_upper_threshold = dBOverB_TD_lower_threshold

BnOverB_ND_lower_threshold = BnOverB_TD_upper_threshold
dBOverB_ND_lower_threshold = dBOverB_RD_upper_threshold

# %% ../nbs/00_ids_finder.ipynb 23
from pyspedas.cotrans.minvar import minvar

# %% ../nbs/00_ids_finder.ipynb 24
def calc_classification_index(data: xr.DataArray):
    
    vrot, v, w = minvar(data.to_numpy()) # NOTE: using `.to_numpy()` will significantly speed up the computation.
    Vl = v[:,0] # Maximum variance direction eigenvector

    B_rot = xr.DataArray(vrot, dims=['time', 'v_dim'], coords={'time': data.time})
    B = calc_vec_mag(B_rot)
    B_n = B_rot.isel(v_dim=2)
    
    B_mean = B.mean(dim="time")
    B_n_mean = B_n.mean(dim="time")
    
    BnOverB = B_n_mean / B_mean
    # BnOverB = np.abs(B_n / B).mean(dim="time")

    dB = B.isel(time=-1) - B.isel(time=0)
    dBOverB = np.abs(dB / B_mean)
    dBOverB_max = (B.max(dim="time") - B.min(dim="time")) / B_mean
    
    
    return pandas.Series({
        'Vl_x': Vl[0],
        'Vl_y': Vl[1],
        'Vl_z': Vl[2],
        'eig0': w[0],
        'eig1': w[1],
        'eig2': w[2],
        'Q_mva': w[1]/w[2],
        'B': B_mean.item(),
        'B_n': B_n_mean.item(),
        'dB': dB.item(),
        'BnOverB': BnOverB.item(), 
        'dBOverB': dBOverB.item(),
        'dBOverB_max': dBOverB_max.item(),
        })

# %% ../nbs/00_ids_finder.ipynb 25
def classify_id(BnOverB, dBOverB):
    BnOverB = np.abs(np.asarray(BnOverB))
    dBOverB = np.asarray(dBOverB)

    s1 = (BnOverB > BnOverB_RD_lower_threshold)
    s2 = (dBOverB > dBOverB_RD_upper_threshold)
    s3 = (BnOverB > BnOverB_TD_upper_threshold)
    s4 = s2 # note: s4 = (dBOverB > dBOverB_TD_lower_threshold)
    
    RD = s1 & ~s2
    TD = ~s3 & s4
    ED = ~s1 & ~s4
    ND = s3 & s2

    # Create an empty result array with the same shape
    result = np.empty_like(BnOverB, dtype=object)

    result[RD] = "RD"
    result[TD] = "TD"
    result[ED] = "ED"
    result[ND] = "ND"

    return result

# %% ../nbs/00_ids_finder.ipynb 27
def calc_rotation_angle(v1, v2):
    """
    Computes the rotation angle between two vectors.
    
    Parameters:
    - v1: The first vector.
    - v2: The second vector.
    """
    
    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same shape.")

    # convert xr.Dataarray to numpy arrays
    if isinstance(v1, DataArray):
        v1 = v1.to_numpy()
    if isinstance(v2, DataArray):
        v2 = v2.to_numpy()
    
    # Normalize the vectors
    v1_u = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    v2_u = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)
    
    # Calculate the cosine of the angle for each time step
    cosine_angle = np.sum(v1_u * v2_u, axis=-1)
    
    # Clip the values to handle potential floating point errors
    cosine_angle = np.clip(cosine_angle, -1, 1)
    
    angle = np.arccos(cosine_angle)
    
    # Convert the angles from radians to degrees
    return np.degrees(angle)

def calc_candidate_rotation_angle(candidates, data:  xr.DataArray):
    """
    Computes the rotation angle(s) at two different time steps.
    """
    
    tstart = candidates['d_tstart']
    tstop = candidates['d_tstop']
    
    # Convert Series to numpy arrays if necessary
    if isinstance(tstart, pd.Series):
        tstart = tstart.to_numpy()
        tstop = tstop.to_numpy()
        # no need to Handle NaT values (as `calibrate_candidate_duration` will handle this)
    
    # Get the vectors at the two time steps
    vecs_before = data.sel(time=tstart, method="nearest")
    vecs_after = data.sel(time=tstop, method="nearest")
    
    # Compute the rotation angle(s)
    rotation_angles = calc_rotation_angle(vecs_before, vecs_after)
    return rotation_angles

# %% ../nbs/00_ids_finder.ipynb 29
def get_candidate_location(candidate, location_data: DataArray):
    return location_data.sel(time = candidate['d_time'], method="nearest").to_series()

# %% ../nbs/00_ids_finder.ipynb 31
def get_ID_filter_condition(
    index_std_threshold = 2,
    index_fluc_threshold = 1,
    index_diff_threshold = 0.1,
    sparse_num = 15
):
    return (
        (pl.col("index_std") > index_std_threshold)
        & (pl.col("index_fluctuation") > index_fluc_threshold)
        & (pl.col("index_diff") > index_diff_threshold)
        & (
            pl.col("index_std").is_finite()
        )  # for cases where neighboring groups have std=0
        & (
            pl.col("count") > sparse_num
        )  # filter out sparse intervals, which may give unreasonable results.
        & (
            pl.col("count_prev") > sparse_num
        ) 
        & (
            pl.col("count_next") > sparse_num
        )
    )


# %% ../nbs/00_ids_finder.ipynb 32
from pdpipe.util import out_of_place_col_insert

# %% ../nbs/00_ids_finder.ipynb 33
#| code-summary: patch `pdp.ApplyToRows` to work with `modin` DataFrames
@patch
def _transform(self: pdp.ApplyToRows, X, verbose):
    new_cols = X.apply(self._func, axis=1)
    if isinstance(new_cols, (pd.Series, pandas.Series)):
        loc = len(X.columns)
        if self._follow_column:
            loc = X.columns.get_loc(self._follow_column) + 1
        return out_of_place_col_insert(
            X=X, series=new_cols, loc=loc, column_name=self._colname
        )
    if isinstance(new_cols, (pd.DataFrame, pandas.DataFrame)):
        sorted_cols = sorted(list(new_cols.columns))
        new_cols = new_cols[sorted_cols]
        if self._follow_column:
            inter_X = X
            loc = X.columns.get_loc(self._follow_column) + 1
            for colname in new_cols.columns:
                inter_X = out_of_place_col_insert(
                    X=inter_X,
                    series=new_cols[colname],
                    loc=loc,
                    column_name=colname,
                )
                loc += 1
            return inter_X
        assign_map = {
            colname: new_cols[colname] for colname in new_cols.columns
        }
        return X.assign(**assign_map)
    raise TypeError(  # pragma: no cover
        "Unexpected type generated by applying a function to a DataFrame."
        " Only Series and DataFrame are allowed."
    )

# %% ../nbs/00_ids_finder.ipynb 34
def calc_candidate_classification_index(candidate, data):
    return calc_classification_index(
        data.sel(time=slice(candidate["d_tstart"], candidate["d_tstop"]))
    )

# %% ../nbs/00_ids_finder.ipynb 35
def convert_to_dataframe(
    data: pl.DataFrame, # orignal Dataframe
)->pd.DataFrame:
    "convert data into a pandas/modin DataFrame"
    if isinstance(data, pl.LazyFrame):
        data = data.collect().to_pandas(use_pyarrow_extension_array=True)
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas(use_pyarrow_extension_array=True)
    if not isinstance(data, pd.DataFrame):  # `modin` supports
        data = pd.DataFrame(data)
    return data

# %% ../nbs/00_ids_finder.ipynb 36
#| code-summary: Pipelines Class for processing IDs
class IDsPipeline:
    def __init__(self):
        pass
    # fmt: off
    def calc_duration(self, sat_fgm: xr.DataArray):
        return pdp.PdPipeline([
            pdp.ApplyToRows(
                lambda candidate: calc_candidate_duration(candidate, sat_fgm),
                func_desc="calculating duration parameters"
            ),
            pdp.ApplyToRows(
                lambda candidate: calc_candidate_d_duration(candidate, sat_fgm),
                func_desc="calculating duration parameters if needed"
            )
        ])

    def calibrate_duration(self, sat_fgm, data_resolution):
        return \
            pdp.ApplyToRows(
                lambda candidate: calibrate_candidate_duration(candidate, sat_fgm, data_resolution),
                func_desc="calibrating duration parameters if needed"
            )

    def classify_id(self, sat_fgm):
        return pdp.PdPipeline([
            pdp.ApplyToRows(
                lambda candidate: calc_candidate_classification_index(candidate, sat_fgm),
                func_desc='calculating index "q_mva", "BnOverB" and "dBOverB"'
            ),
            pdp.ColByFrameFunc(
                "type",
                lambda df: classify_id(df["BnOverB"], df["dBOverB"]),
                func_desc="classifying the type of the ID"
            ),
        ])
    
    def calc_rotation_angle(self, sat_fgm):
        return \
            pdp.ColByFrameFunc(
                "rotation_angle",
                lambda df: calc_candidate_rotation_angle(df, sat_fgm),
                func_desc="calculating rotation angle",
            ) 

    def assign_coordinates(self, sat_state: xr.DataArray):
        "NOTE: not optimized, quite slow"
        return \
            pdp.ApplyToRows(
                lambda candidate: get_candidate_location(candidate, sat_state),
                func_desc="assigning coordinates",
            )
    # fmt: on
    # ... you can add more methods as needed

# %% ../nbs/00_ids_finder.ipynb 37
def process_candidates(
    candidates: pl.DataFrame,  # potential candidates DataFrame
    sat_fgm: xr.DataArray,  # satellite FGM data
    sat_state: pl.DataFrame,  # satellite state data
    data_resolution: timedelta,  # time resolution of the data
) -> pl.DataFrame:  # processed candidates DataFrame
    id_pipelines = IDsPipeline()
    candidates = id_pipelines.calc_duration(sat_fgm).apply(candidates)

    # calibrate duration
    temp_candidates = candidates.loc[
        lambda df: df["d_tstart"].isnull() | df["d_tstop"].isnull()
    ]  # temp_candidates = candidates.query('d_tstart.isnull() | d_tstop.isnull()') # not implemented in `modin`

    if not temp_candidates.empty:
        temp_candidates_updated = id_pipelines.calibrate_duration(
            sat_fgm, data_resolution
        ).apply(temp_candidates)
        candidates.update(temp_candidates_updated)

    ids = (
        id_pipelines.classify_id(sat_fgm)
        + id_pipelines.calc_rotation_angle(sat_fgm)
    ).apply(
        candidates.dropna()  # Remove candidates with NaN values)
    )

    if isinstance(ids, mpd.DataFrame):
        ids = ids._to_pandas()
    if isinstance(ids, pandas.DataFrame):
        ids_pl = pl.DataFrame(ids)

    ids_pl = ids_pl.sort("d_time").join_asof(
        sat_state, left_on="d_time", right_on="time", strategy="nearest"
    ).drop("time_right")

    return ids_pl

# %% ../nbs/00_ids_finder.ipynb 39
from pprint import pprint

# %% ../nbs/00_ids_finder.ipynb 40
class CandidateID:
    def __init__(self, time, df: pl.DataFrame) -> None:
        self.time = pd.Timestamp(time)
        self.data = df.row(
            by_predicate=(pl.col("time") == self.time), 
            named=True
        )

    def __repr__(self) -> str:
        # return self.data.__repr__()
        pprint(self.data)
        return ''
    
    def calc_duration(self, sat_fgm):
        return calc_candidate_duration(self.data, sat_fgm)
    
    def calc_d_duration(self, sat_fgm):
        return calc_candidate_d_duration(self.data, sat_fgm)
    
    def plot(self, sat_fgm, tau):
        plot_candidate(self.data, sat_fgm, tau)
        pass
        
