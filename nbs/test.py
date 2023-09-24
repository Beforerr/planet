# %%
#| hide



# %%
#| output: hide
#| code-summary: import all the packages needed for the project

from ids_finder.utils import *
from ids_finder.core import *
from fastcore.utils import *
from fastcore.test import *

import polars as pl
try:
    import modin.pandas as pd
    import modin.pandas as mpd
except ImportError:
    import pandas as pd

import pandas
import numpy as np

from datetime import timedelta
from loguru import logger
import speasy as spz


# %% [markdown]
# ## Dataset Overview

# %%
artemis_probes = ["b", "c"]
probe = artemis_probes[0]

jno_start_date = "2011-08-25"
jno_end_date = "2016-06-30" 

trange = [jno_start_date, jno_end_date]
test_trange = ["2011-08-25", "2011-09-25"]

# %% [markdown]
# ### Download all the files

# %%
sat = 'thb'
coord = 'gse'
datatype = 'fgs'

sat_fgm_product = f'cda/{sat.upper()}_L2_FGM/{sat}_fgs_gse'
sat_pos_sse_product = f'cda/{sat.upper()}_L1_STATE/{sat}_pos_sse'
sat_pos_gse_product = f'cda/{sat.upper()}_L1_STATE/{sat}_pos_gse'

products = [
    sat_fgm_product,
    sat_pos_sse_product,
    sat_pos_gse_product
]

# %%
@threaded
def download_data(products, trange):
    logger.info("Downloading data")
    spz.get_data(products, trange, progress=True, disable_proxy=True)
    logger.info("Data downloaded")
    # spz.get_data(products, jno_start_date, jno_end_date)   

# %% [markdown]
# Download data in a background thread

# %%
%%script false --no-raise-error
#| eval: false
download_data(products, trange)

# %% [markdown]
# ### Convert data to `parquet` for faster processing

# %%
def thm_rename_col(col: str):
    if "," in col:
        col = col.split(",")[0]
    return col.split()[0].upper()

def spz2parquet(data, force=False):
    output = f"../data/{data.name}.parquet"
    if Path(output).exists() and not force:
        logger.info("Data already converted to parquet")
    else: 
        df = pandas.DataFrame(
            data.values, index=pandas.Series(data.time, name="time"), columns=data.columns
        )
        
        df.to_parquet(output)
        logger.info("Data converted to parquet successfully")

# %%
%%script false --no-raise-error

dataset = spz.get_data(products, trange)

for data in dataset:
    spz2parquet(data, force=False)

# %% [markdown]
# ## Processing the whole data

# %%
def get_thm_state(sat):
    sat_pos_sse_files = f"../data/{sat}_pos_sse.parquet"
    sat_pos_sse = pl.scan_parquet(sat_pos_sse_files).set_sorted("time")
    sat_pos_gse_files = f"../data/{sat}_pos_gse.parquet"
    sat_pos_gse = pl.scan_parquet(sat_pos_gse_files).set_sorted("time")
    sat_state = sat_pos_sse.join(sat_pos_gse, on="time", how="inner")
    return sat_state

# %%
sat = "thb"
coord = "gse"
datatype = "fgs"
tau = timedelta(seconds=60)
data_resolution = timedelta(seconds=4)

files = f"../data/{sat}_{datatype}_{coord}.parquet"
output = f"../data/{sat}_candidates_tau_{tau.seconds}.parquet"

rename_mapping = {
    "Bx FGS-D": "BX",
    "By FGS-D": "BY",
    "Bz FGS-D": "BZ",
}

data = pl.scan_parquet(files).rename(rename_mapping).unique("time").sort("time").collect()
sat_fgm = df2ts(
    data, ["BX", "BY", "BZ"], attrs={"coordinate_system": coord, "units": "nT"}
)
sat_state = get_thm_state(sat)

# indices = compute_indices(data, tau)
# # filter condition
# sparse_num = tau / data_resolution // 3
# filter_condition = get_ID_filter_condition(sparse_num = sparse_num)

# candidates_pl = indices.filter(filter_condition).with_columns(pl_format_time(tau))
# candidates = convert_to_dataframe(candidates_pl)


# %%
from humanize import naturalsize
import xarray as xr

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

get_memory_usage(candidates_pl)
get_memory_usage(candidates)
get_memory_usage(sat_fgm)

# %%
import xarray as xr
def process_candidates(
    candidates: pl.DataFrame, # potential candidates DataFrame
    sat_fgm: xr.DataArray, # satellite FGM data
    data_resolution: timedelta, # time resolution of the data
):
    id_pipelines = IDsPipeline()

    candidates = id_pipelines.calc_duration(sat_fgm).apply(candidates)

    # calibrate duration
    temp_candidates = candidates.loc[
        lambda df: df["d_tstart"].isnull() | df["d_tstop"].isnull()
    ]  # temp_candidates = candidates.query('d_tstart.isnull() | d_tstop.isnull()') # not implemented in `modin`

    if not temp_candidates.empty:
        candidates.update(
            id_pipelines.calibrate_duration(sat_fgm, data_resolution).apply(
                temp_candidates
            )
        )

    ids = (
        id_pipelines.classify_id(sat_fgm)
        + id_pipelines.calc_rotation_angle(sat_fgm)
    ).apply(
        candidates.dropna()  # Remove candidates with NaN values)
    )

    return ids

# %%
# temp_candidate = candidates[candidates.time == pd.Timestamp('2014-03-08 02:30:30')].iloc[0]
temp_candidate = candidates.sample().iloc[0]


# %%
tstart = temp_candidate["tstart"].strftime("%Y-%m-%d %H:%M:%S")
tstop = temp_candidate["tstop"].strftime("%Y-%m-%d %H:%M:%S")
sat_fgm.sel(time=slice(tstart, tstop))

# %%
plot_candidate(temp_candidate, sat_fgm, tau)

# %%
ids = process_candidates(candidates, sat_fgm, data_resolution)


# %%
pl.scan_parquet(convert_thm_state_to_parquet(probe, trange)).filter(
    (pl.col("thm_pos_gse_X") > 0) & (pl.col("thm_pos_sse_X") > 0)
).collect().shape

# %% [markdown]
# ## Obsolete codes

# %%
#| eval: false
import pycdfpp
import pyspedas

# %%
#| eval: false

def convert_thm_state_to_parquet(
    probe: str, trange
):
    file_name = f"./data/th{probe}_state.parquet"
    if os.path.exists(file_name):
        return file_name

    start = trange.start.to_string()
    end = trange.end.to_string()

    files = pyspedas.themis.state(
        probe=probe,
        trange=[start, end],
        downloadonly=True,
        no_update=True,
    )

    thm_pos_sse_Xs = []
    thm_pos_gse_Xs = []
    thm_state_times = []
    for file in files:
        thm_state = pycdfpp.load(file)
        epoch_dt64 = thm_state[
            f"time"
        ].values  #  CATDESC: "thm_state_time, UTC, in seconds since 01-Jan-1970 00:00:00"
        thm_pos_sse_Xs.append(thm_state[f"th{probe}_pos_sse"].values[:, 0])
        thm_pos_gse_Xs.append(thm_state[f"th{probe}_pos_gse"].values[:, 0])
        thm_state_times.append(epoch_dt64)

    thm_pos_sse_X = np.concatenate(thm_pos_sse_Xs)
    thm_pos_gse_X = np.concatenate(thm_pos_gse_Xs)
    thm_state_time = np.concatenate(thm_state_times)

    pl.DataFrame(
        {
            "thm_state_time": thm_state_time,
            "thm_pos_gse_X": thm_pos_gse_X,
            "thm_pos_sse_X": thm_pos_sse_X,
        }
    ).with_columns(
        pl.from_epoch(pl.col("thm_state_time"), time_unit="s")
    ).write_parquet(
        file_name
    )

    return file_name


def convert_thm_fgm_to_parquet(probe, trange):
    file_name = f"./data/th{probe}_fgm.parquet"
    if os.path.exists(file_name):
        return file_name

    start = trange.start.to_string()
    end = trange.end.to_string()
    
    files = pyspedas.themis.fgm(
        probe=probe,
        trange=[start, end],
        downloadonly=True,
        no_update=True,
    )

    thm_fgl_gses = []
    thm_fgl_btotals = []
    thm_fgl_times = []

    for file in files:
        cdf = pycdfpp.load(file)
        thm_fgl_gses.append(cdf[f"th{probe}_fgl_gse"].values)
        thm_fgl_btotals.append(cdf[f"th{probe}_fgl_btotal"].values)
        thm_fgl_times.append(cdf[f"th{probe}_fgl_time"].values)

    thm_fgl_gse = np.concatenate(thm_fgl_gses)
    thm_fgl_btotal = np.concatenate(thm_fgl_btotals)
    thm_fgl_time = np.concatenate(thm_fgl_times)

    pl.DataFrame(
        {
            "time": thm_fgl_time,
            "BX": thm_fgl_gse[:,0],
            "BY": thm_fgl_gse[:,1],
            "BZ": thm_fgl_gse[:,2],
            "B": thm_fgl_btotal,
        }
    ).with_columns(
        pl.from_epoch(pl.col("thm_fgl_time"), time_unit="s"),
    ).write_parquet(   
        file_name
    )
    
    return file_name

# %%
#| eval: false
convert_thm_state_to_parquet(probe, trange)
convert_thm_fgm_to_parquet(probe, trange)


