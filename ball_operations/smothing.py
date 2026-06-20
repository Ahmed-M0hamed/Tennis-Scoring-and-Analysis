import pickle 
import pandas as pd 
import numpy as np 

from scipy.signal import savgol_filter, find_peaks
def smooth_trajectory(df, window = 9, poly = 2) :
    """
    Fill small detection gaps then apply Savitzky-Golay smoothing.
    Why Savitzky-Golay?  It fits a polynomial locally, which preserves
    the sharp peak at a bounce better than a simple moving average.
    """
    # Reindex to dense frame range so gaps become NaN
    full_idx = pd.RangeIndex(df["frame"].min(), df["frame"].max() + 1)
    df = (
        df.set_index("frame")
          .reindex(full_idx)
          .rename_axis("frame")
          .reset_index()
    )

    # Interpolate short gaps (≤5 frames) – handles momentary occlusion
    df["cx"] = df["cx"].interpolate(method="linear", limit=5)
    df["cy"] = df["cy"].interpolate(method="linear", limit=5)

    # Savitzky-Golay needs at least window+1 non-NaN points
    valid = df["cy"].notna()
    if valid.sum() > window:
        df.loc[valid, "cy_smooth"] = savgol_filter(
            df.loc[valid, "cy"], window_length=window, polyorder=poly
        )
        df.loc[valid, "cx_smooth"] = savgol_filter(
            df.loc[valid, "cx"], window_length=window, polyorder=poly
        )
    else:
        df["cy_smooth"] = df["cy"]
        df["cx_smooth"] = df["cx"]

    return df

