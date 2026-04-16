"""Polars DataFrame <-> Orange Table conversion helpers"""

import numpy as np
import pyarrow as pa
import polars as pl

from Orange.data import (
    Table, Domain, DiscreteVariable, StringVariable, TimeVariable,
    ContinuousVariable,
)

__all__ = ['table_from_polars', 'table_to_polars']

def table_to_polars(tab: Table, include_metas: bool = False, variables: list = None) -> pl.DataFrame:
    """
    Convert Orange.data.Table to polars.DataFrame using PyArrow and numpy under the hood.

    Parameters
    ----------
    tab : Table
        The Orange Data Table to convert.
    include_metas : bool, (default=False)
        Include table metas into dataframe.
    variables : list, optional
        If provided, only these Variables will be converted to speed up serialization.
        
    Returns
    -------
    polars.DataFrame
    """
    series_dict = {}

    def _column_to_series(col, vals):
        if col.is_discrete:
            # First map missing values
            valid = ~np.isnan(vals)
            codes = np.where(~valid, -1, vals).astype(np.int32)
            
            try:
                import pyarrow as pa
                indices = pa.array(codes, mask=~valid)
                dictionary = pa.array([str(v) for v in col.values])
                dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
                return pl.from_arrow(dict_array).rename(col.name)
            except Exception:
                # Fallback to pure-python loop for compatibility
                str_vals = np.full(len(codes), None, dtype=object)
                for i, cat in enumerate(col.values):
                    str_vals[codes == i] = cat
                return pl.Series(col.name, str_vals.tolist(), dtype=pl.Utf8).cast(pl.Categorical)
                
        elif col.is_time:
            # Unix timestamps to Datetime – handle object-dtype metas too
            float_vals = np.array(vals, dtype=np.float64) if vals.dtype == object else vals
            return pl.Series(col.name, float_vals).cast(pl.Datetime(time_unit="ms"))
        elif col.is_continuous:
            # metas are stored as object dtype even for float values
            if vals.dtype == object:
                float_vals = np.array(
                    [float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else float('nan')
                     for v in vals],
                    dtype=np.float64,
                )
            else:
                float_vals = vals.astype(np.float64)
            return pl.Series(col.name, float_vals, dtype=pl.Float64)
        elif col.is_string:
            return pl.Series(col.name, vals.tolist(), dtype=pl.Utf8)
        return pl.Series(col.name, vals.tolist() if vals.dtype == object else vals)

    if variables is not None:
        valid_cols = set(var.name for var in variables)
    else:
        valid_cols = None

    domain = tab.domain
    if domain.attributes:
        for i, col in enumerate(domain.attributes):
            if valid_cols is None or col.name in valid_cols:
                series_dict[col.name] = _column_to_series(col, tab.X[:, i])

    if domain.class_vars:
        y_values = tab.Y.reshape(tab.Y.shape[0], len(domain.class_vars))
        for i, col in enumerate(domain.class_vars):
            if valid_cols is None or col.name in valid_cols:
                series_dict[col.name] = _column_to_series(col, y_values[:, i])

    if include_metas and domain.metas:
        for i, col in enumerate(domain.metas):
            if valid_cols is None or col.name in valid_cols:
                series_dict[col.name] = _column_to_series(col, tab.metas[:, i])

    # Ensure ordering is retained
    all_vars = tab.domain.variables
    if include_metas:
        all_vars += tab.domain.metas
        
    final_ordered = []
    for var in all_vars:
        if valid_cols is None or var.name in valid_cols:
            if var.name in series_dict:
                final_ordered.append(series_dict[var.name])
    
    return pl.DataFrame(final_ordered)


def table_from_polars(df: pl.DataFrame, force_nominal: bool = False) -> Table:
    """
    Convert polars.DataFrame to Orange.data.Table.

    Parameters
    ----------
    df : polars.DataFrame
        The data frame to convert.
    force_nominal : bool, (default=False)
        Force all string variables to be nominal/discrete.

    Returns
    -------
    Orange.data.Table
    """
    vars_ = [[], [], []] # X, Y, metas
    arrays = [[], [], []] # X, Y, metas

    for col_name in df.columns:
        s = df[col_name]
        dtype = s.dtype

        var = None
        role = 0 # 0=X, 1=Y, 2=Meta

        if dtype == pl.Categorical or dtype == pl.Enum:
            try:
                cats = list(s.drop_nulls().unique().to_list())
            except Exception:
                cats = []
            try:
                cats.sort()
            except Exception:
                pass
                
            cats_str = [str(c) for c in cats]
            var = DiscreteVariable(str(col_name), cats_str)
            try:
                arr = s.cast(pl.Utf8).cast(pl.Enum(cats_str)).to_physical().cast(pl.Float64).fill_null(np.nan).to_numpy()
            except Exception:
                # Safe fallback if Enum cast fails or isn't supported
                str_arr = s.cast(pl.Utf8).to_numpy(allow_missing=True)
                mapping = {c: float(i) for i, c in enumerate(cats_str)}
                arr = np.array([mapping.get(x, np.nan) for x in str_arr], dtype=np.float64)

        elif dtype == pl.Utf8 or dtype == pl.Object:
            try:
                unique_count = s.n_unique()
            except Exception:
                unique_count = len(s)
                
            if force_nominal or unique_count <= 250 or unique_count < len(s) * 0.05:
                # Deduce categorical (nominal)
                try:
                    cats = list(s.drop_nulls().unique().to_list())
                except Exception:
                    cats = []
                try:
                    cats.sort()
                except Exception:
                    pass
                    
                cats_str = [str(c) for c in cats]
                var = DiscreteVariable(str(col_name), cats_str)
                role = 0
                try:
                    arr = s.cast(pl.Utf8).cast(pl.Enum(cats_str)).to_physical().cast(pl.Float64).fill_null(np.nan).to_numpy()
                except Exception:
                    # Safe fallback
                    str_arr = s.cast(pl.Utf8).to_numpy(allow_missing=True)
                    mapping = {c: float(i) for i, c in enumerate(cats_str)}
                    arr = np.array([mapping.get(x, np.nan) for x in str_arr], dtype=np.float64)
            else:
                var = StringVariable(str(col_name))
                role = 2 # Strings go to metas
                arr = s.cast(pl.Utf8).fill_null(StringVariable.Unknown).to_numpy()

        elif dtype in (pl.Datetime, pl.Date, pl.Time):
            var = TimeVariable(str(col_name))
            arr = s.cast(pl.Datetime(time_unit="ms")).cast(pl.Float64).fill_null(np.nan).to_numpy() / 1000.0

        elif dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            var = ContinuousVariable(str(col_name))
            arr = s.cast(pl.Float64).fill_null(np.nan).to_numpy()

        elif dtype == pl.Boolean:
            var = DiscreteVariable(str(col_name), ['False', 'True'])
            arr = s.cast(pl.Float64).fill_null(np.nan).to_numpy()

        else:
            var = StringVariable(str(col_name))
            role = 2
            arr = s.cast(pl.Utf8).fill_null(StringVariable.Unknown).to_numpy()

        vars_[role].append(var)
        arrays[role].append(arr)

    domain = Domain(vars_[0], vars_[1], vars_[2])

    if arrays[0]:
        X = np.empty((df.height, len(arrays[0])), dtype=np.float64)
        for i, arr in enumerate(arrays[0]):
            X[:, i] = arr
    else:
        X = np.empty((df.height, 0), dtype=np.float64)
        
    if arrays[1]:
        Y = np.empty((df.height, len(arrays[1])), dtype=np.float64)
        for i, arr in enumerate(arrays[1]):
            Y[:, i] = arr
    else:
        Y = np.empty((df.height, 0), dtype=np.float64)

    if arrays[2]:
        metas = np.empty((df.height, len(arrays[2])), dtype=object)
        for i, arr in enumerate(arrays[2]):
            metas[:, i] = arr
    else:
        metas = np.empty((df.height, 0), dtype=object)

    return Table.from_numpy(domain, X, Y, metas)
