"""
Arrow / DuckDB / Polars backend utilities for Orange Tables.

This module provides:
  - Zero-copy Arrow <-> Table conversions
  - DuckDB-powered large-data filter and aggregation helpers
  - Arrow-accelerated domain conversion fast-path
  - In-memory DuckDB engine for vectorized table operations

Usage
-----
>>> from Orange.data.arrow_backend import (
...     table_to_arrow, table_from_arrow,
...     duckdb_filter, duckdb_engine,
... )
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    _ARROW_AVAILABLE = True
except ImportError:
    _ARROW_AVAILABLE = False

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

try:
    import duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False

if TYPE_CHECKING:
    from Orange.data import Table, Domain, Variable

__all__ = [
    "table_to_arrow",
    "table_from_arrow",
    "arrow_to_numpy",
    "numpy_to_arrow",
    "duckdb_engine",
    "duckdb_filter_table",
    "ARROW_AVAILABLE",
    "POLARS_AVAILABLE",
    "DUCKDB_AVAILABLE",
    "LARGE_TABLE_THRESHOLD",
]

log = logging.getLogger(__name__)

# Row threshold above which we switch to DuckDB/Polars paths for filters
LARGE_TABLE_THRESHOLD = 100_000
ARROW_AVAILABLE = _ARROW_AVAILABLE
POLARS_AVAILABLE = _POLARS_AVAILABLE
DUCKDB_AVAILABLE = _DUCKDB_AVAILABLE


# ---------------------------------------------------------------------------
# Thread-local DuckDB connection (in-memory)
# ---------------------------------------------------------------------------

class _DuckDBEngine:
    """Thread-safe wrapper around a per-thread DuckDB in-memory database."""

    _tl = threading.local()

    @property
    def conn(self) -> "duckdb.DuckDBPyConnection":
        if not _DUCKDB_AVAILABLE:
            raise RuntimeError("duckdb is not installed")
        conn = getattr(self._tl, "conn", None)
        if conn is None:
            conn = duckdb.connect(":memory:")
            self._tl.conn = conn
        return conn

    def query(self, sql: str, **kwargs):
        return self.conn.execute(sql, **kwargs)

    def query_arrow(self, sql: str, **kwargs) -> "pa.Table":
        return self.conn.execute(sql, **kwargs).fetch_arrow_table()

    def register_arrow(self, name: str, arrow_table: "pa.Table"):
        self.conn.register(name, arrow_table)

    def unregister(self, name: str):
        try:
            self.conn.unregister(name)
        except Exception:
            pass


duckdb_engine = _DuckDBEngine()


# ---------------------------------------------------------------------------
# numpy <-> Arrow helpers
# ---------------------------------------------------------------------------

def numpy_to_arrow(
    arr: np.ndarray,
    var: "Variable",
) -> "pa.ChunkedArray":
    """
    Convert a 1-D numpy column to an Arrow ChunkedArray respecting Orange
    variable types (NaN becomes null).

    Parameters
    ----------
    arr : np.ndarray
        1-D float64 or object array
    var : Variable
        Orange variable descriptor for type hints

    Returns
    -------
    pa.ChunkedArray
    """
    if not _ARROW_AVAILABLE:
        raise RuntimeError("pyarrow is not installed")

    from Orange.data import StringVariable, DiscreteVariable, TimeVariable

    if isinstance(var, StringVariable):
        mask = np.array([x is None or (isinstance(x, float) and np.isnan(x))
                         or x == var.Unknown for x in arr])
        str_arr = np.where(mask, None, arr)
        return pa.chunked_array([pa.array(str_arr.tolist(), type=pa.string())])

    if isinstance(var, DiscreteVariable):
        # Store as dictionary array
        nan_mask = np.isnan(arr.astype(float))
        indices = np.where(nan_mask, -1, arr).astype(np.int32)
        dict_arr = pa.DictionaryArray.from_arrays(
            pa.array(indices, mask=nan_mask),  # mask must be np.ndarray
            pa.array([str(v) for v in var.values], type=pa.string()),
        )
        return pa.chunked_array([dict_arr])

    if isinstance(var, TimeVariable):
        nan_mask = np.isnan(arr.astype(float))
        ms_arr = np.where(nan_mask, None, (arr * 1000).astype("int64"))
        return pa.chunked_array(
            [pa.array(ms_arr.tolist(), type=pa.timestamp("ms", tz="UTC"))]
        )

    # Default: continuous float64
    float_arr = arr.astype(np.float64)
    nan_mask = np.isnan(float_arr)
    return pa.chunked_array(
        [pa.array(float_arr, mask=nan_mask, type=pa.float64())]
    )


def arrow_to_numpy(
    col: Union["pa.Array", "pa.ChunkedArray"],
    var: "Variable",
) -> np.ndarray:
    """
    Convert an Arrow column back to a numpy array matching Orange's conventions.

    Parameters
    ----------
    col : pa.Array or pa.ChunkedArray
    var : Variable

    Returns
    -------
    np.ndarray
    """
    if not _ARROW_AVAILABLE:
        raise RuntimeError("pyarrow is not installed")

    from Orange.data import StringVariable, DiscreteVariable, TimeVariable

    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()

    if isinstance(var, StringVariable):
        result = col.to_pylist()
        arr = np.array(
            [var.Unknown if x is None else x for x in result],
            dtype=object,
        )
        return arr

    if isinstance(var, DiscreteVariable):
        orange_cats = list(var.values)
        if pa.types.is_dictionary(col.type):
            # Dictionary-encoded: remap indices to Orange's category order
            arrow_cats = col.dictionary.to_pylist()
            remap = np.full(len(arrow_cats), np.nan)
            for i, cat in enumerate(arrow_cats):
                cat_str = str(cat)
                if cat_str in orange_cats:
                    remap[i] = float(orange_cats.index(cat_str))
            indices = col.indices.to_pylist()
            result = np.array(
                [np.nan if idx is None else remap[idx] for idx in indices],
                dtype=np.float64,
            )
        elif pa.types.is_floating(col.type) or pa.types.is_integer(col.type):
            # Numeric indices already (e.g. from a previous Orange conversion)
            null_mask = col.is_null().to_numpy(zero_copy_only=False)
            result = col.cast(pa.float64()).to_numpy(zero_copy_only=False)
            result = np.where(null_mask, np.nan, result)
        else:
            # String column from DuckDB/Parquet – map values to Orange indices
            str_vals = col.cast(pa.string()).to_pylist()
            val_map = {v: float(i) for i, v in enumerate(orange_cats)}
            result = np.array(
                [val_map.get(str(v), np.nan) if v is not None else np.nan
                 for v in str_vals],
                dtype=np.float64,
            )
        return result

    if isinstance(var, TimeVariable):
        if pa.types.is_timestamp(col.type) or pa.types.is_date(col.type):
            ms = col.cast(pa.timestamp("ms")).cast(pa.int64())
            null_mask = col.is_null().to_numpy(zero_copy_only=False)
            seconds = ms.to_numpy(zero_copy_only=False).astype(np.float64) / 1000.0
            return np.where(null_mask, np.nan, seconds)
        return col.cast(pa.float64()).to_numpy(zero_copy_only=False)

    # Continuous / numeric
    try:
        arr = col.to_numpy(zero_copy_only=True)
        return arr.astype(np.float64)
    except Exception:
        null_mask = col.is_null().to_numpy(zero_copy_only=False)
        arr = col.cast(pa.float64()).fill_null(0).to_numpy(zero_copy_only=False)
        return np.where(null_mask, np.nan, arr)


# ---------------------------------------------------------------------------
# Orange Table -> Arrow Table
# ---------------------------------------------------------------------------

def table_to_arrow(table: "Table", include_metas: bool = True) -> "pa.Table":
    """
    Convert an Orange Table to a PyArrow Table (columnar, zero-copy where possible).

    This gives Arrow-native access to all rows without going through pandas.

    Parameters
    ----------
    table : Orange.data.Table
    include_metas : bool
        Include meta-attribute columns (default True)

    Returns
    -------
    pa.Table
    """
    if not _ARROW_AVAILABLE:
        raise RuntimeError("pyarrow is not installed")

    from scipy.sparse import issparse

    columns = []
    fields = []
    domain = table.domain

    def _add_col(var, arr_1d):
        ca = numpy_to_arrow(arr_1d, var)
        columns.append(ca)
        fields.append(pa.field(var.name, ca.type))

    # --- attributes (X) ---
    X = table.X
    if issparse(X):
        X = X.toarray()
    for i, var in enumerate(domain.attributes):
        _add_col(var, X[:, i])

    # --- class variables (Y) ---
    Y = table.Y
    if issparse(Y):
        Y = Y.toarray()
    if Y.ndim == 1:
        Y = Y[:, None]
    for i, var in enumerate(domain.class_vars):
        _add_col(var, Y[:, i] if Y.shape[1] > 0 else np.full(len(table), np.nan))

    # --- metas ---
    if include_metas:
        metas = table.metas
        if issparse(metas):
            metas = metas.toarray()
        for i, var in enumerate(domain.metas):
            _add_col(var, metas[:, i])

    schema = pa.schema(fields)
    return pa.table({f.name: c for f, c in zip(fields, columns)}, schema=schema)


# ---------------------------------------------------------------------------
# Arrow Table -> Orange Table
# ---------------------------------------------------------------------------

def table_from_arrow(
    arrow_table: "pa.Table",
    domain: Optional["Domain"] = None,
) -> "Table":
    """
    Construct an Orange Table from a PyArrow Table (zero-copy where possible).

    Parameters
    ----------
    arrow_table : pa.Table
        Source Arrow table
    domain : Domain, optional
        If provided, the result table will use this domain.  Variable types are
        inferred from the domain rather than from Arrow types.

    Returns
    -------
    Orange.data.Table
    """
    if not _ARROW_AVAILABLE:
        raise RuntimeError("pyarrow is not installed")

    from Orange.data import (
        Table, Domain,
        ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable,
    )

    if domain is None:
        # Infer domain from Arrow schema
        attrs, metas_vars = [], []
        for field in arrow_table.schema:
            t = field.type
            if pa.types.is_dictionary(t):
                cats = arrow_table.column(field.name).combine_chunks().dictionary.to_pylist()
                var = DiscreteVariable(field.name, [str(c) for c in cats])
                attrs.append(var)
            elif pa.types.is_timestamp(t) or pa.types.is_date(t):
                attrs.append(TimeVariable(field.name))
            elif pa.types.is_floating(t) or pa.types.is_integer(t):
                attrs.append(ContinuousVariable(field.name))
            elif pa.types.is_string(t) or pa.types.is_large_string(t):
                metas_vars.append(StringVariable(field.name))
            else:
                metas_vars.append(StringVariable(field.name))
        domain = Domain(attrs, metas=metas_vars)

    n_rows = arrow_table.num_rows
    schema_names = set(arrow_table.schema.names)

    def _extract(var):
        col_name = var.name
        if col_name in schema_names:
            return arrow_to_numpy(arrow_table.column(col_name), var)
        return np.full(n_rows, var.Unknown if isinstance(var, StringVariable) else np.nan)

    def _build_numeric_block(variables):
        """
        Fast batch extraction for all-continuous, non-string columns.
        Uses PyArrow's internal zero-copy path wherever possible.
        """
        cols_to_use = [v for v in variables
                       if not isinstance(v, StringVariable) and v.name in schema_names]
        if not cols_to_use:
            return None, []

        # Try bulk Polars path (fastest for nullable float arrays)
        if _POLARS_AVAILABLE:
            try:
                col_names = [v.name for v in cols_to_use]
                chunk = arrow_table.select(col_names)
                import polars as pl
                pl_df = pl.from_arrow(chunk)
                out = np.empty((n_rows, len(cols_to_use)), dtype=np.float64)
                for j, v in enumerate(cols_to_use):
                    s = pl_df[v.name]
                    if isinstance(v, DiscreteVariable):
                        out[:, j] = arrow_to_numpy(chunk.column(v.name), v)
                    else:
                        out[:, j] = (
                            s.cast(pl.Float64)
                            .fill_nan(None)
                            .fill_null(float('nan'))
                            .to_numpy()
                        )
                return out, cols_to_use
            except Exception:
                pass

        # Fallback: column-by-column
        out = np.empty((n_rows, len(cols_to_use)), dtype=np.float64)
        for j, v in enumerate(cols_to_use):
            out[:, j] = _extract(v)
        return out, cols_to_use

    # Build X
    if domain.attributes:
        bulk_X, bulk_vars = _build_numeric_block(domain.attributes)
        if bulk_X is not None and len(bulk_vars) == len(domain.attributes):
            X = bulk_X
        else:
            X = np.empty((n_rows, len(domain.attributes)), dtype=np.float64)
            for i, var in enumerate(domain.attributes):
                X[:, i] = _extract(var)
    else:
        X = np.empty((n_rows, 0), dtype=np.float64)

    # Build Y
    if domain.class_vars:
        bulk_Y, bulk_vars = _build_numeric_block(domain.class_vars)
        if bulk_Y is not None and len(bulk_vars) == len(domain.class_vars):
            Y = bulk_Y
        else:
            Y = np.empty((n_rows, len(domain.class_vars)), dtype=np.float64)
            for i, var in enumerate(domain.class_vars):
                Y[:, i] = _extract(var)
        if len(domain.class_vars) == 1:
            Y = Y[:, 0]
    else:
        Y = np.empty((n_rows, 0), dtype=np.float64)

    # Build metas
    if domain.metas:
        metas = np.empty((n_rows, len(domain.metas)), dtype=object)
        for i, var in enumerate(domain.metas):
            metas[:, i] = _extract(var)
    else:
        metas = np.empty((n_rows, 0), dtype=object)

    return Table.from_numpy(domain, X, Y, metas)


# ---------------------------------------------------------------------------
# DuckDB-powered filter
# ---------------------------------------------------------------------------

def duckdb_filter_table(
    table: "Table",
    filter_sql: str,
    view_name: str = "_orange_tbl",
) -> "Table":
    """
    Apply a SQL WHERE clause to an Orange Table using DuckDB.

    The table is registered as an Arrow table named *view_name*, then the
    WHERE filter is evaluated and a new Orange Table with the matching rows
    is returned.

    Parameters
    ----------
    table : Table
    filter_sql : str
        A valid DuckDB WHERE expression, e.g.
        ``'"sepal length" > 5.0 AND "petal length" IS NOT NULL'``
    view_name : str
        Temporary table name used inside DuckDB (default ``_orange_tbl``)

    Returns
    -------
    Filtered Orange Table (same type as input)
    """
    if not (_ARROW_AVAILABLE and _DUCKDB_AVAILABLE):
        raise RuntimeError("pyarrow and duckdb are both required for duckdb_filter_table")

    arrow_tab = table_to_arrow(table, include_metas=True)
    eng = duckdb_engine
    eng.register_arrow(view_name, arrow_tab)
    try:
        result_arrow = eng.query_arrow(
            f'SELECT * FROM "{view_name}" WHERE {filter_sql}'
        )
    finally:
        eng.unregister(view_name)

    return table_from_arrow(result_arrow, domain=table.domain)


# ---------------------------------------------------------------------------
# Arrow-accelerated domain conversion (fast-path for from_table)
# ---------------------------------------------------------------------------

def arrow_domain_convert(
    source: "Table",
    destination_domain: "Domain",
    row_indices=...,
) -> Optional["Table"]:
    """
    Attempt a fast Arrow-based domain conversion.

    Only handles the simple sub-domain case where all destination attributes
    are plain sub-columns of the source (no compute_value transformations).
    Returns None if the fast-path cannot be applied (caller should fall back
    to the regular conversion).

    Parameters
    ----------
    source : Table
    destination_domain : Domain
    row_indices : slice, ndarray, or Ellipsis

    Returns
    -------
    Table or None
    """
    if not _ARROW_AVAILABLE:
        return None

    from Orange.data import Table
    from Orange.data.util import SharedComputeValue
    from numbers import Integral

    # Only handle cases where every column is a direct index (no compute_value)
    src_domain = source.domain
    all_vars = (
        list(destination_domain.attributes)
        + list(destination_domain.class_vars)
        + list(destination_domain.metas)
    )
    src_names = set(v.name for v in src_domain.variables + src_domain.metas)
    for var in all_vars:
        if var.name not in src_names:
            return None  # Computed column – can't fast-path
        if hasattr(var, "compute_value") and var.compute_value is not None:
            return None

    # Build Arrow table from source (only the needed columns)
    needed = {v.name for v in all_vars}
    try:
        all_src_vars = src_domain.variables + src_domain.metas
        src_cols_to_use = [v for v in all_src_vars if v.name in needed]

        # Build column arrays
        columns = {}
        from scipy.sparse import issparse
        X = source.X
        if issparse(X):
            X = X.toarray()
        Y = source.Y
        if issparse(Y):
            Y = Y.toarray()
        if Y.ndim == 1:
            Y = Y[:, None]
        metas = source.metas
        if issparse(metas):
            metas = metas.toarray()

        n_src_attrs = len(src_domain.attributes)
        for var in src_cols_to_use:
            idx = src_domain.index(var)
            if 0 <= idx < n_src_attrs:
                arr = X[:, idx]
            elif idx >= n_src_attrs:
                arr = Y[:, idx - n_src_attrs]
            else:  # meta
                arr = metas[:, -1 - idx]
            columns[var.name] = numpy_to_arrow(arr, var)

        arrow_src = pa.table(columns)

        # Apply row selection
        if row_indices is not ...:
            if isinstance(row_indices, slice):
                row_indices = list(range(*row_indices.indices(len(source))))
            arrow_src = arrow_src.take(row_indices)

        # Convert to destination
        return table_from_arrow(arrow_src, domain=destination_domain)
    except Exception as e:
        log.debug("Arrow fast-path domain conversion failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Polars-based fast statistics
# ---------------------------------------------------------------------------

def polars_basic_stats(
    table: "Table",
    columns: Optional[List] = None,
) -> Optional[np.ndarray]:
    """
    Compute basic per-column stats (min, max, mean, var, nan_count, non_nan_count)
    using Polars for speed on large tables.

    Returns None if Polars is unavailable or conversion fails.
    """
    if not _POLARS_AVAILABLE:
        return None

    try:
        from Orange.data.polars_compat import table_to_polars
        from Orange.data import StringVariable

        domain = table.domain
        if columns is None:
            cols = [v for v in domain.attributes + list(domain.class_vars)
                    if not isinstance(v, StringVariable)]
        else:
            # Accept Variable objects directly or name/index lookups
            _resolved = []
            for c in columns:
                try:
                    v = c if hasattr(c, 'is_continuous') else domain[c]
                    if not isinstance(v, StringVariable):
                        _resolved.append(v)
                except Exception:
                    pass
            cols = _resolved

        if not cols:
            return np.zeros((0, 6))

        df = table_to_polars(table, include_metas=False, variables=cols)

        rows = []
        for var in cols:
            try:
                s = df[var.name].cast(pl.Float64)
            except Exception:
                s = df[var.name]
            non_null = s.drop_nulls()
            _is_nan_sum = s.is_nan().sum() if hasattr(s, 'is_nan') else 0
            n_nan = s.null_count() + (int(_is_nan_sum) if _is_nan_sum is not None else 0)
            try:
                s_clean = non_null.filter(~non_null.is_nan())
            except Exception:
                s_clean = non_null
            count = len(s_clean)
            if count == 0:
                rows.append([np.nan, np.nan, np.nan, 0.0, n_nan, 0.0])
            else:
                rows.append([
                    float(s_clean.min()),
                    float(s_clean.max()),
                    float(s_clean.mean()),
                    0.0,  # variance placeholder (not requested here)
                    float(n_nan),
                    float(count),
                ])
        return np.array(rows, dtype=np.float64)
    except Exception as e:
        log.debug("Polars stats fast-path failed: %s", e)
        return None
