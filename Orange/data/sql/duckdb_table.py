"""
DuckDB Table wrapper for Orange.
Allows running Orange algorithms on out-of-core data like Parquet.
"""
import logging

from Orange.data.sql.table import SqlTable
from Orange.data.sql.backend.duckdb_backend import DuckdbBackend

log = logging.getLogger(__name__)

class DuckDBTable(SqlTable):
    def __init__(self, table_or_sql, database=":memory:", type_hints=None, inspect_values=False):
        """
        Create a DuckDB backed SqlTable.
        table_or_sql can be a path to a Parquet/CSV file, e.g. 'read_parquet("large_file.parquet")'
        """
        connection_params = {"database": database}
        super().__init__(connection_params, table_or_sql, backend=DuckdbBackend, 
                         type_hints=type_hints, inspect_values=inspect_values)

    def download_data(self, limit=None, partial=False, offset=None):
        """
        Zero-copy data transfer from DuckDB to Orange using Arrow and Polars.
        """
        import numpy as np
        import polars as pl
        import warnings
        
        if limit and not partial and self.approx_len() > limit:
            raise ValueError("Too many rows to download the data into memory.")

        # Re-use parent logic to build the SELECT query exactly as it does
        attributes = self.domain.attributes + self.domain.class_vars + self.domain.metas
        fields = ["(%s) AS %s" % (attr.to_sql(), self.backend.quote_identifier(attr.name)) for attr in attributes]
        if not fields:
            fields = ["*"]
        
        row_filters = [f.to_sql() for f in self.row_filters]
        # Get query bound to offset/limit if present
        query = self.backend.create_sql_query(
            self.table_name, fields, row_filters, limit=limit, offset=offset
        )
        
        with self.backend.execute_sql_query(query) as cur:
            # Try Arrow/Polars zero-copy path
            try:
                # Use fetch_arrow_table instead of pl() because python's cp1252 charmap encoding fails on utf8 strings
                arrow_tab = cur.fetch_arrow_table()
                if arrow_tab is None:
                    raise RuntimeError("No Arrow table returned")
                # Create polars dataframe from the arrow table safely
                df = pl.from_arrow(arrow_tab)
            except (RuntimeError, ValueError, TypeError) as e:
                # Fallback to standard Python tuple iterator in SqlTable if Arrow fails.
                # NOTE: offset is only supported by DuckDBTable; the parent SqlTable
                # download_data does not accept it, so chunked fallback is not possible.
                warnings.warn(
                    f"DuckDB Arrow fast-path failed; falling back to tuple iteration. "
                    f"Offset={offset} will be ignored in fallback. ({e})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                super().download_data(limit=limit, partial=partial)
                return

        # Map Polars array to memory without iterators
        self._X = np.empty((df.height, len(self.domain.attributes)), dtype=np.float64)
        for i, attr in enumerate(self.domain.attributes):
            self._X[:, i] = self._extract_array(df, attr)

        self._Y = np.empty((df.height, len(self.domain.class_vars)), dtype=np.float64)
        for i, attr in enumerate(self.domain.class_vars):
            self._Y[:, i] = self._extract_array(df, attr)

        self._metas = np.empty((df.height, len(self.domain.metas)), dtype=object)
        for i, attr in enumerate(self.domain.metas):
            self._metas[:, i] = self._extract_array(df, attr, is_meta=True)

        self._W = np.empty((df.height, 0))
        self._init_ids(self)
        if not partial or limit and df.height < limit:
            self._cached__len__ = df.height

    def _extract_array(self, df, attr, is_meta=False):
        import numpy as np
        import polars as pl
        from Orange.data import StringVariable
        
        s = df[attr.name]
        
        if attr.is_discrete:
            # Map strings back to discrete indexes (handling StringCache offsets correctly)
            cat_series = None
            try:
                cat_series = s.cast(pl.Categorical)
                cats = list(cat_series.cat.get_categories().to_list())
            except (pl.exceptions.InvalidOperationError, pl.exceptions.SchemaError,
                    pl.exceptions.ComputeError, ValueError, TypeError) as e:
                log.warning("_extract_array: could not cast column '%s' to Categorical "
                            "(%s); falling back to unique-value enumeration.", attr.name, e)
                try:
                    cats = list(set(s.drop_nulls().to_list()))
                except Exception as inner:
                    log.warning("_extract_array: could not enumerate values for '%s': %s",
                                attr.name, inner)
                    cats = []

            try:
                if cat_series is not None:
                    data_col = cat_series.to_physical().cast(pl.Float64).fill_null(np.nan).to_numpy()
                else:
                    data_col = s.cast(pl.Utf8).cast(pl.Categorical).to_physical().cast(pl.Float64).fill_null(np.nan).to_numpy()
            except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError,
                    ValueError, TypeError) as e:
                log.warning("_extract_array: physical cast failed for '%s' (%s); "
                            "returning all-NaN.", attr.name, e)
                data_col = np.full(len(s), np.nan)

            # Reproject indices to exactly match our Orange domain category indices
            if cats and list(cats) != list(attr.values):
                mapper = np.full(len(cats), np.nan)
                matched = 0
                for i, r_val in enumerate(cats):
                    r_val_str = str(r_val)
                    if r_val_str in attr.values:
                        mapper[i] = attr.values.index(r_val_str)
                        matched += 1
                if matched == 0:
                    log.warning(
                        "_extract_array: none of the %d DuckDB categories for '%s' "
                        "match the Orange domain values. Data will be all-NaN.",
                        len(cats), attr.name,
                    )
                codes = np.where(np.isnan(data_col), -1, data_col).astype(int)
                remapped = np.full(len(codes), np.nan)
                valid = (codes >= 0) & (codes < len(mapper))
                remapped[valid] = mapper[codes[valid]]
                data_col = remapped
                
            if is_meta:
                 # Reconstruct strings from mapped indices for Meta variables
                 # Because Polars Dataframes return arrays for meta discretes
                 str_arr = np.full(len(data_col), np.nan, dtype=object)
                 for i, cat in enumerate(attr.values):
                     str_arr[data_col == i] = cat
                 return str_arr
                 
            return data_col

        elif attr.is_continuous:
            return s.cast(pl.Float64).fill_null(np.nan).to_numpy()
            
        elif attr.is_time:
             return s.cast(pl.Datetime(time_unit="ms")).cast(pl.Float64).fill_null(np.nan).to_numpy() / 1000.0
             
        elif attr.is_string:
            try:
                arr = s.cast(pl.Utf8).fill_null(StringVariable.Unknown).to_numpy()
            except Exception:
                # DuckDB/Arrow might return string columns differently
                arr = np.array([str(x) if x is not None else StringVariable.Unknown for x in s.to_list()], dtype=object)
            return arr
            
        else:
            return s.to_numpy()
    def _compute_distributions(self, columns=None):
        """
        Push down distribution computation to DuckDB.
        """
        import numpy as np
        import polars as pl
        
        if columns is None:
            columns = self.domain.variables
        else:
            columns = [self.domain[col] for col in columns]

        dists = []
        for col in columns:
            field_name = self.backend.quote_identifier(col.name)
            
            # 1. Total count of non-null values
            row_filters = [f.to_sql() for f in self.row_filters]
            q_filter = " AND ".join(row_filters + [f"{field_name} IS NOT NULL"])
            
            # 2. Group by query
            query = f'SELECT {field_name}, COUNT(*) FROM {self.table_name} WHERE {q_filter} GROUP BY 1 ORDER BY 1'
            
            # 3. Unknowns count
            q_unks = " AND ".join(row_filters + [f"{field_name} IS NULL"])
            unk_query = f'SELECT COUNT(*) FROM {self.table_name} WHERE {q_unks}'
            
            with self.backend.execute_sql_query(unk_query) as cur:
                unknowns = cur.fetchone()[0]

            with self.backend.execute_sql_query(query) as cur:
                arrow_tab = cur.fetch_arrow_table()
                if arrow_tab.num_rows == 0:
                    if col.is_discrete:
                        dists.append((np.zeros(len(col.values)), unknowns))
                    else:
                        dists.append((np.zeros((2, 0)), unknowns))
                    continue
                    
                df = pl.from_arrow(arrow_tab)
                
            if col.is_continuous:
                # Continuous distributions in Orange are 2xN (values, counts)
                # DuckDB returns [value, count] columns
                val_col = df.get_column(df.columns[0]).cast(pl.Float64).to_numpy()
                cnt_col = df.get_column(df.columns[1]).cast(pl.Float64).to_numpy()
                dists.append((np.vstack([val_col, cnt_col]), unknowns))
            else:
                # Discrete: Map back to Orange indices
                counts = np.zeros(len(col.values))
                vals = df.get_column(df.columns[0]).cast(pl.Utf8).to_list()
                c_counts = df.get_column(df.columns[1]).to_numpy()
                for v, c in zip(vals, c_counts):
                    v_str = str(v)
                    if v_str in col.values:
                        counts[col.values.index(v_str)] = c
                dists.append((counts, unknowns))
        return dists

    def _compute_contingency(self, col_vars=None, row_var=None):
        """
        Push down contingency computation to DuckDB.
        """
        import numpy as np
        import polars as pl

        if col_vars is None:
            col_vars = range(len(self.domain.variables))
        
        row_variable = self.domain[row_var]
        if not row_variable.is_discrete:
            raise TypeError("Row variable must be discrete")
            
        columns = [self.domain[var] for var in col_vars]
        row_field = self.backend.quote_identifier(row_variable.name)
        
        all_contingencies = []
        for col in columns:
            col_field = self.backend.quote_identifier(col.name)
            row_filters = [f.to_sql() for f in self.row_filters]
            q_filter = " AND ".join(row_filters + [f"{row_field} IS NOT NULL", f"{col_field} IS NOT NULL"])
            
            # Query: [row_val, col_val, count]
            query = f"SELECT {row_field} AS row_val, {col_field} AS col_val, COUNT(*) AS cnt FROM {self.table_name} WHERE {q_filter} GROUP BY 1, 2 ORDER BY 2"
            
            # Unknowns
            unk_filter = " AND ".join(row_filters + [f"({row_field} IS NULL OR {col_field} IS NULL)"])
            unk_query = f"SELECT COUNT(*) FROM {self.table_name} WHERE {unk_filter}"
            with self.backend.execute_sql_query(unk_query) as cur:
                unknowns = cur.fetchone()[0]
                
            with self.backend.execute_sql_query(query) as cur:
                arrow_tab = cur.fetch_arrow_table()
                df = pl.from_arrow(arrow_tab)

            if col.is_continuous:
                # Continuous: (values, counts_matrix)
                # row_vals (discrete) are categories 0, 1, 2...
                # col_vals (continuous) are unique values
                # counts_matrix is (n_rows, n_cols)
                # Use column indices for robustness
                unique_cols = df.get_column("col_val").unique(maintain_order=True).cast(pl.Float64).to_numpy()
                counts = np.zeros((len(row_variable.values), len(unique_cols)))
                
                # Pivot-like mapping
                col_map = {val: i for i, val in enumerate(unique_cols)}
                row_map = {val: i for i, val in enumerate(row_variable.values)}
                
                for r_val_str, c_val, cnt in df.iter_rows():
                    # Correctly handle potential None and convert to str/float
                    r_val_str = str(r_val_str) if r_val_str is not None else ""
                    c_val = float(c_val) if c_val is not None else 0.0
                    if r_val_str in row_map and c_val in col_map:
                        counts[row_map[r_val_str], col_map[c_val]] = cnt
                
                all_contingencies.append(((unique_cols, counts), np.zeros(len(unique_cols)), np.zeros(len(row_variable.values)), unknowns))
            else:
                # Discrete x Discrete: simple matrix
                counts = np.zeros((len(row_variable.values), len(col.values)))
                row_map = {val: i for i, val in enumerate(row_variable.values)}
                col_map = {val: i for i, val in enumerate(col.values)}
                
                for r_val_str, c_val_str, cnt in df.iter_rows():
                    r_val_str = str(r_val_str) if r_val_str is not None else ""
                    c_val_str = str(c_val_str) if c_val_str is not None else ""
                    if r_val_str in row_map and c_val_str in col_map:
                        counts[row_map[r_val_str], col_map[c_val_str]] = cnt
                
                all_contingencies.append((counts, np.zeros(len(col.values)), np.zeros(len(row_variable.values)), unknowns))
                
        return all_contingencies
