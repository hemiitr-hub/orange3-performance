# Pull members from modules to Orange.data namespace
# pylint: disable=wildcard-import

from .variable import *
from .instance import *
from .domain import *
from .storage import *
from .table import *
from .io_util import *
from .io import *
from .filter import *
from .pandas_compat import *
from .aggregate import *

# Arrow / DuckDB / Polars integration
try:
    from .polars_compat import table_to_polars, table_from_polars
except ImportError:
    pass

try:
    from .arrow_backend import (
        table_to_arrow,
        table_from_arrow,
        duckdb_engine,
        duckdb_filter_table,
        ARROW_AVAILABLE,
        POLARS_AVAILABLE,
        DUCKDB_AVAILABLE,
    )
except ImportError:
    pass

# DuckDB-backed out-of-core table
try:
    from .sql.duckdb_table import DuckDBTable
except ImportError:
    pass
