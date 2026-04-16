import warnings
import logging

import numpy as np

from Orange.data import Table
from Orange.widgets.utils.itemmodels import TableModel

log = logging.getLogger(__name__)


class VirtualTableModel(TableModel):
    """
    A TableModel for DuckDBTable that fetches data in chunks to avoid
    loading all 4.5M+ rows into memory at once.

    Rows are fetched in ``chunk_size``-row pages on demand. Each page is
    downloaded via DuckDBTable.download_data(offset=…) and cached as a
    plain Orange Table so that subsequent per-cell accesses never hit the
    database again.
    """

    def __init__(self, sourcedata, parent=None, chunk_size=500):
        super().__init__(sourcedata, parent)
        self.chunk_size = chunk_size
        self._current_chunk: Table = None   # cached in-memory page
        self._last_chunk_start: int = -1    # row offset of cached page

    # ------------------------------------------------------------------
    # Core per-cell accessor
    # ------------------------------------------------------------------

    def _get_source_item(self, row, coldesc):
        chunk_start = (row // self.chunk_size) * self.chunk_size

        if chunk_start != self._last_chunk_start:
            self._load_chunk(chunk_start)

        if self._current_chunk is None:
            return None

        relative_row = row - chunk_start
        if relative_row >= len(self._current_chunk):
            return None

        if isinstance(coldesc, self.Basket):
            if coldesc.role is self.Meta:
                return self._current_chunk[relative_row:relative_row + 1].metas
            if coldesc.role is self.Attribute:
                return self._current_chunk[relative_row:relative_row + 1].X

        return self._current_chunk[relative_row, coldesc.var]

    # ------------------------------------------------------------------
    # Chunk loading
    # ------------------------------------------------------------------

    def _load_chunk(self, chunk_start: int):
        """
        Download rows [chunk_start, chunk_start+chunk_size) from the
        DuckDB backend into a plain Orange Table and cache it.
        """
        try:
            # SqlTable.copy() is lightweight — no data is transferred.
            chunk_copy = self.source.copy()
            # DuckDBTable.download_data supports offset= for pagination.
            chunk_copy.download_data(
                limit=self.chunk_size,
                partial=True,
                offset=chunk_start,
            )
            # Convert to a plain in-memory Table so that per-row access
            # never triggers additional DB round-trips.
            X = chunk_copy._X if chunk_copy._X is not None else np.zeros((0, 0))
            Y = chunk_copy._Y if (chunk_copy._Y is not None and chunk_copy._Y.size) else None
            M = chunk_copy._metas if (chunk_copy._metas is not None and chunk_copy._metas.size) else None
            self._current_chunk = Table.from_numpy(chunk_copy.domain, X, Y, M)
            self._last_chunk_start = chunk_start
        except Exception as exc:
            log.exception("VirtualTableModel: failed to load chunk at offset %d: %s",
                          chunk_start, exc)
            self._current_chunk = None
            self._last_chunk_start = chunk_start

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def sortColumnData(self, column):
        """
        In-memory sort of a multi-million-row SQL table is not feasible.
        Return an empty array so the caller falls back to unsorted display.
        Sorting should be pushed down to SQL (ORDER BY) at the query level.
        """
        warnings.warn(
            "VirtualTableModel: in-memory sort is disabled for large SQL tables. "
            "Use an ORDER BY clause in your SQL query instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.array([])
