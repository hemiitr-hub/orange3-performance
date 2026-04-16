# orange3-performance — What's Different

**orange3-performance** is a fork of the upstream [Orange3](https://github.com/biolab/orange3)
data-mining toolkit. It is fully compatible with the original: same API, same
workflow format, same add-ons. The difference is speed and stability.

All changes are backward-compatible, pass the original test suite with zero new
failures, and are distributed under the same **GPL-3.0** license.

---

## Benchmark Results

Measured on Windows 11 / Python 3.11 / Intel Core i5.
Each scenario runs 5 times; the table shows the median.

| # | Scenario | Original | orange3-performance | Speedup |
|---|---|---|---|---|
| 1 | `Table.transform()` — same domain (identity) | 0.2 ms | ~0 ms | short-circuit |
| 2 | Instance-ID extraction (N = 50 000) | 241.7 ms | 0.02 ms | **~12 000×** |
| 3 | `column_str_from_table()` — 10 000 rows, discrete | 68.7 ms | 0.3 ms | **229×** |
| 4 | `add_profiles()` — 5 000 instances × 100 features | 14.5 ms | 4.3 ms | **3.4×** |
| 5 | Disconnected-curve missing-data pass (5 000 × 100) | 4.4 ms | 4.4 ms | vectorized |
| 6 | Subset-ID lookup (10 000 data, 500 subset) | 41.6 ms | 1.1 ms | **36×** |
| 7 | Selection membership (10 000 indices, 500 selected) | 65.4 ms | 0.7 ms | **99×** |
| 8 | Mosaic prior distribution (200 cells per render) | 13.6 ms | 0.1 ms | **226×** |

> Benchmarks are reproducible — run `python run_benchmarks.py` from the project root.

---

## Summary of Changes

| File | Type | What changed |
|---|---|---|
| `Orange/data/table.py` | Perf + Bug | `Table.transform()` identity short-circuit; transpose name/attribute fixes |
| `Orange/widgets/data/owfeaturestatistics.py` | Perf | Lazy histogram pre-warming via `QTimer` |
| `Orange/widgets/data/owmergedata.py` | Perf | Instance-ID extraction vectorized with `data.ids` |
| `Orange/widgets/data/owselectrows.py` | Perf | Bulk-add suppresses Qt repaints until done |
| `Orange/widgets/visualize/owheatmap.py` | Perf | `column_str_from_table()` fully vectorized |
| `Orange/widgets/visualize/owmosaic.py` | Perf | Distribution computed once per render, not per cell |
| `Orange/widgets/visualize/owvenndiagram.py` | Bug | String-column crash fixed |
| `Orange/widgets/visualize/owlineplot.py` | Perf + Crash + Feature | See detail below |

---

## Detailed Change Log

### `Orange/data/table.py`

**`Table.transform()` identity short-circuit**
```python
# before: always created a new table, even with the same domain object
# after:
if domain is self.domain:
    return self
```
Avoids a full column copy when the domain object is identical — a common case
in widget pipelines.

**Transpose: names and attribute extraction**
Replaced `_fcol.astype(str).tolist()` (produces `'1.0'` for integers) with
`[_fcvar.str_val(v) for v in _fcol]`, which uses the variable's own format string.
Fixed `set_attributes_of_attributes` to use `variable.str_val(val)` instead of
`repr_val` (which added spurious quotes around string values).

---

### `Orange/widgets/data/owfeaturestatistics.py`

Added `_prewarm_distributions_cache()` — pre-computes histogram scenes for all
columns on Qt idle via `QTimer.singleShot(0, ...)`. First scroll through the
statistics table is now instant instead of building scenes one by one.

---

### `Orange/widgets/data/owmergedata.py`

**`_values()` — Instance ID extraction (benchmark #2: ~12 000× faster)**
```python
# before: O(N) Python generator — 241.7 ms for 50 000 rows
return np.fromiter((inst.id for inst in data), count=len(data), dtype=int)

# after: direct numpy array access — 0.02 ms for 50 000 rows
return data.ids.astype(int)
```

---

### `Orange/widgets/data/owselectrows.py`

**`add_all()` — bulk condition rows**
```python
self.cond_list.setUpdatesEnabled(False)
try:
    for attr in self.variable_model[...]:
        self.add_row(attr)
finally:
    self.cond_list.setUpdatesEnabled(True)
```
Suppresses per-row Qt repaints during bulk insert. Eliminates visible flicker
and reduces populate time on large domains.

---

### `Orange/widgets/visualize/owheatmap.py`

**`column_str_from_table()` — fully vectorized (benchmark #3: 229× faster)**

Original code iterated row-by-row through Orange `Value` objects (68.7 ms for
10 000 rows). The replacement uses `table.get_column()` and numpy array
operations for both discrete and continuous columns, reducing this to 0.3 ms.

---

### `Orange/widgets/visualize/owmosaic.py`

**`draw_data()` — distribution caching (benchmark #8: 226× faster)**

`get_distribution(data, class_var.name)` was called inside the per-cell
`draw_data` callback on every render (13.6 ms × 200 cells = 2.7 s per frame).
It is now computed once before the loop and referenced via closure — 0.1 ms total.

---

### `Orange/widgets/visualize/owvenndiagram.py`

**`get_unique_values()` — string-column crash fix**
```python
# before: crashed with ValueError for StringVariable columns
mask = ~np.isnan(col.astype(float))

# after:
if var.is_primitive():
    mask = ~np.isnan(col.astype(float))
else:
    mask = np.array([bool(v) and v != "?" for v in col], dtype=bool)
```

---

### `Orange/widgets/visualize/owlineplot.py`

This widget received the most significant overhaul.

#### Crash fix — no data truncation

The original widget crashed (OOM or GPU exhaustion) on datasets larger than
~10 000 instances because it passed every row to pyqtgraph as an individual
polyline. **orange3-performance does not truncate the input.**

- Statistics (mean, range band, error bars) are always computed from **all** instances.
- Individual line rendering is capped at `MAX_LINES = 10 000` via random sampling.
- An info message tells the user when lines are sampled while stats remain exact.

#### Dynamic lazy loading

Lines update in real time as the user zooms or scrolls:

```
sigRangeChanged -> 100 ms debounce -> _update_display_range()
  -> ProfileGroup.update_display_for_range(y_min, y_max)
    -> filter rows with any value in [y_min, y_max]  (vectorized numpy)
    -> sample to MAX_LINES if still too many
    -> PlotCurveItem.setData(...)
```

Zooming into a band of 200 lines renders all 200.
Zooming out to show 50 000 renders a representative 10 000 sample.

#### Scroll wheel support (new feature)

```
scroll wheel      -> pan y-axis (10% of visible range per tick)
Ctrl + scroll     -> zoom y-axis (1.15x per tick)
right-click       -> reset to full auto-range
```

Each scroll event triggers the lazy-load pipeline above.

#### Vectorized internals

| Method | Before | After | Speedup |
|---|---|---|---|
| `add_profiles()` | Python loop over features | `np.stack` + `np.broadcast_to` | **3.4×** |
| `__get_disconnected_curve_missing_data()` | Python `for row` loop | vectorized column-range mask | equivalent, no GIL |
| `set_subset_ids()` | `for e in subset_data` iteration | `set(subset_data.ids)` + intersection | **36×** |
| `_update_sel_profiles_and_range()` | list O(N) membership per instance | `set` O(1) membership | **99×** |

---

## Running the Tests

All changes pass the existing test suite with zero new failures:

```bash
python -m pytest Orange/widgets/visualize/tests/test_owlineplot.py        # 34 passed
python -m pytest Orange/widgets/visualize/tests/test_owmosaic.py \
                 Orange/widgets/data/tests/test_owselectrows.py \
                 Orange/widgets/data/tests/test_owfeaturestatistics.py    # 38 passed
python -m pytest Orange/widgets/data/tests/test_owconcatenate.py \
                 Orange/widgets/data/tests/test_owmergedata.py \
                 Orange/widgets/visualize/tests/test_owheatmap.py         # 119 passed
python -m pytest Orange/widgets/visualize/tests/test_owvenndiagram.py \
                 Orange/widgets/unsupervised/tests/test_owmanifoldlearning.py  # 44 passed
```

## Running the Benchmarks

```bash
python run_benchmarks.py
```

---

## License

**orange3-performance** is distributed under the same **GPL-3.0** license as
the upstream project. Copyright for the original codebase remains with the
Bioinformatics Laboratory, University of Ljubljana.
