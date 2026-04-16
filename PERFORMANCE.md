# Performance Improvements & Bug Fixes

This fork of [Orange3](https://github.com/biolab/orange3) contains targeted performance
optimizations and crash fixes across the core data engine and several widgets.
All changes are backward-compatible, pass the existing test suite, and are made
under the original **GPL-3.0** license.

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

**`Table.transform()` short-circuit**
```python
# before: always created a new table, even with same domain
# after:
if domain is self.domain:
    return self
```
Avoids full column copy when the domain object is identical (common in widget pipelines).

**Transpose: names and attribute extraction**  
Replaced `_fcol.astype(str).tolist()` (produces `'1.0'` for integers) with
`[_fcvar.str_val(v) for v in _fcol]` which uses the variable's own format string.  
Fixed `set_attributes_of_attributes` to use `variable.str_val(val)` instead of
`repr_val` (which added spurious quotes around string values).

---

### `Orange/widgets/data/owfeaturestatistics.py`

Added `_prewarm_distributions_cache()` — pre-computes histogram scenes for all
columns on idle via `QTimer.singleShot(0, ...)`. First scroll through the statistics
table is now instant instead of building scenes one by one.

---

### `Orange/widgets/data/owmergedata.py`

**`_values()` — Instance ID extraction**
```python
# before: O(N) Python generator
return np.fromiter((inst.id for inst in data), count=len(data), dtype=int)

# after: direct numpy array access
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
and greatly reduces time to populate large domains.

---

### `Orange/widgets/visualize/owheatmap.py`

**`column_str_from_table()` — fully vectorized**

Original code iterated row-by-row through Orange `Value` objects.
Replacement uses `table.get_column()` + numpy array operations for both
discrete (index-based lookup) and continuous (vectorized `str_val`) columns.
Handles NaN correctly in both cases.

---

### `Orange/widgets/visualize/owmosaic.py`

**`draw_data()` — distribution caching**

`get_distribution(data, class_var.name)` was called inside the per-cell `draw_data`
callback on every render. It is now computed once before the loop and referenced
via closure — from O(cells) distribution queries to O(1).

---

### `Orange/widgets/visualize/owvenndiagram.py`

**`get_unique_values()` — string column crash**
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

#### Crash fix — no hard data cap

The original widget would crash (OOM or GPU exhaustion) on datasets with more
than ~10 000 instances, because it tried to pass every row to pyqtgraph as an
individual polyline. The fix does **not** truncate the input data. Instead:

- Statistics (mean, range band, error bars) are always computed from **all** instances.
- Individual line rendering is limited to `MAX_LINES = 10 000` via random sampling.
- An info message tells the user that lines are sampled while stats remain exact.

#### Dynamic lazy loading

Lines rendered update in real time as the user zooms or scrolls:

```
sigRangeChanged → 100 ms debounce → _update_display_range()
  → ProfileGroup.update_display_for_range(y_min, y_max)
    → filter rows with any value in [y_min, y_max]  (vectorized numpy)
    → sample to MAX_LINES if still too many
    → PlotCurveItem.setData(...)
```

Zooming into a narrow y-band that contains only 200 lines renders all 200.
Zooming out to show 50 000 lines renders a representative 10 000 sample.

#### Scroll wheel support

```
scroll wheel       → pan y-axis (10 % of visible range per tick)
Ctrl + scroll      → zoom y-axis (1.15× per tick)
right-click        → reset to full auto-range
```

Each scroll event triggers the lazy-load pipeline above.

#### Vectorized internals

| Method | Before | After |
|---|---|---|
| `add_profiles()` | Python loop over features building list of arrays | `np.stack` + `np.broadcast_to` — single allocation |
| `__get_disconnected_curve_missing_data()` | Python `for row in ...` to zero leading-NaN flags | `col_range[None,:] < first_non_nan[:,None]` mask |
| `set_subset_ids()` | `for e in subset_data` instance iteration | `set(subset_data.ids)` + set intersection |
| `_update_sel_profiles_and_range()` | `self.selection` list → O(N) membership per instance | `set(self.selection)` → O(1) |

---

## Running the Tests

All changes pass the existing Orange3 test suite:

```bash
python -m pytest Orange/widgets/visualize/tests/test_owlineplot.py       # 34 passed
python -m pytest Orange/widgets/visualize/tests/test_owmosaic.py \
                 Orange/widgets/data/tests/test_owselectrows.py \
                 Orange/widgets/data/tests/test_owfeaturestatistics.py    # 38 passed
python -m pytest Orange/widgets/data/tests/test_owconcatenate.py \
                 Orange/widgets/data/tests/test_owmergedata.py \
                 Orange/widgets/visualize/tests/test_owheatmap.py         # 119 passed
python -m pytest Orange/widgets/visualize/tests/test_owvenndiagram.py \
                 Orange/widgets/unsupervised/tests/test_owmanifoldlearning.py  # 44 passed
```

Zero new failures introduced.

---

## License

This fork is distributed under the same **GPL-3.0** license as the original
Orange3 project. Copyright for the original codebase remains with the
Bioinformatics Laboratory, University of Ljubljana.
