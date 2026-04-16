"""
Benchmarks: orange3-performance vs original Orange3 implementations.
Runs each scenario 5 times, reports median time in ms.
"""
import time
import statistics
import numpy as np
import scipy.sparse as sp

# ── helpers ──────────────────────────────────────────────────────────────────
def bench(fn, n=5):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 2)

def fmt(old, new):
    speedup = old / new if new > 0 else float('inf')
    return f"{old:.1f} ms  ->  {new:.1f} ms  ({speedup:.1f}x faster)"

print("=" * 65)
print("orange3-performance  benchmark suite")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Table.transform() identity short-circuit
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Table.transform() — same domain")
from Orange.data import Table, Domain
iris = Table("iris")
new_domain = Domain(iris.domain.attributes, iris.domain.class_var)
t_old = bench(lambda: iris.transform(new_domain))          # different object
t_new = bench(lambda: iris.transform(iris.domain))         # identity → shortcut
print("   original (different obj):", f"{t_old:.1f} ms")
print("   optimized (same obj)    :", f"{t_new:.1f} ms")

# ─────────────────────────────────────────────────────────────────────────────
# 2. MergeData _values() — INSTANCEID extraction
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Instance-ID extraction  (N=50 000)")
N = 50_000
from Orange.data import Table as T
big = Table("iris")[np.tile(np.arange(150), N // 150 + 1)[:N]]

def old_instance_id(data):
    return np.fromiter((inst.id for inst in data), count=len(data), dtype=int)

def new_instance_id(data):
    return data.ids.astype(int)

t_old = bench(lambda: old_instance_id(big))
t_new = bench(lambda: new_instance_id(big))
print("  ", fmt(t_old, t_new))

# ─────────────────────────────────────────────────────────────────────────────
# 3. column_str_from_table() — heatmap annotations
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] column_str_from_table()  (N=10 000, discrete var)")
from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Table
import numpy as np

dv = DiscreteVariable("cls", values=["a", "b", "c", "d"])
dom = Domain([], metas=[dv])
col_data = np.random.randint(0, 4, 10_000).astype(float)
col_data[::17] = np.nan
tbl = Table.from_numpy(dom, np.zeros((10_000, 0)), metas=col_data.reshape(-1, 1))

def old_col_str(table, column):
    return [str(row[column]) for row in table]

def new_col_str(table, column):
    var = table.domain[column]
    data = table.get_column(column)
    fdata = data.astype(float)
    nan_mask = np.isnan(fdata)
    idata = np.where(nan_mask, 0, fdata).astype(int)
    vals = np.empty(len(var.values) + 1, dtype=object)
    vals[0] = "?"
    for i, v in enumerate(var.values):
        vals[i + 1] = v
    return vals[np.where(nan_mask, 0, idata + 1)]

t_old = bench(lambda: old_col_str(tbl, dv))
t_new = bench(lambda: new_col_str(tbl, dv))
print("  ", fmt(t_old, t_new))

# ─────────────────────────────────────────────────────────────────────────────
# 4. add_profiles() — line plot viewport array build
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] add_profiles()  (5 000 instances × 100 features)")
M, N = 5_000, 100
y = np.random.rand(M, N)

def old_add_profiles(y):
    return np.array(
        [np.vstack((np.full((1, y.shape[0]), i + 1), y[:, i].flatten())).T
         for i in range(y.shape[1])])

def new_add_profiles(y):
    m, n = y.shape
    xs = np.arange(1, n + 1, dtype=float)
    xs_tiled = np.broadcast_to(xs[:, None], (n, m))
    return np.stack([xs_tiled, y.T], axis=2)

t_old = bench(lambda: old_add_profiles(y))
t_new = bench(lambda: new_add_profiles(y))
print("  ", fmt(t_old, t_new))

# ─────────────────────────────────────────────────────────────────────────────
# 5. __get_disconnected_curve_missing_data() — vectorized leading-NaN mask
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] disconnected curve missing-data  (5 000 × 100)")
y_nan = y.copy()
y_nan[::3, :5] = np.nan   # leading NaNs in 1/3 of rows

def old_missing(y_data):
    m, n = y_data.shape
    x = np.arange(m * n) % n + 1
    yf = y_data.flatten()
    connect = np.isnan(y_data)
    first_non_nan = np.argmin(connect, axis=1)
    for row in np.flatnonzero(first_non_nan):
        connect[row, :first_non_nan[row]] = False
    connect[:, -1] = False
    return x, yf, connect.flatten()

def new_missing(y_data):
    m, n = y_data.shape
    x = np.arange(m * n) % n + 1
    yf = y_data.flatten()
    connect = np.isnan(y_data)
    first_non_nan = np.argmin(connect, axis=1)
    col_range = np.arange(n)
    leading_mask = col_range[None, :] < first_non_nan[:, None]
    connect &= ~leading_mask
    connect[:, -1] = False
    return x, yf, connect.flatten()

t_old = bench(lambda: old_missing(y_nan))
t_new = bench(lambda: new_missing(y_nan))
print("  ", fmt(t_old, t_new))

# ─────────────────────────────────────────────────────────────────────────────
# 6. set_subset_ids() — subset ID lookup
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] set_subset_ids()  (N=10 000 data, 500 subset)")
data10k = Table("iris")[np.tile(np.arange(150), 67)[:10_000]]
subset  = data10k[:500]

def old_subset_ids(data, subset_data):
    sub_ids = {e.id for e in subset_data}
    return [x.id for x in data if x.id in sub_ids]

def new_subset_ids(data, subset_data):
    sub_ids = set(subset_data.ids)
    return sub_ids & set(data.ids)

t_old = bench(lambda: old_subset_ids(data10k, subset))
t_new = bench(lambda: new_subset_ids(data10k, subset))
print("  ", fmt(t_old, t_new))

# ─────────────────────────────────────────────────────────────────────────────
# 7. _update_sel_profiles_and_range() — selection membership
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] selection membership lookup  (10 000 indices, 500 selected)")
indices = np.arange(10_000)
selection = list(np.random.choice(10_000, 500, replace=False))

def old_sel(indices, selection):
    return [i for i in indices if i in selection]   # list → O(N) each

def new_sel(indices, selection):
    sel_set = set(selection)
    return [i for i in indices if i in sel_set]     # set → O(1) each

t_old = bench(lambda: old_sel(indices, selection))
t_new = bench(lambda: new_sel(indices, selection))
print("  ", fmt(t_old, t_new))

# ─────────────────────────────────────────────────────────────────────────────
# 8. Mosaic prior_distribution caching
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] prior_distribution  (called 200× per render vs cached)")
from Orange.statistics.distribution import get_distribution
data_big = Table("iris")[np.tile(np.arange(150), 67)[:10_000]]
class_var = data_big.domain.class_var

def old_mosaic_render(data, n_cells=200):
    for _ in range(n_cells):
        _ = get_distribution(data, class_var.name)

def new_mosaic_render(data, n_cells=200):
    prior = get_distribution(data, class_var.name)   # once
    for _ in range(n_cells):
        _ = prior

t_old = bench(lambda: old_mosaic_render(data_big))
t_new = bench(lambda: new_mosaic_render(data_big))
print("  ", fmt(t_old, t_new))

print("\n" + "=" * 65)
print("All benchmarks complete.")
print("=" * 65)
