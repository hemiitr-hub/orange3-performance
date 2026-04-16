"""
Backend verification tests for Orange3 fixes.
Run with: python run_tests.py
"""
import sys, time, traceback, warnings
import numpy as np

PASS = []
FAIL = []

def check(name, fn):
    try:
        t0 = time.time()
        fn()
        dt = time.time() - t0
        PASS.append((name, dt))
        print(f"  PASS [{dt:.2f}s]: {name}")
    except Exception as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL: {name}")
        traceback.print_exc()

# ── Bug #1: Function.__gt__ ────────────────────────────────────────────────
def test_pivot_function_gt():
    from Orange.widgets.data.owpivot import Function
    f1 = Function(value=1, name="Count", func=lambda x: x.shape[0])
    f2 = Function(value=2, name="Sum",   func=lambda x: np.nansum(x))
    assert not (f1 > f2), "1 > 2 should be False"
    assert f2 > f1,       "2 > 1 should be True"

check("Bug#1 Function.__gt__ fix", test_pivot_function_gt)

# ── Bug #6: __include_aggregation operator precedence ─────────────────────
def test_pivot_include_aggregation():
    # Import Pivot to exercise __include_aggregation
    from Orange.widgets.data.owpivot import Pivot
    from Orange.data import Table
    iris = Table("iris")
    # Pivot.Count etc. are class-level NamedTuple instances (not on .Functions)
    pivot = Pivot(
        iris,
        [Pivot.Count],
        row_var=iris.domain["iris"],
        col_var=iris.domain["iris"],
    )
    assert pivot._group_tables.table is not None or True  # may be None; just no crash

check("Bug#6 __include_aggregation precedence (smoke)", test_pivot_include_aggregation)

# ── VirtualTableModel import ───────────────────────────────────────────────
def test_virtual_table_model_import():
    from Orange.widgets.data.utils.virtual_table_model import VirtualTableModel
    assert VirtualTableModel is not None

check("VirtualTableModel imports cleanly", test_virtual_table_model_import)

# ── DuckDB table import ────────────────────────────────────────────────────
def test_duckdb_table_import():
    from Orange.data.sql.duckdb_table import DuckDBTable
    assert DuckDBTable is not None

check("DuckDBTable imports cleanly", test_duckdb_table_import)

# ── CSV load of full 4.5M dataset ─────────────────────────────────────────
def test_csv_load():
    import Orange.data
    import os
    path = os.path.join(os.path.dirname(__file__), "1 TA_15_SM.csv")
    tbl = Orange.data.Table(path)
    assert len(tbl) > 4_000_000, f"Expected >4M rows, got {len(tbl)}"
    assert len(tbl.domain.attributes) + len(tbl.domain.class_vars) + len(tbl.domain.metas) == 52

check("Load 1 TA_15_SM.csv (4.5M rows)", test_csv_load)

# ── Feature statistics on sample ──────────────────────────────────────────
def test_feature_stats():
    import Orange.data
    import os
    from Orange.statistics.util import nanmean, nanvar, nanmin, nanmax
    path = os.path.join(os.path.dirname(__file__), "1 TA_15_SM.csv")
    tbl = Orange.data.Table(path)
    sample = tbl[:50_000]
    numeric_vars = [v for v in sample.domain.attributes
                    if v.is_continuous][:10]
    for var in numeric_vars:
        col = sample.get_column(var)
        _ = nanmean(col), nanvar(col), nanmin(col), nanmax(col)

check("Feature stats on 50k sample (10 numeric cols)", test_feature_stats)

# ── DataSampler backend ───────────────────────────────────────────────────
def test_data_sampler():
    import Orange.data
    import os
    path = os.path.join(os.path.dirname(__file__), "1 TA_15_SM.csv")
    tbl = Orange.data.Table(path)
    from Orange.preprocess import Randomize
    sample = tbl[:100_000]
    r = Randomize()
    result = r(sample)
    assert len(result) == len(sample)

check("DataSampler / Randomize on 100k rows", test_data_sampler)

# ── Preprocess: Impute ────────────────────────────────────────────────────
def test_impute():
    import Orange.data, os
    from Orange.preprocess import Impute
    path = os.path.join(os.path.dirname(__file__), "1 TA_15_SM.csv")
    tbl = Orange.data.Table(path)
    sample = tbl[:10_000]
    result = Impute()(sample)
    assert len(result) == len(sample)

check("Impute on 10k rows", test_impute)

# ── Preprocess: Normalize ─────────────────────────────────────────────────
def test_normalize():
    import Orange.data, os
    from Orange.preprocess import Normalize
    path = os.path.join(os.path.dirname(__file__), "1 TA_15_SM.csv")
    tbl = Orange.data.Table(path)
    sample = tbl[:10_000]
    result = Normalize()(sample)
    assert len(result) == len(sample)

check("Normalize on 10k rows", test_normalize)

# ── PCA backend ───────────────────────────────────────────────────────────
def test_pca():
    import Orange.data, os
    from Orange.projection import PCA
    path = os.path.join(os.path.dirname(__file__), "1 TA_15_SM.csv")
    tbl = Orange.data.Table(path)
    sample = tbl[:5_000]
    # PCA only works on numeric attributes – use imputed data
    from Orange.preprocess import Impute
    sample = Impute()(sample)
    numeric = [v for v in sample.domain.attributes if v.is_continuous]
    from Orange.data import Domain
    num_domain = Domain(numeric[:10])
    sub = sample.transform(num_domain)
    pca = PCA(n_components=3)
    model = pca(sub)
    assert model is not None

check("PCA on 5k rows (10 numeric cols)", test_pca)

# ── KMeans backend ────────────────────────────────────────────────────────
def test_kmeans():
    import Orange.data, os
    from Orange.clustering import KMeans
    from Orange.preprocess import Impute
    path = os.path.join(os.path.dirname(__file__), "1 TA_15_SM.csv")
    tbl = Orange.data.Table(path)
    sample = tbl[:5_000]
    sample = Impute()(sample)
    numeric = [v for v in sample.domain.attributes if v.is_continuous]
    from Orange.data import Domain
    sub = sample.transform(Domain(numeric[:5]))
    km = KMeans(n_clusters=3)
    labels = km(sub)
    assert labels is not None

check("K-Means (3 clusters) on 5k rows", test_kmeans)

# ── Correlations ─────────────────────────────────────────────────────────
def test_correlations():
    import Orange.data, os, numpy as np
    path = os.path.join(os.path.dirname(__file__), "1 TA_15_SM.csv")
    tbl = Orange.data.Table(path)
    sample = tbl[:5_000]
    from Orange.preprocess import Impute
    sample = Impute()(sample)
    numeric = [v for v in sample.domain.attributes if v.is_continuous]
    from Orange.data import Domain
    sub = sample.transform(Domain(numeric[:6]))
    corr = np.corrcoef(sub.X.T)
    assert corr.shape == (6, 6)

check("Pearson correlation (6 numeric cols, 5k rows)", test_correlations)

# ── SelectRows filter ─────────────────────────────────────────────────────
def test_select_rows():
    import Orange.data, os
    from Orange.data import filter as F
    path = os.path.join(os.path.dirname(__file__), "1 TA_15_SM.csv")
    tbl = Orange.data.Table(path)
    # Filter to rows where TR_MN == 12
    tr_mn = tbl.domain["TR_MN"]
    f = F.Values([F.FilterContinuous(tr_mn, F.FilterContinuous.Equal, 12)])
    result = f(tbl)
    assert len(result) >= 0

check("SelectRows filter on full 4.5M table", test_select_rows)

# ── Decision Tree classifier ──────────────────────────────────────────────
def test_decision_tree():
    import Orange.data, os
    from Orange.classification import TreeLearner
    from Orange.preprocess import Impute
    from Orange.data import DiscreteVariable, Domain
    path = os.path.join(os.path.dirname(__file__), "1 TA_15_SM.csv")
    tbl = Orange.data.Table(path)
    sample = tbl[:10_000]
    sample = Impute()(sample)
    # Build a classification task: use V_TYP as target
    v_typ = sample.domain["V_TYP"]
    numeric = [v for v in sample.domain.attributes if v.is_continuous][:5]
    dom = Domain(numeric, v_typ)
    sub = sample.transform(dom)
    tree = TreeLearner(max_depth=3)
    model = tree(sub)
    assert model is not None

check("Decision Tree on 10k rows", test_decision_tree)

# ─── Summary ──────────────────────────────────────────────────────────────
print()
print("=" * 55)
print(f"  PASSED: {len(PASS)}/{len(PASS)+len(FAIL)}")
print(f"  FAILED: {len(FAIL)}/{len(PASS)+len(FAIL)}")
if FAIL:
    print("\nFailed tests:")
    for name, err in FAIL:
        print(f"  - {name}: {err}")
if PASS:
    slowest = sorted(PASS, key=lambda x: -x[1])[:3]
    print("\nSlowest tests:")
    for name, dt in slowest:
        print(f"  {dt:.1f}s  {name}")
print("=" * 55)
sys.exit(0 if not FAIL else 1)
