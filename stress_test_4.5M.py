import time
import os
import sys
import psutil
import gc
import numpy as np

# Ensure Orange is in path
sys.path.insert(0, r"c:\Users\AG AUDIT\Desktop\orange3-master")

from AnyQt.QtWidgets import QApplication
from AnyQt.QtTest import QTest
from Orange.data.sql.duckdb_table import DuckDBTable
from Orange.widgets.visualize.owscatterplot import OWScatterPlot
from Orange.widgets.visualize.owdistributions import OWDistributions
from Orange.widgets.data.owtable import OWTable
from Orange.data import Domain, DiscreteVariable

def get_mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def run_stress_test():
    app = QApplication.instance() or QApplication([])
    csv_path = r"c:\Users\AG AUDIT\Desktop\1 TA_15_SM.csv"
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found at {csv_path}")
        return

    print("="*60)
    print(f"STARTING 4.5M ROW AUTOMATED STRESS TEST")
    print("="*60)
    print(f"Initial Memory Usage: {get_mem():.2f} MB")

    # 1. Loading DuckDBTable
    start = time.time()
    # Explicitly fetch categories for V_TYP to avoid contingency issues in test
    import duckdb
    conn = duckdb.connect(':memory:')
    cats = [str(r[0]) for r in conn.execute(f"SELECT DISTINCT V_TYP FROM read_csv_auto('{csv_path}') WHERE V_TYP IS NOT NULL").fetchall()]
    conn.close()
    
    table = DuckDBTable(f"read_csv_auto('{csv_path}')", type_hints=Domain([], None, [DiscreteVariable("V_TYP", values=cats)]))
    load_time = time.time() - start
    print(f"[LOAD] DuckDBTable initialized in {load_time:.2f}s. Rows: {len(table):,}")
    print(f"Memory after Load: {get_mem():.2f} MB")

    results = []

    # 2. Test Scatter Plot (Adaptive Sampling)
    print("\n[SCATTER PLOT] Testing Adaptive Sampling...")
    w_scatter = OWScatterPlot()
    w_scatter.show()
    
    start = time.time()
    w_scatter.set_data(table)
    w_scatter.handleNewSignals()
    initial_plot_time = time.time() - start
    print(f"  - Initial Plot Time (4.5M rows): {initial_plot_time:.2f}s")
    
    # Simulate Zoom-In
    print("  - Simulating Zoom (triggering adaptive re-sample)...")
    graph = w_scatter.graph
    start = time.time()
    # Set range to a small 10% window in the middle
    graph.view_box.setRange(xRange=(40, 60), yRange=(40, 60))
    # Adaptive sampling is on a timer (50ms), wait for it
    QTest.qWait(200) 
    zoom_update_time = time.time() - start
    print(f"  - Zoom Update Time: {zoom_update_time:.2f}s")
    
    results.append(("Scatter Plot Initial", initial_plot_time))
    results.append(("Scatter Plot Zoom", zoom_update_time))
    w_scatter.close()

    # 3. Test Distributions (Statistical Pushdown)
    print("\n[DISTRIBUTIONS] Testing Statistical Pushdown...")
    w_dist = OWDistributions()
    w_dist.show()
    
    start = time.time()
    w_dist.set_data(table)
    # Trigger first variable histogram
    w_dist.handleNewSignals()
    hist_time = time.time() - start
    print(f"  - Initial Histogram Time (Pushdown): {hist_time:.2f}s")
    
    # Toggle Split-By (V_TYP)
    print("  - Toggling 'Split By' (Contingency Pushdown)...")
    start = time.time()
    # Find V_TYP index in columns
    v_typ_idx = -1
    for i, var in enumerate(table.domain.variables + table.domain.metas):
        if var.name == "V_TYP":
            v_typ_idx = i
            break
    
    try:
        # Simulate selecting "Split By"
        v_typ_var = table.domain["V_TYP"]
        w_dist.cvar = v_typ_var
        w_dist._on_cvar_changed()
        QTest.qWait(100)
        split_time = time.time() - start
        print(f"  - Split-By Time (Pushdown): {split_time:.2f}s")
    except KeyError:
        print("  - WARNING: V_TYP not found in domain, skipping split test.")
        split_time = 0
    else:
        print("  - WARNING: V_TYP not found, skipping split test.")
        split_time = 0
        
    results.append(("Distributions Hist", hist_time))
    results.append(("Distributions Split", split_time))
    w_dist.close()

    # 4. Test Data Table (Lazy Loading)
    print("\n[DATA TABLE] Testing Lazy Loading...")
    w_table = OWTable()
    w_table.show()
    
    start = time.time()
    w_table.set_dataset(table)
    w_table.handleNewSignals()
    table_init_time = time.time() - start
    print(f"  - Data Table Initial Load: {table_init_time:.2f}s (Should be <0.5s)")

    # Seek deep
    print("  - Seeking Row 4,000,000 (Lazy Fetch)...")
    model = w_table.view.model()
    start = time.time()
    val = model.index(4000000, 0).data()
    seek_time = time.time() - start
    print(f"  - Row 4M Seek Time: {seek_time:.2f}s (Val: {val})")

    results.append(("Data Table Init", table_init_time))
    results.append(("Data Table Seek 4M", seek_time))
    w_table.close()

    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    for name, t in results:
        status = "PASS" if t < 15 else "SLOW" # 15s is generous for 4.5M rows
        print(f"{name:<25}: {t:.2f}s [{status}]")
    print(f"Final Memory Usage: {get_mem():.2f} MB")
    print("="*60)

if __name__ == "__main__":
    run_stress_test()
