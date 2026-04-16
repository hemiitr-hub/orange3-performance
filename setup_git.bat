@echo off
cd /d "C:\Users\AG AUDIT\Desktop\orange3-master"
git init
git config user.email "hem.iitr@gmail.com"
git config user.name "hemchand jajoria"
git add .
git commit -m "perf: widget performance optimizations and crash fixes

- Table.transform() identity short-circuit
- OWLinePlot: dynamic lazy loading, scroll wheel, vectorized internals, crash fix
- OWHeatMap: vectorized column_str_from_table
- OWMosaic: cache prior_distribution outside draw_data loop
- OWMergeData: vectorized instance-ID extraction via data.ids
- OWSelectRows: suppress Qt repaints during bulk add_all
- OWFeatureStatistics: QTimer-based histogram pre-warming
- OWVennDiagram: fix string-column crash in get_unique_values
- Add PERFORMANCE.md and update README.md with fork notice"
echo DONE > "C:\Users\AG AUDIT\Desktop\git_init_done.txt"
