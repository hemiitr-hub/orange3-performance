@echo off
cd /d "C:\Users\AG AUDIT\Desktop\orange3-master"
"C:\Users\AG AUDIT\AppData\Local\Programs\Python\Python311\python.exe" run_benchmarks.py > "C:\Users\AG AUDIT\Desktop\bench_out.txt" 2>&1
echo DONE >> "C:\Users\AG AUDIT\Desktop\bench_out.txt"
