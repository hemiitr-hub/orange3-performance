import duckdb
con = duckdb.connect(':memory:')
con.execute('CREATE TABLE iris (sl DOUBLE, sw DOUBLE, pl DOUBLE, pw DOUBLE, species VARCHAR)')
con.execute("INSERT INTO iris VALUES (1.0, 2.0, 3.0, 4.0, 's')")
con.execute("COPY iris TO 'test_iris.parquet' (FORMAT PARQUET)")
cur = con.execute("SELECT * FROM read_parquet('test_iris.parquet') LIMIT 0")
print([str(x[1]) for x in cur.description])
print(cur.description)
