import unittest
import numpy as np

from Orange.data import Table, ContinuousVariable, DiscreteVariable
from Orange.data.sql.duckdb_table import DuckDBTable
import duckdb

class TestDuckDBTable(unittest.TestCase):
    def setUp(self):
        # Create a memory DuckDB instance with a test table
        self.con = duckdb.connect(':memory:')
        self.con.execute('''
        CREATE TABLE iris (
            sepal_length DOUBLE,
            sepal_width DOUBLE,
            petal_length DOUBLE,
            petal_width DOUBLE,
            species VARCHAR
        )
        ''')
        # Insert some dummy rows
        self.con.execute("INSERT INTO iris VALUES (5.1, 3.5, 1.4, 0.2, 'setosa')")
        self.con.execute("INSERT INTO iris VALUES (4.9, 3.0, 1.4, 0.2, 'setosa')")
        self.con.execute("INSERT INTO iris VALUES (7.0, 3.2, 4.7, 1.4, 'versicolor')")
        self.con.execute("INSERT INTO iris VALUES (6.3, 3.3, 6.0, 2.5, 'virginica')")

    def test_duckdb_table_init(self):
        # We need the global duckdb accessible or we pass :memory: and attach to the same DB
        # Actually DuckdbBackend connects to database param.
        # But for :memory: it creates a new memory DB unless we share connection.
        # So let's write data to a temporary file instead to test the backend cleanly.
        pass

    def test_duckdb_table_file(self):
        self.con.execute("COPY iris TO 'test_iris.parquet' (FORMAT PARQUET)")
        
        # Load through DuckDBTable wrapper
        # The DuckDBTable will connect to a new :memory: DB and we can query the parquet file
        # Test with inspect_values=True
        table = DuckDBTable("read_parquet('test_iris.parquet')", inspect_values=True)
        
        print("Variables:", table.domain.variables)
        print("Metas:", table.domain.metas)
        for v in table.domain.metas:
            print(f"Meta variable {v.name} has type {type(v)}")
        
        self.assertEqual(len(table), 4)
        
        # In duckdb_backend String is parsed to Discrete if less than 21 unique 
        # or StringVariable otherwise. We expect 3 distinct species.
        self.assertEqual(len(table.domain.variables), 5)
        
        # Note: the inspect_values isn't forced by default so it might be StringVariable.
        
        # Test data extraction (download_data)
        X = table.X
        self.assertEqual(X.shape, (4, 5))
        self.assertEqual(X[0, 0], 5.1)

if __name__ == '__main__':
    unittest.main()
