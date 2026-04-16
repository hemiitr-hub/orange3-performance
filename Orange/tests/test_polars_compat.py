import unittest
import numpy as np
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
try:
    import polars as pl
    from Orange.data.polars_compat import table_to_polars, table_from_polars
except ImportError:
    pl = None

class TestPolarsCompat(unittest.TestCase):
    @unittest.skipIf(pl is None, "polars is not installed")
    def test_conversion(self):
        # Create a simple Orange Table
        attr1 = ContinuousVariable("f1")
        attr2 = DiscreteVariable("f2", values=["a", "b", "c"])
        domain = Domain([attr1, attr2])
        
        X = np.array([
            [1.5, 0.0],
            [2.5, 1.0],
            [np.nan, 2.0]
        ])
        
        table = Table.from_numpy(domain, X)
        
        # Convert to polars
        pldf = table_to_polars(table)
        
        self.assertEqual(pldf.shape, (3, 2))
        self.assertEqual(pldf.columns, ["f1", "f2"])
        
        # Convert back
        new_table = table_from_polars(pldf)
        
        self.assertEqual(len(new_table), 3)
        self.assertEqual(len(new_table.domain.attributes), 2)
        
        # Verify data
        np.testing.assert_array_almost_equal(new_table.X[:2, 0], [1.5, 2.5])
        self.assertTrue(np.isnan(new_table.X[2, 0]))

if __name__ == '__main__':
    unittest.main()
