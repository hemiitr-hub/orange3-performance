import logging
from contextlib import contextmanager

try:
    import duckdb
except ImportError:
    duckdb = None

from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable
from Orange.data.sql.backend.base import Backend, ToSql, BackendError

log = logging.getLogger(__name__)

class DuckdbBackend(Backend):
    display_name = "DuckDB"
    
    def __init__(self, connection_params):
        if duckdb is None:
            raise BackendError("duckdb is not installed")
            
        super().__init__(connection_params)
        
        db_path = connection_params.get("database", ":memory:")
        try:
            self.conn = duckdb.connect(db_path)
        except Exception as ex:
            raise BackendError(str(ex)) from ex

    def create_sql_query(self, table_name, fields, filters=(),
                         group_by=None, order_by=None,
                         offset=None, limit=None,
                         use_time_sample=None):
        sql = ["SELECT", ', '.join(fields),
               "FROM", table_name]
        if use_time_sample is not None:
            # DuckDB uses percentages or row counts for TABLESAMPLE, fallback to USING SAMPLE
            sql.append(f"USING SAMPLE {use_time_sample}% (bernoulli)")
        if filters:
            sql.extend(["WHERE", " AND ".join(filters)])
        if group_by is not None:
            sql.extend(["GROUP BY", ", ".join(group_by)])
        if order_by is not None:
            sql.extend(["ORDER BY", ",".join(order_by)])
        if limit is not None:
            sql.extend(["LIMIT", str(limit)])
        if offset is not None:
            sql.extend(["OFFSET", str(offset)])
        return " ".join(sql)

    @contextmanager
    def execute_sql_query(self, query, params=None):
        try:
            cur = self.conn.cursor()
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            yield cur
        except Exception as ex:
            raise BackendError(str(ex)) from ex

    def quote_identifier(self, name):
        # Do not quote function calls like read_parquet() or read_csv() used as table names in DuckDB
        if '(' in name and ')' in name:
            return name
        return '"%s"' % name

    def unquote_identifier(self, quoted_name):
        if quoted_name.startswith('"') and quoted_name.endswith('"'):
            return quoted_name[1:-1]
        return quoted_name

    def list_tables_query(self, schema=None):
        if schema:
            return f"SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema='{schema}'"
        return "SELECT table_schema, table_name FROM information_schema.tables"

    def n_tables_query(self, schema=None) -> str:
        if schema:
            return f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='{schema}'"
        return "SELECT COUNT(*) FROM information_schema.tables"

    def create_variable(self, field_name, field_metadata,
                        type_hints, inspect_table=None):
        if field_name in type_hints:
            var = type_hints[field_name]
        else:
            var = self._guess_variable(field_name, field_metadata,
                                       inspect_table)

        field_name_q = self.quote_identifier(field_name)
        if isinstance(var, TimeVariable):
            var.to_sql = ToSql(f"EXTRACT(EPOCH FROM {field_name_q})")
        elif var.is_continuous:
            var.to_sql = ToSql(f"CAST({field_name_q} AS DOUBLE)")
        else:
            var.to_sql = ToSql(f"CAST({field_name_q} AS VARCHAR)")
        return var

    def _guess_variable(self, field_name, field_metadata, inspect_table):
        type_code = field_metadata[0] 
        
        if type_code is None:
            return StringVariable.make(field_name)
            
        type_str = str(type_code).upper()
        
        if any(x in type_str for x in ('FLOAT', 'DOUBLE', 'DECIMAL', 'NUMERIC')):
            return ContinuousVariable.make(field_name)
            
        if any(x in type_str for x in ('TIME', 'DATE')):
            tv = TimeVariable.make(field_name)
            tv.have_date = 'DATE' in type_str or 'TIMESTAMP' in type_str
            tv.have_time = 'TIME' in type_str or 'TIMESTAMP' in type_str
            return tv
            
        if any(x in type_str for x in ('INT', 'BIGINT', 'TINYINT', 'SMALLINT')):
            if inspect_table:
                values = self.get_distinct_values(field_name, inspect_table)
                if values:
                    return DiscreteVariable.make(field_name, values)
            return ContinuousVariable.make(field_name)
            
        if 'BOOL' in type_str:
            return DiscreteVariable.make(field_name, ['false', 'true'])
            
        if any(x in type_str for x in ('VARCHAR', 'CHAR', 'TEXT', 'STRING')):
            if inspect_table:
                values = self.get_distinct_values(field_name, inspect_table)
                if values:
                    return DiscreteVariable.make(field_name, values)
                    
        return StringVariable.make(field_name)

    def count_approx(self, query):
        sql = f"SELECT COUNT(*) FROM ({query}) AS sub"
        with self.execute_sql_query(sql) as cur:
            return cur.fetchone()[0]

    def distinct_values_query(self, field_name: str, table_name: str) -> str:
        fields = [self.quote_identifier(field_name)]
        return self.create_sql_query(
            table_name, fields, group_by=fields, order_by=fields, limit=21
        )
