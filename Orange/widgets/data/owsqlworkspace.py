from AnyQt.QtWidgets import QTextEdit, QVBoxLayout, QPushButton, QMessageBox
from AnyQt.QtCore import Qt

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.data import Table
import duckdb

try:
    from Orange.data.polars_compat import table_to_polars, table_from_polars
except ImportError:
    pass

class OWSQLWorkspace(widget.OWWidget, ConcurrentWidgetMixin):
    name = "SQL Workspace"
    description = "Execute raw DuckDB SQL queries against the upstream data stream."
    icon = "icons/SQL.svg"  
    priority = 100

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    want_main_area = False
    
    sql_query = Setting("SELECT * FROM input_table LIMIT 100")

    def __init__(self):
        widget.OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        
        self.data = None
        self.error_msg = None
        
        box = gui.widgetBox(self.controlArea, "DuckDB Interactive Workspace")
        
        self.editor = QTextEdit(self)
        self.editor.setPlainText(self.sql_query)
        self.editor.textChanged.connect(self._on_text_changed)
        box.layout().addWidget(self.editor)
        
        self.execute_btn = gui.button(box, self, "Execute Query", callback=self.execute_query)
        self.execute_btn.setStyleSheet("font-weight: bold;")
        
        self.info_label = gui.widgetLabel(box, "Connect data to register 'input_table'.")

    def _on_text_changed(self):
        self.sql_query = self.editor.toPlainText()

    @Inputs.data
    def set_data(self, data):
        self.data = data
        if data is not None:
            self.info_label.setText(f"Connected: {len(data)} rows available as 'input_table'")
        else:
            self.info_label.setText("No data. 'input_table' is not registered.")
            self.Outputs.data.send(None)

    def execute_query(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please connect a dataset first.")
            return
            
        query = self.editor.toPlainText().strip()
        if not query:
            return
            
        try:
            # Prepare dataframe in main thread
            df = self._get_dataframe()
        except Exception as e:
            QMessageBox.critical(self, "SQL Error", str(e))
            return
            
        self.execute_btn.setEnabled(False)
        self.info_label.setText("Executing query...")
        
        # Start background task
        self.start(self._run_query, query, df)

    def _run_query(self, query, df, state: TaskState):
        state.set_status("Executing DuckDB query...")
        input_table = df
        con = duckdb.connect(':memory:')
        
        if state.is_interruption_requested():
            return None
            
        result_df = con.query(query).to_df()
        
        state.set_status("Converting result to Orange Table...")
        from Orange.data.pandas_compat import table_from_frame
        result_table = table_from_frame(result_df)
        
        return result_table

    def on_done(self, result: Table):
        self.execute_btn.setEnabled(True)
        if result is not None:
            self.Outputs.data.send(result)
            self.info_label.setText(f"Query successful: {len(result)} rows generated.")

    def on_exception(self, ex: Exception):
        self.execute_btn.setEnabled(True)
        self.Outputs.data.send(None)
        self.info_label.setText("Query Execution Error")
        self.error(str(ex))
        
    def _get_dataframe(self):
        try:
            from Orange.data.polars_compat import table_to_polars
            return table_to_polars(self.data)
        except ImportError:
            from Orange.data.pandas_compat import table_to_frame
            return table_to_frame(self.data)

if __name__ == "__main__":
    from AnyQt.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    w = OWSQLWorkspace()
    w.show()
    sys.exit(app.exec())
