from typing import Type
from collections import OrderedDict

from AnyQt.QtWidgets import QLineEdit, QSizePolicy

from Orange.data import Table
from Orange.data.sql.backend import Backend
from Orange.data.sql.backend.base import BackendError
from Orange.widgets import gui, report
from Orange.widgets.credentials import CredentialManager
from Orange.widgets.settings import Setting
from Orange.widgets.utils.signals import Output
from Orange.widgets.widget import OWWidget, Msg


class OWBaseSql(OWWidget, openclass=True):
    """Base widget for connecting to a database.
    Override `get_backend` when subclassing to get corresponding backend.
    """
    class Outputs:
        data = Output("Data", Table)

    class Error(OWWidget.Error):
        connection = Msg("{}")

    want_main_area = False
    resizing_enabled = False

    host = Setting(None)  # type: Optional[str]
    port = Setting(None)  # type: Optional[str]
    database = Setting(None)  # type: Optional[str]
    schema = Setting(None)  # type: Optional[str]
    username = ""
    password = ""

    def __init__(self):
        super().__init__()
        self.backend = None  # type: Optional[Backend]
        self.data_desc_table = None  # type: Optional[Table]
        self.database_desc = None  # type: Optional[OrderedDict]
        self._setup_gui()
        self.connect()

    def _setup_gui(self):
        self.controlArea.setMinimumWidth(360)

        vbox = gui.vBox(self.controlArea, "Server")
        self.serverbox = gui.vBox(vbox)
        self.servertext = QLineEdit(self.serverbox)
        self.servertext.setPlaceholderText("Server")
        self.servertext.setToolTip("Server")
        self.servertext.editingFinished.connect(self._load_credentials)
        if self.host:
            self.servertext.setText(self.host if not self.port else
                                    "{}:{}".format(self.host, self.port))
        self.serverbox.layout().addWidget(self.servertext)

        db_box = gui.hBox(self.serverbox)
        self.databasetext = QLineEdit(db_box)
        self.databasetext.setPlaceholderText("Database[/Schema]")
        self.databasetext.setToolTip("Database or optionally Database/Schema")
        if self.database:
            self.databasetext.setText(
                self.database if not self.schema else
                "{}/{}".format(self.database, self.schema))
        db_box.layout().addWidget(self.databasetext)
        
        # Add browse button for DuckDB/file backends
        self.browse_button = gui.button(db_box, self, "...", callback=self.browse_file)
        self.browse_button.setMaximumWidth(30)
        self.browse_button.setToolTip("Browse for local file (DuckDB/SQLite)")
        # Usually hidden, shown when DuckDB is selected
        self.browse_button.hide()
        self.usernametext = QLineEdit(self.serverbox)
        self.usernametext.setPlaceholderText("Username")
        self.usernametext.setToolTip("Username")

        self.serverbox.layout().addWidget(self.usernametext)
        self.passwordtext = QLineEdit(self.serverbox)
        self.passwordtext.setPlaceholderText("Password")
        self.passwordtext.setToolTip("Password")
        self.passwordtext.setEchoMode(QLineEdit.Password)

        self.serverbox.layout().addWidget(self.passwordtext)

        self._load_credentials()

        self.connectbutton = gui.button(self.serverbox, self, "Connect",
                                        callback=self.connect)
        self.connectbutton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def browse_file(self):
        from AnyQt.QtWidgets import QFileDialog
        filters = "Any File (*.*);;Parquet Files (*.parquet);;CSV Files (*.csv)"
        filename, _ = QFileDialog.getOpenFileName(self, "Open Database File", "", filters)
        if filename:
            self.databasetext.setText(filename)

    def _load_credentials(self):
        self._parse_host_port()
        cm = self._credential_manager(self.host, self.port)
        self.username = cm.username
        self.password = cm.password

        if self.username:
            self.usernametext.setText(self.username)
        if self.password:
            self.passwordtext.setText(self.password)

    def _save_credentials(self):
        cm = self._credential_manager(self.host, self.port)
        cm.username = self.username or ""
        cm.password = self.password or ""

    @staticmethod
    def _credential_manager(host, port):
        return CredentialManager("SQL Table: {}:{}".format(host, port))

    def _parse_host_port(self):
        hostport = self.servertext.text().split(":")
        self.host = hostport[0]
        self.port = hostport[1] if len(hostport) == 2 else None

    def _check_db_settings(self):
        self._parse_host_port()
        self.database, _, self.schema = self.databasetext.text().partition("/")
        self.username = self.usernametext.text() or None
        self.password = self.passwordtext.text() or None

    def connect(self):
        self.clear()
        self._check_db_settings()
        backend = self.get_backend()
        if backend is None:
            return
            
        # If duckdb backend, we only need the database (which can be a file path or :memory:)
        is_duckdb = getattr(backend, 'display_name', '') == "DuckDB"
        
        if not is_duckdb and (not self.host or not self.database):
            return
            
        try:
            self.backend = backend(dict(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            ))
            self.on_connection_success()
        except BackendError as err:
            self.on_connection_error(err)

    def get_backend(self) -> Type[Backend]:
        """
        Derived widgets should override this to get corresponding backend.

        Returns
        -------
        backend: Type[Backend]
        """
        raise NotImplementedError

    def on_connection_success(self):
        self._save_credentials()
        self.database_desc = OrderedDict((
            ("Host", self.host), ("Port", self.port),
            ("Database", self.database), ("User name", self.username)
        ))

    def on_connection_error(self, err):
        error = str(err).split("\n")[0]
        self.Error.connection(error)

    def open_table(self):
        data = self.get_table()
        self.data_desc_table = data
        self.Outputs.data.send(data)

    def get_table(self) -> Table:
        """
        Derived widgets should override this to get corresponding table.

        Returns
        -------
        table: Table
        """
        raise NotImplementedError

    def clear(self):
        self.Error.connection.clear()
        self.database_desc = None
        self.data_desc_table = None
        self.Outputs.data.send(None)

    def send_report(self):
        if not self.database_desc:
            self.report_paragraph("No database connection.")
            return
        self.report_items("Database", self.database_desc)
        if self.data_desc_table:
            self.report_items("Data",
                              report.describe_data(self.data_desc_table))
