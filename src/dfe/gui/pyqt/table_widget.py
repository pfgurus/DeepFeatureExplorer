from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QRadioButton, QButtonGroup, QVBoxLayout, QWidget


class SelectionTableWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 800)

        # Create the table
        self.table = QTableWidget(0, 3)  # 5 rows, 4 columns
        self.table.setHorizontalHeaderLabels(["Name", "Resolution", "Channels",])
        self.table.setColumnWidth(1, 75)
        self.table.setColumnWidth(2, 75)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)

    def add_item(self, name: str, resolution: int,num_channels: int, index: int):
        self.table.insertRow(index)
        self.table.setItem(index, 0, QTableWidgetItem(f"{name}"))
        self.table.setItem(index, 1, QTableWidgetItem(f"{resolution}"))
        self.table.setItem(index, 2, QTableWidgetItem(f"{num_channels}"))



# Run the application
if __name__ == "__main__":
    app = QApplication([])
    window = TableWithRadioButtons()
    window.setWindowTitle("Table with Exclusive Radio Buttons")
    window.resize(600, 400)
    window.show()
    app.exec_()
