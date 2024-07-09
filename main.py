import datetime
from re import search
from sqlite3 import connect, Error

from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, QDateTime
from cv2 import VideoCapture
from ultralytics import YOLO

from PyQt5.QtCore import QThread, pyqtSignal
from persiantools import digits
from sys import argv


# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# Initialize database
def initialize_database():
    conn = connect('license.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS licenses
                                          (user_id INTEGER PRIMARY KEY, code_meli INTEGER, name VARCHAR(50), plak VARCHAR(15), status VARCHAR(70), date TEXT)''')
    conn.commit()
    conn.close()




class WorkerThread(QThread):
    data_ready = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = YOLO("best.pt")  # Initialize YOLO model once
        self.conn = connect('license.db', check_same_thread=False)
        self.cursor = self.conn.cursor()

    def run(self):
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            results = self.model.predict(frame, imgsz=640,
                                         classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                  21, 22, 23, 24, 25, 26, 27])

            for r in results:
                boxes = r.boxes.xyxy.numpy()
                classes = r.boxes.cls.numpy()

            idx_sort = boxes[:, 0].argsort()
            plate = classes[idx_sort[::1]].astype(int).tolist()
            converter = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'ع',
                         11: 'الف', 12: 'ب', 13: 'د', 14: 'ق', 15: 'ه', 16: 'ج', 17: 'ل', 18: 'م', 19: 'ن', 20: 'plate',
                         21: 'ث', 22: 'ص', 23: 'س', 24: 'ت', 25: 'ط', 26: 'و', 27: 'ی'}
            try:
                plate_full = f"{plate[0]}{plate[1]} {converter[plate[2]]} {plate[3]}{plate[4]}{plate[5]} {plate[6]}{plate[7]}"

                if not search(
                        r"\b(?:[1-9]\d{1}|0[1-9]) [|الفآابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]{1,3} (?:[1-9]\d{2}|0[1-9]\d|100) (?:[1-9]\d{1}|0[1-9])\b",
                        plate_full):
                    plate_full = None
                    print("none")
            except IndexError:
                plate_full = None
            if plate_full:
                print(plate_full)
                self.check_plate_in_database(plate_full)

    def check_plate_in_database(self, plate_full):
        # if not search(
        #         r"\b(?:[1-9]\d{1}|0[1-9]) [|الفآابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]{1,3} (?:[1-9]\d{2}|0[1-9]\d|100) (?:[1-9]\d{1}|0[1-9])\b",
        #         plate_full):
        #     return

        try:
            query = "SELECT code_meli, name, plak, status, date FROM licenses WHERE plak LIKE ?"
            self.cursor.execute(query, (f"%{plate_full}%",))
            exists = self.cursor.fetchone()
            if exists:
                new_data = list(exists[:4])
                new_data.append(datetime.datetime.now())
                self.data_ready.emit(list(new_data))
            else:
                self.data_ready.emit(['ثبت نشده', 'نامشخص', plate_full, 'نامشخص', datetime.datetime.now()])
        except  Error as e:
            print(f"Database Error: {e}")

    def __del__(self):
        self.cursor.close()
        self.conn.close()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("./front/win1.ui", self)
        # self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setup_ui()

        self.worker_thread = WorkerThread()
        self.worker_thread.data_ready.connect(self.insert_data_at_last_row)
        self.worker_thread.start()

    def setup_ui(self):
        self.log = QPixmap('front/HakimSABZ.png')
        self.logo.setPixmap(self.log)

        self.image_timer = QTimer()
        self.image_timer.timeout.connect(self.update_image)
        self.image_timer.start(100)

        self.datetime_timer = QTimer()
        self.datetime_timer.timeout.connect(lambda: self.dateTimeEdit.setDateTime(QDateTime.currentDateTime()))
        self.datetime_timer.start(1000)

        self.open.clicked.connect(lambda: self.show_alert("باز", "این پیام برای باز است"))
        self.close.clicked.connect(lambda: self.show_alert("بسته", "این پیام برای بسته است"))

        self.save.clicked.connect(self.save_data)
        self.search.clicked.connect(self.search_clicked)
        self.search_2.clicked.connect(self.search_clicked_edit)

        self.report_plak.clicked.connect(self.report_plaks)
        self.setup_table()
        self.setup_del_plak_table()
        self.setup_table_edit()

    ################################################# Window edit Plak
    def search_clicked_edit(self):
        search_by_index = int(self.comboBox_3.currentIndex())
        search_term = self.search_box_2.text().strip()
        di = {0: 'code_meli', 1: 'name', 2: 'status', 3: 'plak', 4: 'date'}
        search_by = di.get(search_by_index)

        self.populate_table_edit(search_by, search_term)

    def setup_table_edit(self):
        # Set initial column headers
        self.editPlakTable.setColumnCount(5)  # Adjust based on your number of columns
        self.editPlakTable.setHorizontalHeaderLabels(['کد ملی', 'نام و نام خانواگی', 'پلاک', 'سمت', 'تاریخ'])

        # Disable editing of cell contents
        # self.editPlakTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # Set uniform width for all headers based on table width
        header = self.editPlakTable.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table_width = self.editPlakTable.width()
        header.setDefaultSectionSize(table_width // self.editPlakTable.columnCount())

        # Populate table with initial data
        self.populate_table_edit()

        self.editPlakTable.itemChanged.connect(self.cell_changed)

        # Connect context menu signal
        self.editPlakTable.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

    def cell_changed(self, item):
        """Handle changes in tableWidget cells."""
        row = item.row()
        col = item.column()
        new_value = item.text()
        column_names = {0: 'code_meli', 1: 'name', 2: 'plak', 3: 'status', 4: 'date'}
        column_name = column_names.get(col)

        self.confirm_and_update_db(row, column_name, new_value)

    def confirm_and_update_db(self, row, column_name, new_value):
        """Confirm and update the database with new value."""
        code_meli = self.editPlakTable.item(row, 0).text()

        reply = QtWidgets.QMessageBox.question(
            self, 'تایید',
            f"آیا مطمئن هستید که می‌خواهید مقدار را به '{new_value}' ویرایش کنید؟",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            self.update_row_in_db(code_meli, column_name, new_value)

    def update_row_in_db(self, code_meli, column_name, new_value):
        """Update a row in the database."""
        print(column_name, new_value)
        if (column_name == "code_meli" and (not search(r"\d{9,10}$", new_value))) or (
                column_name == "plak" and (
                not search(r"\b(?:[1-9]\d{1}|0[1-9]) \S{1,3} (?:[1-9]\d{2}|0[1-9]\d|100) (?:[1-9]\d{1}|0[1-9])\b",
                           new_value))):
            self.show_alert("ارور", "مقادر وارد شده برای تغییر اشتباه است")
        else:
            try:
                conn = connect('license.db')
                cursor = conn.cursor()
                print(column_name)
                query = f"UPDATE licenses SET {column_name} = ? WHERE code_meli = ?"
                cursor.execute(query, (new_value, code_meli))
                conn.commit()
                conn.close()
            except  Error as e:
                self.show_alert("Database Error", str(e))
                print(f"Database Error: {e}")

    def populate_table_edit(self, search_by=None, search_term=None):
        try:
            # Temporarily disconnect the itemChanged signal
            self.editPlakTable.blockSignals(True)

            conn = connect('license.db')
            cursor = conn.cursor()
            if search_by and search_term:
                query = f"SELECT code_meli, name, plak, status, date FROM licenses WHERE {search_by} LIKE ?"
                cursor.execute(query, (f"%{search_term}%",))
                self.error_3.setText("برای ویرایش دو بار راست کلیک کرده و آن را ادیت کنید")
            else:
                cursor.execute('SELECT code_meli, name, plak, status, date FROM licenses')

            data = cursor.fetchall()
            conn.close()

            if len(data) == 0 and search_by and search_term:
                self.show_alert("راهنما", "چیزی پیدا نشد")

            self.editPlakTable.setRowCount(len(data))
            for row, rowData in enumerate(data):
                for col, value in enumerate(rowData):
                    if col == 2:
                        item = QtWidgets.QTableWidgetItem("\u202B"+str(value))  # Convert to string for display
                    else:
                        item = QtWidgets.QTableWidgetItem(str(value))  # Convert to string for display

                    self.editPlakTable.setItem(row, col, item)

            if not search_by and not search_term:
                self.error_3.setText("")

            # Reconnect the itemChanged signal
            self.editPlakTable.blockSignals(False)

        except  Error as e:
            self.show_alert("Database Error", str(e))
            print(f"Database Error: {e}")

    ################################################# Window delete Plak
    def search_clicked(self):
        search_by_index = int(self.comboBox.currentIndex())
        search_term = self.search_box.text().strip()
        di = {0: 'code_meli', 1: 'name', 2: 'status', 3: 'plak', 4: 'date'}
        search_by = di.get(search_by_index)

        self.populate_del_table(search_by, search_term)

    def setup_del_plak_table(self):
        # Set initial column headers
        self.delPlakTable.setColumnCount(5)  # Adjust based on your number of columns
        self.delPlakTable.setHorizontalHeaderLabels(['کد ملی', 'نام و نام خانواگی', 'پلاک', 'سمت', 'تاریخ'])

        # Disable editing of cell contents
        self.delPlakTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # Set uniform width for all headers based on table width
        header = self.delPlakTable.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table_width = self.delPlakTable.width()
        header.setDefaultSectionSize(table_width // self.delPlakTable.columnCount())

        # Populate table with initial data
        self.populate_del_table()

        # Connect context menu signal
        self.delPlakTable.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.delPlakTable.customContextMenuRequested.connect(self.show_context_menu)

    def populate_del_table(self, search_by=None, search_term=None):
        try:
            conn = connect('license.db')
            cursor = conn.cursor()
            if search_by and search_term:
                query = f"SELECT code_meli, name, plak, status, date FROM licenses WHERE {search_by} LIKE ?"
                cursor.execute(query, (f"%{search_term}%",))
                self.error_2.setText("برای حذف کلیک راست کنید و حذف کنید")
            else:
                cursor.execute('SELECT code_meli, name, plak, status, date FROM licenses')

            data = cursor.fetchall()
            conn.close()

            if len(data) == 0 and search_by and search_term:
                self.show_alert("راهنما", "چیزی پیدا نشد")

            self.delPlakTable.setRowCount(len(data))
            for row, rowData in enumerate(data):
                for col, value in enumerate(rowData):
                    print(row, col, value)
                    if col == 2:
                        item = QtWidgets.QTableWidgetItem("\u202B"+str(value))  # Convert to string for display
                    else:
                        item = QtWidgets.QTableWidgetItem(str(value))  # Convert to string for display

                    self.delPlakTable.setItem(row, col, item)

            if not search_by and not search_term:
                self.error_2.setText("")

        except  Error as e:
            self.show_alert("Database Error", str(e))
            print(f"Database Error: {e}")

    def show_context_menu(self, pos):
        # Get the index of the cell where the context menu was requested
        index = self.delPlakTable.indexAt(pos)

        if index.isValid():
            menu = QtWidgets.QMenu(self)
            delete_action = menu.addAction("حذف ردیف")
            action = menu.exec_(self.delPlakTable.viewport().mapToGlobal(pos))
            if action == delete_action:
                self.confirm_and_delete_row(index.row())

    def confirm_and_delete_row(self, row):
        code_meli = self.delPlakTable.item(row, 0).text()
        reply = QtWidgets.QMessageBox.question(self, 'تایید',
                                               "آیا مطمئن هستید که می‌خواهید این ردیف را حذف کنید؟",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            self.remove_row_from_db(code_meli)
            self.delPlakTable.removeRow(row)
            self.populate_del_table()

    def remove_row_from_db(self, code_meli):
        try:
            conn = connect('license.db')
            cursor = conn.cursor()
            cursor.execute("DELETE FROM licenses WHERE code_meli = ?", (code_meli,))
            conn.commit()
            conn.close()
        except  Error as e:
            self.show_alert("Database Error", str(e))
            print(f"Database Error: {e}")

    ################################################# Window Sign Plak
    def updateDateTime(self):
        self.dateTimeEdit.setDateTime(QDateTime.currentDateTime())

    def TwoDigitSpinBox(self, value):
        return f"{value:02d}"

    def ThreeDigitSpinBox(self, value):
        return f"{value:03d}"

    def save_data(self):
        # Example: Assuming you have QLineEdit widgets named input1 and input2 in your UI
        name = self.name.text()
        status = self.status.currentText()
        plak_1 = self.TwoDigitSpinBox(self.spinBox.value())
        plak_2 = self.comboBox_2.currentText()
        plak_3 = self.ThreeDigitSpinBox(self.spinBox_2.value())
        plak_4 = self.TwoDigitSpinBox(self.spinBox_3.value())
        try:
            code_meli = int(self.code_meli.text())
        except ValueError:
            code_meli = ""
        plak = f"{plak_1} {plak_2} {plak_3} {plak_4}"
        print(name, status, plak, code_meli)

        # Check is it true inputs!
        if plak_2 == " " or len(str(code_meli)) < 9 or name == "" or status == " ":
            self.error.setText("لطفا مقادیر درست و صحیح وارد کنید")
            self.show_alert("ارور", "مقادیر وارد شده اشتباه است")
        else:
            conn = connect('license.db')
            cursor = conn.cursor()
            cursor.execute(
                f'INSERT INTO licenses (code_meli, name, plak, status, date) VALUES ({code_meli}, "{name}", "{plak}", "{status}", "{datetime.datetime.now()}")')
            conn.commit()
            conn.close()
            self.error.setText("ثبت شد")
            self.show_alert("موفق", "با موفقیت ثبت شد")

    ################################################# Button report export csv
    def get_table_data(self):
        row_count = self.tableWidget.rowCount()
        column_count = self.tableWidget.columnCount()
        data = []

        for row in range(row_count):
            row_data = []
            for column in range(column_count):
                item = self.tableWidget.item(row, column)
                row_data.append(item.text() if item is not None else '')
            data.append(row_data)

        return data

    def report_plaks(self):
        data = self.get_table_data()
        self.export_to_csv(data)

    def export_to_csv(self, data):
        # print(data)
        if not data:
            self.show_alert("ارور", "هیچ پلاکی برای گزارش گیری در جدول موجود نیست")
            return

        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Save CSV", "", "CSV Files (*.csv);;All Files (*)",
                                                             options=options)
        if file_path:
            try:
                with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(['کد ملی', 'نام و نام خانواگی', 'پلاک', 'سمت', 'تاریخ'])  # Writing header
                    writer.writerows(data)
                print(f"Data exported successfully to {file_path}")
            except Exception as e:
                print(f"Error exporting data: {e}")

    def insert_data_at_last_row(self, new_data):
        row_count = self.tableWidget.rowCount()

        for row in range(row_count):
            if self.tableWidget.item(row, 2).text() == new_data[2]:  # Compare the license plate
                return  # Data already exists
        row_count = self.tableWidget.rowCount()
        self.tableWidget.insertRow(row_count)
        for col, value in enumerate(new_data):
            item = QtWidgets.QTableWidgetItem("\u202B"+str(value))
            self.tableWidget.setItem(row_count, col, item)

    def setup_table(self):
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setHorizontalHeaderLabels(['کد ملی', 'نام و نام خانواگی', 'پلاک', 'سمت', 'تاریخ مشاهده'])
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table_width = self.tableWidget.width()
        header.setDefaultSectionSize(table_width // self.tableWidget.columnCount())
        # self.populate_del_table()
        # self.table_update_timer = QTimer()
        # self.table_update_timer.timeout.connect(self.populate_del_table)
        # self.table_update_timer.start(5000)
        # self.tableWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # self.tableWidget.customContextMenuRequested.connect(self.show_context_menu)

    def show_alert(self, head, message):
        alert = QtWidgets.QMessageBox()
        alert.setWindowTitle(head)
        alert.setText(message)
        alert.setIcon(QtWidgets.QMessageBox.Information)
        alert.exec_()

    def update_image(self):
        ret, frame = cap.read()
        if ret:
            height, width, channel = frame.shape
            bytesPerLine = channel * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            qPixmap = QPixmap.fromImage(qImg)
            self.camera.setPixmap(qPixmap)
        else:
            self.show_alert("Error", "Failed to capture frame from webcam")
            # self.close()

if __name__ == "__main__":
    initialize_database()
    cap = VideoCapture(0)
    app = QtWidgets.QApplication(argv)
    window = MainWindow()
    window.show()
    app.exec()
