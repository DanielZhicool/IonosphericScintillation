import sys

from PySide6.QtWidgets import QApplication

from gui.main_window import Uran4App


def main():
    app = QApplication(sys.argv)
    window = Uran4App()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
