from PyQt5.QtWidgets import QApplication
from gui import CellAnalyzerApp

def main():
    """
    主程序入口
    """
    app = QApplication([])
    window = CellAnalyzerApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()