import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QSplitter, QTreeView, QVBoxLayout, QWidget, QFileSystemModel
from PyQt5.QtWidgets import QSizePolicy, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

global ROOT_DIR
ROOT_DIR = 'C:\\Users\\bcduc\\OneDrive - Emergent Materials\\4_Portfolio\\ILL_IN8_Plotter\\Data'


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("파일 트리와 Bokeh 그래프 예제")
        self.setGeometry(100, 100, 800, 600)  # (x, y, 너비, 높이)

        # Splitter 생성
        splitter = QSplitter(self)
        splitter.setGeometry(0, 0, 800, 600)

        # 파일 트리 영역 설정
        file_tree_widget = QWidget()
        file_tree_layout = QVBoxLayout()

        # 파일 시스템 모델 생성
        self.file_system_model = QFileSystemModel()
        self.file_system_model.setRootPath(ROOT_DIR)  # 현재 디렉토리부터 시작

        # 파일 트리 뷰 생성 및 설정
        self.file_tree_view = QTreeView()
        self.file_tree_view.setModel(self.file_system_model)
        self.file_tree_view.setRootIndex(self.file_system_model.index(ROOT_DIR))  # 루트 디렉토리 설정

        file_tree_layout.addWidget(self.file_tree_view)
        file_tree_widget.setLayout(file_tree_layout)

        # 파일 트리 뷰에서 아이템 클릭 시 이벤트 핸들러 연결
        self.file_tree_view.clicked.connect(self.on_file_tree_clicked)

        file_tree_layout.addWidget(self.file_tree_view)
        file_tree_widget.setLayout(file_tree_layout)


        # Matplotlib 그래프 설정
        mpl_widget = QWidget()
        mpl_layout = QVBoxLayout()

        # 데이터 준비
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 4, 2, 3])

        
        # Matplotlib 그래프 생성
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(x, y)
        self.ax.set_title('Matplotlib Example')
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')

        # Matplotlib 그래프를 PyQt5에 통합하기 위한 Canvas 생성
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        mpl_layout.addWidget(self.canvas)
        mpl_widget.setLayout(mpl_layout)

        # 마우스 좌표를 표시할 QLabel 위젯
        self.coord_label = QLabel()
        mpl_layout.addWidget(self.coord_label)

        # 실시간 선을 그리기 위한 수평선과 수직선 초기화
        self.horizontal_line = self.ax.axhline(y=0, color='gray', linestyle='--', visible=False)
        self.vertical_line = self.ax.axvline(x=0, color='gray', linestyle='--', visible=False)

        # Matplotlib 그래프에 hover 이벤트 추가
        def on_hover(event):
            if event.inaxes:
                x, y = event.xdata, event.ydata
                self.coord_label.setText(f'좌표: ({x:.2f}, {y:.2f})')
                self.update_lines(x, y)

        self.fig.canvas.mpl_connect('motion_notify_event', on_hover)

        # Splitter에 위젯 추가
        splitter.addWidget(file_tree_widget)
        splitter.addWidget(mpl_widget)


        # Splitter 설정
        splitter.setSizes([200, 600])  # 초기 사이즈 설정 [파일 트리 영역 너비, Bokeh 그래프 영역 너비]
        splitter.setCollapsible(0, False)  # 파일 트리 영역은 축소 불가능하도록 설정
        splitter.setStretchFactor(1, 1)  # Bokeh 그래프 영역이 창 크기 조정 시 늘어나도록 설정

        # 메인 윈도우의 central widget 설정
        self.setCentralWidget(splitter)

    def update_lines(self, x, y):
        # 마우스 위치에 따라 수평선과 수직선 위치 업데이트
        if x is not None and y is not None:
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            self.horizontal_line.set_visible(True)
            self.vertical_line.set_visible(True)
        else:
            self.horizontal_line.set_visible(False)
            self.vertical_line.set_visible(False)

        self.canvas.draw()
        
    def on_file_tree_clicked(self, index):
        # 파일 트리 뷰에서 아이템 클릭 시 호출되는 함수
        file_path = self.file_system_model.filePath(index)
        if os.path.isfile(file_path):  # 파일일 경우에만 데이터를 읽음
            self.load_data(file_path)

    def load_data(self, file_path):
        # 파일에서 데이터 읽어와서 Matplotlib 그래프에 플로팅
        with open(file_path, 'r') as f:
            data = f.readlines()
    
         # 2번쨰 라인: 파일 이름
        # 16번 라인: 커맨드 라인
        # 60번 라인: 데이터 컬럼
        # 61번 라인 ~ 61 + np: 데이터
    
        data_name = data[1].split()[0]
        command_line = data[15].split(': ')[1]
        Q_position = data[16]

        # hkl 및 중성자빔 에너지는 소수점 2째자리에서 반올림 하기
        h = round(float(Q_position.split(': ')[1].strip().split(', ')[0].split()[-1]),1)
        k = round(float(Q_position.split(': ')[1].strip().split(', ')[1].split()[-1]),1)
        l = round(float(Q_position.split(': ')[1].strip().split(', ')[2].split()[-1]),1)
        E_start = round(float(Q_position.split(': ')[1].strip().split(', ')[3].split()[-1]), 1)
        print(f'h:{h}, k:{k}, l:{l}, E_start:{E_start}meV')

        dh = command_line.split(' ')[7]
        dk = command_line.split(' ')[8]
        dl = command_line.split(' ')[9]
        dE = command_line.split(' ')[10]
        np = int(command_line.split(' ')[12])
        mn = int(command_line.split(' ')[14].strip())

        col_names = data[59].strip().split('      ')
        col_names = [x.strip() for x in col_names]
        # print('col_names: ', col_names)/

        data_cut = data[60 : 60 + np]
        data_tmp = []
        for line in data_cut:
            data_tmp.append(line.strip().split())
            
        data_modified =  pd.DataFrame(data_tmp, columns=col_names)
        data_modified = data_modified.astype('float')
        
        total_scan_time = data_modified['TIME'].sum()/60 # min 단위로 표시
        print(f'total scan time: {total_scan_time} mins')

        data_info_dict = {'data_name': data_name, 'h': h, 'k':k, 'l':l, 'E_start': E_start}
        
        col_extract = ['PNT', 'EN', 'M1', 'CNTS']
        data_extracted = data_modified[col_extract]
        data_extracted['EN'] = data_extracted['EN'].apply(lambda x: round(x, 1))

        x = data_extracted['EN']
        y = data_extracted['CNTS']

        self.ax.clear()  # 기존 그래프 지우기
        self.ax.plot(x, y)
        self.ax.set_title('Matplotlib Example')
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')

        self.canvas.draw()  # 그래프 업데이트
        
        return data, data_modified, data_extracted, total_scan_time, data_info_dict
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
