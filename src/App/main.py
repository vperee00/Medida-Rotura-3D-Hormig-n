#!/usr/bin/env python3
import sys
import os
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QActionGroup, QFileDialog,
    QSizePolicy, QSplitter, QWidget, QLabel, QFrame,
    QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QMessageBox,
    QProgressDialog
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QProcess, QPointF
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush

# VTK imports
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# --------------------------------------------------------------------------------------------------

class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)
    def minimumSizeHint(self):
        return QSize(0, 0)

# --------------------------------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplicación para el análisis de rotura en probetas de hormigón")
        self.resize(1700, 900)

        self.left_image_path  = None
        self.right_image_path = None
        self.result_dir       = None
        self.roi_dir          = None

        # Buffers para picks
        self.last_pixel = []
        self.last_pts3d = []
        self.pts_flat   = None
        self.img_h = self.img_w = 0
        self.input_pixmap = None

        # Pixmap original
        self.input_pixmap = None

        self._create_menu()
        self._create_ui()
        
    # ----------------------------------------------------------------------------------------------

    def _create_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("Archivo")
        cargar_act = QAction("Cargar 3D", self)
        cargar_act.triggered.connect(self.load_model_from_files)
        fm.addAction(cargar_act)
        fm.addAction("Salir", self.close)

        mm = mb.addMenu("Modelos")
        mg = QActionGroup(self)
        act = QAction("VGGt", self, checkable=True); act.setChecked(True)
        mg.addAction(act); mm.addAction(act)
        mb.addAction("Ejecutar", self.run_model)
        
    # ----------------------------------------------------------------------------------------------

    def _prep_widget(self, w):
        w.setFrameShape(QFrame.Box)
        w.setAlignment(Qt.AlignCenter)
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        w.setMinimumSize(0,0)
        
    # ----------------------------------------------------------------------------------------------

    def _create_ui(self):
        root = QWidget(); 
        vl = QVBoxLayout(root)
        vl.setContentsMargins(5,5,5,5); 
        vl.setSpacing(10)

        # Selector de directorio
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Directorio de resultados:"))
        self.dir_edit = QLineEdit(); self.dir_edit.setReadOnly(True)
        btn = QPushButton("Seleccionar"); btn.clicked.connect(self.select_dir)
        hl.addWidget(self.dir_edit); hl.addWidget(btn)
        vl.addLayout(hl)
        
        # Selector de directorio ROI
        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel("Directorio de ROI:"))
        self.roi_edit = QLineEdit(); self.roi_edit.setReadOnly(True)
        btn2 = QPushButton("Seleccionar"); btn2.clicked.connect(self.select_roi_dir)
        hl2.addWidget(self.roi_edit); hl2.addWidget(btn2)
        vl.addLayout(hl2)

        # Panel superior: imágenes clicables
        self.left_lbl = ClickableLabel("Izquierda"); self._prep_widget(self.left_lbl)
        self.left_lbl.clicked.connect(self.load_left)
        self.right_lbl= ClickableLabel("Derecha");  self._prep_widget(self.right_lbl)
        self.right_lbl.clicked.connect(self.load_right)

        top = QSplitter(Qt.Horizontal)
        top.addWidget(self.left_lbl); top.addWidget(self.right_lbl)
        top.setStretchFactor(0,1); top.setStretchFactor(1,1)

        # Panel inferior izquierdo: información
        self.dist_label = QLabel("Distancia:\n0.000")
        self.dist_label.setFrameShape(QFrame.Box)
        self.dist_label.setAlignment(Qt.AlignLeft|Qt.AlignTop)
        self.dist_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dist_label.setMinimumSize(0,0)

        self.image_lbl = QLabel("Input Image")
        self.image_lbl.setFrameShape(QFrame.Box)
        self.image_lbl.setAlignment(Qt.AlignCenter)
        self.image_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_lbl.setMinimumSize(0,0)

        left_inner = QVBoxLayout()
        left_inner.addWidget(self.dist_label, 1)
        left_inner.addWidget(self.image_lbl, 2)
        left_bot = QWidget(); left_bot.setLayout(left_inner)

        # Panel inferior derecho: VTK 3D view
        self.vtk_widget = QVTKRenderWindowInteractor()
        self.vtk_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_bot = QFrame()
        vb = QVBoxLayout(right_bot); vb.setContentsMargins(0,0,0,0)
        vb.addWidget(self.vtk_widget)

        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(left_bot); bottom.addWidget(right_bot)
        bottom.setStretchFactor(0,1); bottom.setStretchFactor(1,2)

        # Layout principal con split vertical
        main_split = QSplitter(Qt.Vertical)
        main_split.addWidget(top); main_split.addWidget(bottom)
        main_split.setStretchFactor(0,1); main_split.setStretchFactor(1,1)
        vl.addWidget(main_split)
        
        # Label para mostrar el overlay ROI ---
        self.roi_lbl = QLabel(self)
        self.roi_lbl.setObjectName("roi_lbl")
        self.roi_lbl.setAlignment(Qt.AlignCenter)
        self.roi_lbl.setScaledContents(False)
        #self.roi_lbl.setPixmap(roi_pixmap)
        #self.roi_lbl.setMinimumSize(roi_pixmap.size())
        bottom.addWidget(self.roi_lbl)
        
        self.setCentralWidget(root)

        # Configuración VTK
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.renderer.SetBackground(0.1, 0.1, 0.1)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        self.picker = vtk.vtkPointPicker()
        self.interactor.SetPicker(self.picker)
        self.interactor.AddObserver("LeftButtonPressEvent", self.on_left_click)

        self.interactor.Initialize()
        self.vtk_widget.GetRenderWindow().Render()
        
    # ----------------------------------------------------------------------------------------------

    def select_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta", "")
        if d:
            self.result_dir = d
            self.dir_edit.setText(d)
            
    # ----------------------------------------------------------------------------------------------
            
    def select_roi_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta ROI", "")
        if d:
            self.roi_dir = d
            self.roi_edit.setText(d)
            
    # ----------------------------------------------------------------------------------------------

    def load_left(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Seleccionar Izquierda", "", "Images (*.png *.jpg *.jpeg)")
        if fn:
            self.left_image_path = fn
            pix = QPixmap(fn).scaled(self.left_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.left_lbl.setPixmap(pix)
            
    # ----------------------------------------------------------------------------------------------

    def load_right(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Seleccionar Derecha", "", "Images (*.png *.jpg *.jpeg)")
        if fn:
            self.right_image_path = fn
            pix = QPixmap(fn).scaled(self.right_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.right_lbl.setPixmap(pix)

	# ----------------------------------------------------------------------------------------------

    def run_model(self):
        if not (self.left_image_path and self.right_image_path and self.result_dir):
            QMessageBox.warning(self, "Error", "Carga imágenes y directorio primero.")
            return
        script = os.path.join(os.path.dirname(__file__), "run_vggt.py")
        args   = ["--left", self.left_image_path,
                  "--right",self.right_image_path,
                  "--out_dir",self.result_dir]
        dlg = QProgressDialog("Ejecutando modelo...", None, 0, 0, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setCancelButton(None)
        dlg.show()
        proc = QProcess(self)
        
        # Conectar esto para ver mensajes de error en STDERR
        proc.readyReadStandardError.connect(lambda: 
            print(proc.readAllStandardError().data().decode())
        )
        
        proc.setWorkingDirectory(self.result_dir)
        proc.setProgram(sys.executable)
        proc.setArguments([script] + args)
        proc.finished.connect(lambda c,s: (dlg.close(), self.on_model_done(c,s)))
        proc.start()
        self.progress_dialog = dlg
        self.model_process = proc
        
    # ----------------------------------------------------------------------------------------------
        
    def on_model_done(self, code, status):
        # Tras generar salida 3D, la cargamos y luego ejecutamos distancias
        self.load_model_from_files()
        self.calcular_distancias()
    
    # ----------------------------------------------------------------------------------------------
    
    def calcular_distancias(self):
        script = os.path.join(os.path.dirname(__file__), "vggt_Distancias.py")
        args = [
            "--input_dir",  self.result_dir,
            "--output_dir", self.result_dir,
            ]
        # Si no hay ROI seleccionado, no añadimos --roi_dir
        if self.roi_dir:
            args += ["--roi_dir", self.roi_dir]
        
        # Resto de parámetros siempre
        args += [
            "--camera",      "0",
            "--grad_thresh", "0.00175",
            "--rows_radius", "2",
            ]
                 
        dlg = QProgressDialog("Calculando distancias...", None, 0, 0, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setCancelButton(None)
        dlg.show()
        proc = QProcess(self)
        
        # Conectar esto para ver mensajes de error en STDERR
        proc.readyReadStandardError.connect(lambda: 
            print(proc.readAllStandardError().data().decode())
        )
        
        proc.setWorkingDirectory(self.result_dir)
        proc.setProgram(sys.executable)
        proc.setArguments([script] + args)
        proc.finished.connect(lambda c,s: (dlg.close(), self.on_distances_done(c,s)))
        proc.start()
        self.progress_dialog = dlg
        self.model_process = proc
        
    # ----------------------------------------------------------------------------------------------
        
    def on_distances_done(self, code, status):
        # Cargar overlay ROI tras cálculo de distancias
        sub = os.path.basename(os.path.normpath(self.result_dir))
        overlay = os.path.join(self.result_dir, f"{sub}_crack_overlay_roi.png")
        if os.path.exists(overlay):
            # Carga la imagen sin escalarla
            pix = QPixmap(overlay)
            self.roi_lbl.setScaledContents(False)
            self.roi_lbl.setPixmap(pix)
            # Ajusta el tamaño mínimo del label al de la imagen
            self.roi_lbl.setMinimumSize(pix.size())
        else:
            QMessageBox.warning(self, "Error", f"No se encontró {sub}_crack_overlay_roi.png")

	# ----------------------------------------------------------------------------------------------

    def load_model_from_files(self):  # Nueva función de carga
        if not self.result_dir:
            QMessageBox.warning(self, "Error", "Selecciona primero un directorio de resultados.")
            return
        npz = os.path.join(self.result_dir, "predictions.npz")
        img_path = os.path.join(self.result_dir, "input_vggt.png")
        if not os.path.exists(npz) or not os.path.exists(img_path):
            QMessageBox.warning(self, "Error", "Archivos de modelo 3D no encontrados en el directorio.")
            return
        data = np.load(npz)
        pts = data["world_points"][0,0]
        h, w, _ = pts.shape
        self.img_h, self.img_w = h, w
        self.pts_flat = pts.reshape(-1,3)

        img = Image.open(img_path).convert("RGB")
        if img.size != (w,h):
            img = img.resize((w,h), Image.BILINEAR)
        colors = np.asarray(img, np.uint8).reshape(-1,3)
        self.input_pixmap = QPixmap(img_path)

        vtk_pts    = vtk.vtkPoints()
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")
        for (x,y,z),(r,g,b) in zip(self.pts_flat, colors):
            vtk_pts.InsertNextPoint(x,y,z)
            vtk_colors.InsertNextTuple3(r,g,b)

        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_pts)
        poly.GetPointData().SetScalars(vtk_colors)
        mapper = vtk.vtkGlyph3DMapper()
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(0.005 * max(w,h) / 1000.0)
        mapper.SetSourceConnection(sphere.GetOutputPort())
        mapper.SetInputData(poly)
        mapper.ScalingOff()
        mapper.Update()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.interactor.Initialize()
        
        # Para que se muestre la imagen de input
        self._mostrar_input()

        sub = os.path.basename(os.path.normpath(self.result_dir))
        overlay = os.path.join(self.result_dir, f"{sub}_crack_overlay_roi.png")
        if os.path.exists(overlay):
            pix = QPixmap(overlay)
            self.roi_lbl.setScaledContents(False)
            self.roi_lbl.setPixmap(pix)
            self.roi_lbl.setMinimumSize(pix.size())
        else:
            # limpiar o dejar mensaje vacío si no existe aún
            self.roi_lbl.clear()
    
    # ----------------------------------------------------------------------------------------------
            
    def _mostrar_input(self):
        img = os.path.join(self.result_dir, "input_vggt.png")
        if os.path.exists(img):
           pix = QPixmap(img).scaled(
               self.image_lbl.size(),
               Qt.KeepAspectRatio,
               Qt.SmoothTransformation
           )
           self.image_lbl.setPixmap(pix)
        else:
           self.image_lbl.clear()
           
    # ----------------------------------------------------------------------------------------------

    def on_left_click(self, obj, event):
        x,y = self.interactor.GetEventPosition()
        self.picker.Pick(x,y,0,self.renderer)
        pid = self.picker.GetPointId()
        if pid < 0: return
        r = pid // self.img_w; c = pid % self.img_w
        self.last_pixel.insert(0,(r,c)); self.last_pixel = self.last_pixel[:2]
        pt3d = self.pts_flat[pid]
        self.last_pts3d.insert(0,pt3d); self.last_pts3d = self.last_pts3d[:2]
        if len(self.last_pts3d) >= 2:
            d = np.linalg.norm(self.last_pts3d[0] - self.last_pts3d[1])
        else:
            d = 0.0
        info_lines = [f"Distancia:\n{d:.3f}"]
        for idx, (pix, pt) in enumerate(zip(self.last_pixel, self.last_pts3d)):
            rr, cc = pix
            x3, y3, z3 = pt
            info_lines.append(
                f"Punto {idx+1}: Pixel ({rr}, {cc}), Profundidad: {z3:.3f}, Coord3D: ({x3:.3f}, {y3:.3f}, {z3:.3f})"
            )
        self.dist_label.setText("\n".join(info_lines))
        if self.input_pixmap:
            disp = self.input_pixmap.scaled(
                self.image_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            pw, ph = disp.width(), disp.height()
            fx, fy = pw/self.img_w, ph/self.img_h
            painter = QPainter(disp)
            for i,(rr,cc) in enumerate(self.last_pixel):
                x_lbl, y_lbl = cc*fx, rr*fy
                color = QColor('red') if i==0 else QColor('blue')
                painter.setPen(QPen(color,3)); painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(x_lbl, y_lbl), 6, 6)
            painter.end()
            self.image_lbl.setPixmap(disp)
        self.vtk_widget.GetRenderWindow().Render()

# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

# --------------------------------------------------------------------------------------------------
