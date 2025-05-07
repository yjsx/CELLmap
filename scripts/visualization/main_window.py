import sys
import numpy as np
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QComboBox,
                             QStackedWidget, QLabel, QSizePolicy)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QVector3D, QVector4D
import os
import subprocess
import rospy
from nav_msgs.msg import Odometry
import threading
import time
class CustomGLView(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.main_window = parent  # Keep reference to main window
    
    def mousePressEvent(self, event):
        super().mousePressEvent(event)  # Maintain parent class camera control
        self.main_window.handle_mouse_press(event)  # Forward event
    
    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)  # Maintain parent class camera control 
        self.main_window.handle_mouse_release(event)  # Forward event

class CELLmap(object):
    def __init__(self):
        self.traj_file = None
        self.rosbag_file = None
        self.pcd_folder = None
        self.lidar_topic = None
        self.out_folder = None
        self.raw_traj = []
        self.opti_traj = []
    def odom_callback(self, msg):
        # Here we can process the received Odometry message
        # For example, print position information
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.raw_traj.append([msg.header.stamp.to_sec(), position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w])
  
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout divided into left and right parts
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Left button panel (using stacked layout)
        self.button_stack = QStackedWidget()
        main_layout.addWidget(self.button_stack, stretch=1)
        
        # Right 3D view
        self.view = CustomGLView(parent=self)
        main_layout.addWidget(self.view, stretch=4)

        # Initialize 3D scene
        self.init_3d_scene()
        
        # Initialize button panels
        self.init_main_buttons()
        self.init_mapping_buttons()
        self.init_localization_buttons()
        
        # Data storage
        self.points = []
        self.lines = []
        self.selected_points = []
        
        # Current mode state
        self.line_mode = False

        self.mouse_press_pos = None
        self.mouse_pressed = False

        self.CELLmap_agent = CELLmap()
      
    def init_3d_scene(self):
        """Initialize 3D scene"""
        grid = gl.GLGridItem()
        self.view.addItem(grid)
        self.scatter = gl.GLScatterPlotItem()
        self.view.addItem(self.scatter)
        
        # Set initial view
        self.view.setCameraPosition(distance=15)

    def init_main_buttons(self):
        """Initialize main button panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        buttons = [
            ('Add random', self.add_random_point),
            ('Add line', self.enter_line_mode),
            ('Mapping', lambda: self.button_stack.setCurrentIndex(1)),
            ('Localization', lambda: self.button_stack.setCurrentIndex(2)),
        ]
        
        for text, callback in buttons:
            btn = QPushButton(text)
            btn.setFixedHeight(40)
            btn.setStyleSheet("QPushButton {font-size: 14px;}")
            btn.clicked.connect(callback)
            layout.addWidget(btn)
        
        layout.addStretch()
        self.button_stack.addWidget(widget)

    def init_mapping_buttons(self):
        """Initialize Mapping function button panel"""

        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Trajectory selection
        self.btn_traj = QPushButton("Trajectory")
        self.btn_traj.clicked.connect(lambda: self.select_file("trajectory"))
        layout.addWidget(self.btn_traj)
        
        # Rosbag selection
        h_widget = QWidget()
        h_layout = QHBoxLayout(h_widget)
        h_layout.setContentsMargins(0, 0, 0, 0)  # Remove margin

        self.btn_rosbag = QPushButton("Rosbag")
        self.btn_rosbag.clicked.connect(lambda: self.select_file("rosbag"))
        
        self.btn_pcds = QPushButton("PCDs")
        self.btn_pcds.clicked.connect(lambda: self.select_directory("pcds"))

        # Add buttons to horizontal layout
        h_layout.addWidget(self.btn_rosbag)
        h_layout.addWidget(self.btn_pcds)
        
        # Add horizontal container to main layout
        layout.addWidget(h_widget)
        btn_outfolder = QPushButton("OutFolder")
        btn_outfolder.clicked.connect(lambda: self.select_directory("outfolder"))
        layout.addWidget(btn_outfolder)

        # btn_pcds = QPushButton("PCDs")
        # btn_pcds.clicked.connect(lambda: self.select_file("pcds"))
        # layout.addWidget(btn_pcds)
        
        # LiDAR Topic dropdown menu
        self.lidar_combo = QComboBox()
        self.lidar_combo.addItems(["topic1", "topic2", "topic3"])
        self.lidar_combo.currentIndexChanged.connect(self.handle_lidar_change)
        layout.addWidget(QLabel("LiDAR Topic:"))
        layout.addWidget(self.lidar_combo)
        
        # Start button
        self.btn_start = QPushButton("Start")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.mapping_start)
        layout.addWidget(self.btn_start)
        
        # Add Loop Closure button
        self.btn_loop = QPushButton("Add Loop Closure")
        self.btn_loop.setEnabled(False)
        layout.addWidget(self.btn_loop)
        
        # Back button
        btn_back = QPushButton("Back")
        btn_back.clicked.connect(lambda: self.button_stack.setCurrentIndex(0))
        layout.addWidget(btn_back)
        
        layout.addStretch()
        self.button_stack.addWidget(widget)

        btn_back.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)

    def init_localization_buttons(self):
        """Initialize Localization function button panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Map selection
        btn_map = QPushButton("Map")
        btn_map.clicked.connect(self.select_directory)
        layout.addWidget(btn_map)
        
        # LiDAR Topic dropdown menu
        self.loc_combo = QComboBox()
        self.loc_combo.addItems(["topicA", "topicB", "topicC"])
        layout.addWidget(QLabel("LiDAR Topic:"))
        layout.addWidget(self.loc_combo)
        
        # Click Start Point button
        btn_start_pt = QPushButton("Click Start Point")
        layout.addWidget(btn_start_pt)
        
        # Back button
        btn_back = QPushButton("Back")
        btn_back.clicked.connect(lambda: self.button_stack.setCurrentIndex(0))
        layout.addWidget(btn_back)
        
        layout.addStretch()
        self.button_stack.addWidget(widget)

        btn_back.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)

    def handle_lidar_change(self):
        self.CELLmap_agent.lidar_topic = self.lidar_combo.currentText()
        self.statusBar().showMessage(f"LiDAR Topic changed to {self.lidar_combo.currentText()}")

    def select_directory(self, directory_type):
        """Select directory method"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")

        if directory_type == "pcds":
            self.CELLmap_agent.pcd_folder = directory
            self.lidar_combo.setEnabled(False)
            self.btn_rosbag.setEnabled(False)
        elif directory_type == "outfolder":
            self.CELLmap_agent.out_folder = directory
        self.statusBar().showMessage(f"{directory_type.capitalize()} selected: {directory}")
        if self.CELLmap_agent.traj_file and (self.CELLmap_agent.rosbag_file or self.CELLmap_agent.pcd_folder) and self.CELLmap_agent.out_folder:
            self.btn_start.setEnabled(True)


    def select_file(self, file_type):
        """Select file method"""
        filename, _ = QFileDialog.getOpenFileName(
            self, 
            "Select File", 
            "", 
            # "All Files (*);;Text Files (*.txt)",
            options=QFileDialog.Options()
        )
        if filename:
            if file_type == "trajectory":
                self.CELLmap_agent.traj_file = filename
            elif file_type == "rosbag":
                self.CELLmap_agent.rosbag_file = filename
                self.btn_pcds.setEnabled(False)

            self.statusBar().showMessage(f"{file_type.capitalize()} selected: {filename}")
        if self.CELLmap_agent.traj_file and (self.CELLmap_agent.rosbag_file or self.CELLmap_agent.pcd_folder) and self.CELLmap_agent.out_folder:
            self.btn_start.setEnabled(True)
        return filename

    def add_random_point(self):
        """Add random point"""
        pos = np.random.uniform(-5, 5, size=(3,))
        color = (0.5, 0.5, 1.0, 1.0)  # Default blue
        self.points.append({'pos': pos, 'color': color})
        self.update_scatter()

    def enter_line_mode(self):
        self.line_mode = True
        self.selected_points = []
        # Provide visual feedback
        self.statusBar().showMessage("Line Mode: Select two points", 3000)

    def left_click_to_idx(self, event):
        viewport = self.view.geometry()
        width = viewport.width()
        height = viewport.height()
        
        # Convert mouse coordinates to normalized device coordinates (NDC)
        mouse_x = (2.0 * (event.pos().x() - viewport.x()) / width) - 0.5
        mouse_y = 1.0 - (2.0 * (event.pos().y() - viewport.y()) / height)
            # Get matrix
        view_proj = self.view.projectionMatrix() * self.view.viewMatrix()
    
        points = [p['pos'] for p in self.points]
        if not points:
            return -1
        
        # Calculate screen projection coordinates of all points
        screen_dists = []
        for p in points:
            # Convert point to homogeneous coordinates
            homog_point = QVector4D(p[0], p[1], p[2], 1.0)
            
            # Apply view projection matrix
            clip_coords = view_proj.map(homog_point)
            
            # Perspective division
            ndc_coords = clip_coords / clip_coords.w()
            # Calculate 2D distance from mouse coordinates
            dx = ndc_coords.x() - mouse_x
            dy = ndc_coords.y() - mouse_y
            dist = dx*dx + dy*dy  # Use squared distance to avoid square root calculation
            
            screen_dists.append(dist)
        
        # Find index of nearest point
        idx = np.argmin(screen_dists)
        return idx

    def handle_mouse_press(self, event):
        super().mousePressEvent(event)
        self.mouse_press_pos = event.pos()  # Record press position
        self.mouse_pressed = True

    def handle_mouse_release(self, event):
        super().mouseReleaseEvent(event)
        if self.mouse_pressed:
            move_distance = (event.pos() - self.mouse_press_pos).manhattanLength()
            if move_distance > 5:  # Move more than 5 pixels considered drag
                self.mouse_pressed = False
                return
        if self.line_mode and event.button() == Qt.LeftButton:
            idx = self.left_click_to_idx(event)
            self.selected_points.append(idx)
            if len(self.selected_points) == 2:
                self.create_line(self.selected_points[0], self.selected_points[1])
        elif event.button() == Qt.LeftButton:
            idx = self.left_click_to_idx(event)
            if self.points[idx]['color'] == (1,0,0,1):
                self.points[idx]['color'] = (0.5, 0.5, 1.0, 1.0)
            elif self.points[idx]['color'] == (0.5, 0.5, 1.0, 1.0):
                self.points[idx]['color'] = (1,0,0,1)

            self.update_scatter()
        self.mouse_pressed = False

    def create_line(self, idx1, idx2):
        """Create line between two points"""
        p1 = self.points[idx1]['pos']
        p2 = self.points[idx2]['pos']
        line = gl.GLLinePlotItem(pos=np.array([p1, p2]), color=(1,1,1,1), width=2)
        self.view.addItem(line)
        self.lines.append(line)
        self.statusBar().showMessage(f"Created line between point {idx1} and {idx2}")
        self.line_mode = False  # Ensure exit line mode

    def update_scatter(self):
        pos = np.array([p['pos'] for p in self.points])
        colors = np.array([p['color'] for p in self.points])
        if len(pos) > 0:
            self.scatter.setData(pos=pos, color=colors, size=10) #, pxMode=False
    def draw_trajectory(self):
        while True:
            print(len(self.CELLmap_agent.raw_traj))
            self.points = []
            for i, pose in enumerate(self.CELLmap_agent.raw_traj):
                if i % 50 == 0:
                    pos = [pose[1], pose[2], pose[3]]
                    color = (0.5, 0.5, 1.0, 1.0)  
                    self.points.append({'pos': pos, 'color': color})
            self.update_scatter()
            time.sleep(0.1)
    def mapping_start(self):
        frontend_odom_path = os.path.join(self.CELLmap_agent.out_folder, "frontend_odom.txt")
        backend_outfile = os.path.join(self.CELLmap_agent.out_folder, "backend_odom.txt")
        print("roslaunch",
                "sphere_slam", 
                "pub_plugin_backend_kitti.launch", 
                "input_odom_path:="+self.CELLmap_agent.traj_file,
                "pcd_folder:="+self.CELLmap_agent.pcd_folder, 
                "frontend_odom_path:="+frontend_odom_path,
                "backend_odom_path:="+backend_outfile)
        launch_proc = subprocess.Popen(["roslaunch",
                "sphere_slam", 
                "pub_plugin_backend_kitti.launch", 
                "input_odom_path:="+self.CELLmap_agent.traj_file,
                "pcd_folder:="+self.CELLmap_agent.pcd_folder, 
                "frontend_odom_path:="+frontend_odom_path,
                "backend_odom_path:="+backend_outfile], shell=False)
    
        # Subscribe to topic /odom, message type Odometry, and specify callback function odom_callback
        rospy.Subscriber("/odom", Odometry, self.CELLmap_agent.odom_callback)
        
        # Enter ROS message loop, wait for messages
        rospy_thread = threading.Thread(target=rospy.spin)
        rospy_thread.start()

        draw_raw_traj_thread = threading.Thread(target=self.draw_trajectory)
        draw_raw_traj_thread.start()

if __name__ == '__main__':
    rospy.init_node('CELLmap_GUI', anonymous=True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())