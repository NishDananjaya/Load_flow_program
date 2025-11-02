import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,
                             QLabel, QComboBox, QTabWidget, QLineEdit, QDialog,
                             QFormLayout, QMessageBox, QFileDialog, QTextEdit,
                             QScrollArea, QFrame, QSplitter, QGridLayout, QHeaderView)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')

# Import your algorithms
from algorithms.newton_raphson import NRLoadFlow
from algorithms.gauss_seidel import GaussSeidel


class ModernButton(QPushButton):
    """Custom styled button with hover effects"""
    def __init__(self, text, color="#2196F3", parent=None):
        super().__init__(text, parent)
        self.color = color
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: {self.adjust_color(color, 20)};
            }}
            QPushButton:pressed {{
                background-color: {self.adjust_color(color, -20)};
            }}
            QPushButton:disabled {{
                background-color: #CCCCCC;
                color: #666666;
            }}
        """)
    
    def adjust_color(self, hex_color, amount):
        """Adjust color brightness"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = max(0, min(255, r + amount))
        g = max(0, min(255, g + amount))
        b = max(0, min(255, b + amount))
        return f'#{r:02x}{g:02x}{b:02x}'


class StatCard(QFrame):
    """Modern stat card widget"""
    def __init__(self, title, value, icon=None, color="#2196F3", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border-radius: 10px;
                border-left: 5px solid {color};
                padding: 15px;
            }}
        """)
        
        layout = QVBoxLayout()
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #666; font-size: 12px; font-weight: bold;")
        
        value_label = QLabel(str(value))
        value_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        self.setLayout(layout)
        self.value_label = value_label
    
    def update_value(self, value):
        self.value_label.setText(str(value))


class ChartWidget(QWidget):
    """Widget for displaying matplotlib charts"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6), facecolor='#f5f5f5')
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot_voltage_profile(self, V_final):
        """Plot voltage magnitude profile"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        bus_numbers = np.arange(1, len(V_final) + 1)
        voltages = np.abs(V_final)
        
        bars = ax.bar(bus_numbers, voltages, color='#2196F3', alpha=0.8, edgecolor='#1976D2', linewidth=2)
        
        # Color code based on voltage limits
        for i, bar in enumerate(bars):
            if voltages[i] < 0.95:
                bar.set_color('#F44336')  # Red for low voltage
            elif voltages[i] > 1.05:
                bar.set_color('#FF9800')  # Orange for high voltage
            else:
                bar.set_color('#4CAF50')  # Green for normal
        
        ax.axhline(y=1.0, color='#666', linestyle='--', linewidth=1, label='Nominal (1.0 pu)')
        ax.axhline(y=0.95, color='#F44336', linestyle=':', linewidth=1, alpha=0.5, label='Lower Limit')
        ax.axhline(y=1.05, color='#FF9800', linestyle=':', linewidth=1, alpha=0.5, label='Upper Limit')
        
        ax.set_xlabel('Bus Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Voltage Magnitude (pu)', fontsize=12, fontweight='bold')
        ax.set_title('Bus Voltage Profile', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')
        ax.set_ylim([min(0.9, min(voltages) - 0.05), max(1.1, max(voltages) + 0.05)])
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_voltage_angles(self, V_final):
        """Plot voltage angle profile"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        bus_numbers = np.arange(1, len(V_final) + 1)
        angles = np.degrees(np.angle(V_final))
        
        ax.plot(bus_numbers, angles, marker='o', linewidth=2, markersize=8, 
                color='#2196F3', markerfacecolor='#fff', markeredgewidth=2, 
                markeredgecolor='#2196F3')
        
        ax.set_xlabel('Bus Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Voltage Angle (degrees)', fontsize=12, fontweight='bold')
        ax.set_title('Bus Voltage Angles', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='#666', linestyle='--', linewidth=1)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_convergence(self, iter_data):
        """Plot convergence characteristics"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        iterations = [d['Iteration'] for d in iter_data]
        
        # Try different possible keys for mismatch data
        if 'Max Mismatch' in iter_data[0]:
            mismatches = [d['Max Mismatch'] for d in iter_data]
        elif 'Max Voltage Mismatch' in iter_data[0]:
            mismatches = [d['Max Voltage Mismatch'] for d in iter_data]
        else:
            mismatches = [0] * len(iterations)
        
        ax.semilogy(iterations, mismatches, marker='o', linewidth=2, markersize=8,
                   color='#FF5722', markerfacecolor='#fff', markeredgewidth=2,
                   markeredgecolor='#FF5722', label='Max Mismatch')
        
        ax.set_xlabel('Iteration Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Maximum Mismatch (log scale)', fontsize=12, fontweight='bold')
        ax.set_title('Convergence Characteristics', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        ax.legend(loc='best')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_pq_distribution(self, bus_data):
        """Plot P and Q distribution"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        bus_numbers = bus_data[:, 0]
        P_net = bus_data[:, 4] - bus_data[:, 6]  # PG - PL
        Q_net = bus_data[:, 5] - bus_data[:, 7]  # QG - QL
        
        x = np.arange(len(bus_numbers))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, P_net, width, label='Active Power (P)', 
                      color='#2196F3', alpha=0.8, edgecolor='#1976D2', linewidth=2)
        bars2 = ax.bar(x + width/2, Q_net, width, label='Reactive Power (Q)', 
                      color='#FF9800', alpha=0.8, edgecolor='#F57C00', linewidth=2)
        
        ax.set_xlabel('Bus Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Power (pu)', fontsize=12, fontweight='bold')
        ax.set_title('Active and Reactive Power Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([int(b) for b in bus_numbers])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.axhline(y=0, color='#666', linestyle='-', linewidth=1)
        
        self.figure.tight_layout()
        self.canvas.draw()


class BusDialog(QDialog):
    """Modern dialog for adding/editing bus data"""
    def __init__(self, parent=None, bus_data=None, edit_mode=False):
        super().__init__(parent)
        self.setWindowTitle("Edit Bus Data" if edit_mode else "Add Bus Data")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.edit_mode = edit_mode
        
        # Apply modern styling
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333;
                font-size: 13px;
            }
            QLineEdit, QComboBox {
                padding: 8px;
                border: 2px solid #ddd;
                border-radius: 5px;
                background-color: white;
                font-size: 13px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 2px solid #2196F3;
            }
        """)
        
        layout = QFormLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Bus number
        self.bus_num = QLineEdit()
        if bus_data:
            self.bus_num.setText(str(int(bus_data[0])))
            if edit_mode:
                self.bus_num.setReadOnly(True)
        layout.addRow("Bus Number:", self.bus_num)
        
        # Bus type
        self.bus_type = QComboBox()
        self.bus_type.addItems(["1 - Slack", "2 - PV", "3 - PQ"])
        if bus_data:
            self.bus_type.setCurrentIndex(int(bus_data[1]) - 1)
        layout.addRow("Bus Type:", self.bus_type)
        
        # Voltage magnitude
        self.v_mag = QLineEdit()
        self.v_mag.setText(str(bus_data[2]) if bus_data else "1.0")
        layout.addRow("Voltage Magnitude (pu):", self.v_mag)
        
        # Voltage angle
        self.v_ang = QLineEdit()
        self.v_ang.setText(str(bus_data[3]) if bus_data else "0.0")
        layout.addRow("Voltage Angle (deg):", self.v_ang)
        
        # Active generation
        self.p_gen = QLineEdit()
        self.p_gen.setText(str(bus_data[4]) if bus_data else "0.0")
        layout.addRow("Active Generation (pu):", self.p_gen)
        
        # Reactive generation
        self.q_gen = QLineEdit()
        self.q_gen.setText(str(bus_data[5]) if bus_data else "0.0")
        layout.addRow("Reactive Generation (pu):", self.q_gen)
        
        # Active load
        self.p_load = QLineEdit()
        self.p_load.setText(str(bus_data[6]) if bus_data else "0.0")
        layout.addRow("Active Load (pu):", self.p_load)
        
        # Reactive load
        self.q_load = QLineEdit()
        self.q_load.setText(str(bus_data[7]) if bus_data else "0.0")
        layout.addRow("Reactive Load (pu):", self.q_load)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.ok_button = ModernButton("OK", "#4CAF50")
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = ModernButton("Cancel", "#F44336")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addRow(button_layout)
        self.setLayout(layout)
    
    def get_data(self):
        """Get bus data from dialog"""
        try:
            return [
                int(self.bus_num.text()),
                int(self.bus_type.currentText()[0]),
                float(self.v_mag.text()),
                float(self.v_ang.text()),
                float(self.p_gen.text()),
                float(self.q_gen.text()),
                float(self.p_load.text()),
                float(self.q_load.text())
            ]
        except ValueError:
            return None


class LineDialog(QDialog):
    """Modern dialog for adding/editing line data"""
    def __init__(self, parent=None, line_data=None, bus_numbers=None, edit_mode=False):
        super().__init__(parent)
        self.setWindowTitle("Edit Line Data" if edit_mode else "Add Line Data")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.edit_mode = edit_mode
        
        # Apply modern styling
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333;
                font-size: 13px;
            }
            QLineEdit, QComboBox {
                padding: 8px;
                border: 2px solid #ddd;
                border-radius: 5px;
                background-color: white;
                font-size: 13px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 2px solid #2196F3;
            }
        """)
        
        layout = QFormLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # From bus
        self.from_bus = QComboBox()
        if bus_numbers:
            self.from_bus.addItems([str(int(b)) for b in bus_numbers])
        if line_data:
            self.from_bus.setCurrentText(str(int(line_data[0])))
            if edit_mode:
                self.from_bus.setEnabled(False)
        layout.addRow("From Bus:", self.from_bus)
        
        # To bus
        self.to_bus = QComboBox()
        if bus_numbers:
            self.to_bus.addItems([str(int(b)) for b in bus_numbers])
        if line_data:
            self.to_bus.setCurrentText(str(int(line_data[1])))
            if edit_mode:
                self.to_bus.setEnabled(False)
        layout.addRow("To Bus:", self.to_bus)
        
        # Resistance
        self.r = QLineEdit()
        self.r.setText(str(line_data[2]) if line_data else "0.01")
        layout.addRow("Resistance R (pu):", self.r)
        
        # Reactance
        self.x = QLineEdit()
        self.x.setText(str(line_data[3]) if line_data else "0.1")
        layout.addRow("Reactance X (pu):", self.x)
        
        # Susceptance
        self.b = QLineEdit()
        self.b.setText(str(line_data[4]) if line_data else "0.0")
        layout.addRow("Susceptance B (pu):", self.b)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.ok_button = ModernButton("OK", "#4CAF50")
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = ModernButton("Cancel", "#F44336")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addRow(button_layout)
        self.setLayout(layout)
    
    def get_data(self):
        """Get line data from dialog"""
        try:
            return [
                int(self.from_bus.currentText()),
                int(self.to_bus.currentText()),
                float(self.r.text()),
                float(self.x.text()),
                float(self.b.text())
            ]
        except ValueError:
            return None


class ModernLoadFlowGUI(QMainWindow):
    """Modern Load Flow Analysis GUI"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Load Flow Analysis - Modern Interface")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage
        self.bus_data = []
        self.line_data = []
        self.results = None
        
        # Apply modern theme
        self.apply_modern_theme()
        
        # Create UI
        self.init_ui()
    
    def apply_modern_theme(self):
        """Apply modern color scheme and styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: none;
                background-color: white;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                color: #333;
                padding: 15px 30px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #2196F3;
            }
            QTabBar::tab:hover {
                background-color: #d0d0d0;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                gridline-color: #e0e0e0;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::item:selected {
                background-color: #E3F2FD;
                color: #1976D2;
            }
            QHeaderView::section {
                background-color: #2196F3;
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
                font-size: 12px;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
            }
            QComboBox {
                padding: 8px;
                border: 2px solid #ddd;
                border-radius: 5px;
                background-color: white;
                min-width: 150px;
                font-size: 13px;
            }
            QComboBox:focus {
                border: 2px solid #2196F3;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QLabel {
                color: #333;
                font-size: 13px;
            }
        """)
    
    def init_ui(self):
        """Initialize the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Main content (tabs)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_bus_data_tab()
        self.create_line_data_tab()
        self.create_results_tab()
        self.create_charts_tab()
        
        main_layout.addWidget(self.tabs)
    
    def create_header(self):
        """Create modern header section"""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1976D2, stop:1 #2196F3);
                border-radius: 10px;
                padding: 20px;
            }
        """)
        
        layout = QHBoxLayout(header)
        
        # Title
        title = QLabel("⚡ Load Flow Analysis System")
        title.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Method selection
        method_label = QLabel("Analysis Method:")
        method_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        layout.addWidget(method_label)
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Newton-Raphson", "Gauss-Seidel"])
        self.method_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                color: #333;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
                min-width: 180px;
            }
        """)
        layout.addWidget(self.method_combo)
        
        # Run button
        self.run_button = ModernButton("▶ Run Analysis", "#4CAF50")
        self.run_button.clicked.connect(self.run_analysis)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        layout.addWidget(self.run_button)
        
        return header
    
    def create_dashboard_tab(self):
        """Create dashboard overview tab"""
        dashboard = QWidget()
        layout = QVBoxLayout(dashboard)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Stats cards
        stats_layout = QGridLayout()
        stats_layout.setSpacing(15)
        
        self.buses_card = StatCard("Total Buses", "0", color="#2196F3")
        self.lines_card = StatCard("Total Lines", "0", color="#4CAF50")
        self.slack_card = StatCard("Slack Bus", "Not Set", color="#FF9800")
        self.status_card = StatCard("Status", "Ready", color="#9C27B0")
        
        stats_layout.addWidget(self.buses_card, 0, 0)
        stats_layout.addWidget(self.lines_card, 0, 1)
        stats_layout.addWidget(self.slack_card, 0, 2)
        stats_layout.addWidget(self.status_card, 0, 3)
        
        layout.addLayout(stats_layout)
        
        # Quick info
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        
        info_title = QLabel("📊 System Overview")
        info_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333; margin-bottom: 15px;")
        info_layout.addWidget(info_title)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(300)
        self.info_text.setPlainText("Welcome to Load Flow Analysis System\n\n"
                                     "Getting Started:\n"
                                     "1. Add bus data in the 'Bus Data' tab\n"
                                     "2. Add line data in the 'Line Data' tab\n"
                                     "3. Select analysis method\n"
                                     "4. Click 'Run Analysis' to solve\n"
                                     "5. View results and charts\n\n"
                                     "Supported Methods:\n"
                                     "• Newton-Raphson (Fast convergence)\n"
                                     "• Gauss-Seidel (Simple and robust)")
        info_layout.addWidget(self.info_text)
        
        layout.addWidget(info_frame)
        layout.addStretch()
        
        self.tabs.addTab(dashboard, "📊 Dashboard")
    
    def create_bus_data_tab(self):
        """Create bus data input tab"""
        bus_widget = QWidget()
        layout = QVBoxLayout(bus_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("🔌 Bus Data Configuration")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        layout.addWidget(title)
        
        # Table
        self.bus_table = QTableWidget()
        self.bus_table.setColumnCount(8)
        self.bus_table.setHorizontalHeaderLabels([
            'Bus #', 'Type', 'V mag (pu)', 'V ang (°)', 
            'PG (pu)', 'QG (pu)', 'PL (pu)', 'QL (pu)'
        ])
        self.bus_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.bus_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.bus_table.setAlternatingRowColors(True)
        layout.addWidget(self.bus_table)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        add_btn = ModernButton("➕ Add Bus", "#4CAF50")
        add_btn.clicked.connect(self.add_bus)
        
        edit_btn = ModernButton("✏️ Edit Bus", "#2196F3")
        edit_btn.clicked.connect(self.edit_bus)
        
        delete_btn = ModernButton("🗑️ Delete Bus", "#F44336")
        delete_btn.clicked.connect(self.delete_bus)
        
        clear_btn = ModernButton("Clear All", "#FF9800")
        clear_btn.clicked.connect(self.clear_buses)
        
        button_layout.addWidget(add_btn)
        button_layout.addWidget(edit_btn)
        button_layout.addWidget(delete_btn)
        button_layout.addWidget(clear_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        self.tabs.addTab(bus_widget, "🔌 Bus Data")
    
    def create_line_data_tab(self):
        """Create line data input tab"""
        line_widget = QWidget()
        layout = QVBoxLayout(line_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("📡 Line Data Configuration")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        layout.addWidget(title)
        
        # Table
        self.line_table = QTableWidget()
        self.line_table.setColumnCount(5)
        self.line_table.setHorizontalHeaderLabels([
            'From Bus', 'To Bus', 'R (pu)', 'X (pu)', 'B (pu)'
        ])
        self.line_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.line_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.line_table.setAlternatingRowColors(True)
        layout.addWidget(self.line_table)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        add_btn = ModernButton("➕ Add Line", "#4CAF50")
        add_btn.clicked.connect(self.add_line)
        
        edit_btn = ModernButton("✏️ Edit Line", "#2196F3")
        edit_btn.clicked.connect(self.edit_line)
        
        delete_btn = ModernButton("🗑️ Delete Line", "#F44336")
        delete_btn.clicked.connect(self.delete_line)
        
        clear_btn = ModernButton("Clear All", "#FF9800")
        clear_btn.clicked.connect(self.clear_lines)
        
        button_layout.addWidget(add_btn)
        button_layout.addWidget(edit_btn)
        button_layout.addWidget(delete_btn)
        button_layout.addWidget(clear_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        self.tabs.addTab(line_widget, "📡 Line Data")
    
    def create_results_tab(self):
        """Create modern results display tab with cards and sections"""
        results_widget = QWidget()
        main_layout = QVBoxLayout(results_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Title
        title = QLabel("📋 Analysis Results")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        main_layout.addWidget(title)
        
        # Scroll area for results
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        # Container for all result cards
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        results_layout.setSpacing(15)
        results_layout.setContentsMargins(0, 0, 10, 0)
        
        # Summary Statistics Card
        self.summary_card = QFrame()
        self.summary_card.setMinimumHeight(120)
        self.summary_card.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 10px;
                padding: 20px;
            }
        """)
        summary_layout = QGridLayout(self.summary_card)
        
        self.summary_method = QLabel("Method: Not Run")
        self.summary_method.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        
        self.summary_iterations = QLabel("Iterations: -")
        self.summary_iterations.setStyleSheet("color: white; font-size: 14px;")
        
        self.summary_status = QLabel("Status: Ready")
        self.summary_status.setStyleSheet("color: white; font-size: 14px;")
        
        self.summary_time = QLabel("Buses: -")
        self.summary_time.setStyleSheet("color: white; font-size: 14px;")
        
        summary_layout.addWidget(self.summary_method, 0, 0, 1, 2)
        summary_layout.addWidget(self.summary_iterations, 1, 0)
        summary_layout.addWidget(self.summary_status, 1, 1)
        summary_layout.addWidget(self.summary_time, 2, 0)
        
        results_layout.addWidget(self.summary_card)
        
        # Voltage Statistics Card
        voltage_stats_frame = QFrame()
        voltage_stats_frame.setMinimumHeight(180)
        voltage_stats_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid #e0e0e0;
            }
        """)
        voltage_stats_layout = QVBoxLayout(voltage_stats_frame)
        voltage_stats_layout.setSpacing(15)

        voltage_title = QLabel("⚡ Voltage Statistics")
        voltage_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; margin-bottom: 10px;")
        voltage_stats_layout.addWidget(voltage_title)

        self.voltage_stats_grid = QGridLayout()
        self.voltage_stats_grid.setSpacing(15)
        self.voltage_stats_grid.setContentsMargins(0, 0, 0, 0)

        # Create stat cards for voltage info
        self.vmin_card = self.create_mini_stat_card("Min Voltage", "- pu", "#F44336")
        self.vmax_card = self.create_mini_stat_card("Max Voltage", "- pu", "#4CAF50")
        self.vavg_card = self.create_mini_stat_card("Avg Voltage", "- pu", "#2196F3")
        self.vstd_card = self.create_mini_stat_card("Std Dev", "- pu", "#FF9800")

        self.voltage_stats_grid.addWidget(self.vmin_card, 0, 0)
        self.voltage_stats_grid.addWidget(self.vmax_card, 0, 1)
        self.voltage_stats_grid.addWidget(self.vavg_card, 0, 2)
        self.voltage_stats_grid.addWidget(self.vstd_card, 0, 3)

        voltage_stats_layout.addLayout(self.voltage_stats_grid)
        voltage_stats_layout.addStretch()

        results_layout.addWidget(voltage_stats_frame)
        
        # Bus Voltages Table Card
        bus_voltages_frame = QFrame()
        bus_voltages_frame.setMinimumHeight(400)
        bus_voltages_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid #e0e0e0;
            }
        """)
        bus_voltages_layout = QVBoxLayout(bus_voltages_frame)
        bus_voltages_layout.setSpacing(10)

        bus_voltage_title = QLabel("🔌 Bus Voltages")
        bus_voltage_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; margin-bottom: 10px;")
        bus_voltages_layout.addWidget(bus_voltage_title)

        self.bus_voltage_table = QTableWidget()
        self.bus_voltage_table.setColumnCount(4)
        self.bus_voltage_table.setHorizontalHeaderLabels(['Bus', 'Voltage (pu)', 'Angle (°)', 'Status'])
        self.bus_voltage_table.horizontalHeader().setStretchLastSection(True)
        self.bus_voltage_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.bus_voltage_table.horizontalHeader().setMinimumHeight(35)
        self.bus_voltage_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.bus_voltage_table.verticalHeader().setVisible(False)
        self.bus_voltage_table.setAlternatingRowColors(True)
        self.bus_voltage_table.setMinimumHeight(300)
        self.bus_voltage_table.setStyleSheet("""
            QTableWidget {
                border: none;
                background-color: #fafafa;
            }
            QHeaderView::section {
                background-color: #2196F3;
                color: white;
                padding: 8px;
                border: none;
                font-weight: bold;
                font-size: 13px;
                min-height: 35px;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)
        bus_voltages_layout.addWidget(self.bus_voltage_table)

        results_layout.addWidget(bus_voltages_frame)
        
        # Iteration Details Card
        iteration_frame = QFrame()
        iteration_frame.setMinimumHeight(350)
        iteration_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid #e0e0e0;
            }
        """)
        iteration_layout = QVBoxLayout(iteration_frame)
        iteration_layout.setSpacing(10)

        iteration_title = QLabel("🔄 Iteration Details")
        iteration_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; margin-bottom: 10px;")
        iteration_layout.addWidget(iteration_title)

        self.iteration_table = QTableWidget()
        self.iteration_table.setMinimumHeight(250)
        self.iteration_table.horizontalHeader().setMinimumHeight(35)
        self.iteration_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.iteration_table.verticalHeader().setVisible(False)
        self.iteration_table.setAlternatingRowColors(True)
        self.iteration_table.setStyleSheet("""
            QTableWidget {
                border: none;
                background-color: #fafafa;
            }
            QHeaderView::section {
                background-color: #2196F3;
                color: white;
                padding: 12px;
                border: none;
                font-weight: bold;
                font-size: 13px;
                min-height: 40px;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)
        iteration_layout.addWidget(self.iteration_table)

        results_layout.addWidget(iteration_frame)
        
        # Y-Bus Matrix Card (Collapsible)
        ybus_frame = QFrame()
        ybus_frame.setMinimumHeight(280)
        ybus_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid #e0e0e0;
            }
        """)
        ybus_layout = QVBoxLayout(ybus_frame)
        ybus_layout.setSpacing(10)
        
        ybus_title = QLabel("📊 Y-Bus Matrix")
        ybus_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; margin-bottom: 10px;")
        ybus_layout.addWidget(ybus_title)
        
        self.ybus_text = QTextEdit()
        self.ybus_text.setReadOnly(True)
        self.ybus_text.setMinimumHeight(200)
        self.ybus_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        ybus_layout.addWidget(self.ybus_text)
        
        results_layout.addWidget(ybus_frame)
        
        # Don't add stretch here - let content determine height
        
        scroll.setWidget(results_container)
        main_layout.addWidget(scroll)
        
        # Action buttons at bottom
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        copy_btn = ModernButton("📋 Copy Results", "#2196F3")
        copy_btn.clicked.connect(self.copy_results)
        
        save_btn = ModernButton("💾 Save to File", "#4CAF50")
        # save_btn.clicked.connect(self.save_results)
        
        export_excel_btn = ModernButton("📊 Export Excel", "#FF9800")
        export_excel_btn.clicked.connect(self.export_to_excel)
        
        button_layout.addWidget(copy_btn)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(export_excel_btn)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
        
        self.tabs.addTab(results_widget, "📋 Results")

    def create_mini_stat_card(self, title, value, color):
        """Create a mini statistics card"""
        card = QFrame()
        card.setMinimumHeight(100)
        card.setMinimumWidth(150)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {color};
                border-radius: 8px;
                padding: 15px;
            }}
        """)
        
        layout = QVBoxLayout(card)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: white; font-size: 12px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        
        value_label = QLabel(value)
        value_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setObjectName("value_label")
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addStretch()
        
        return card
    
    def create_charts_tab(self):
        """Create visualization/charts tab"""
        charts_widget = QWidget()
        layout = QVBoxLayout(charts_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("📈 Visualizations & Charts")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        layout.addWidget(title)
        
        # Chart selection
        chart_select_layout = QHBoxLayout()
        chart_label = QLabel("Select Chart:")
        chart_label.setStyleSheet("font-weight: bold;")
        
        self.chart_combo = QComboBox()
        self.chart_combo.addItems([
            "Voltage Profile",
            "Voltage Angles", 
            "Convergence Plot",
            "Power Distribution"
        ])
        self.chart_combo.currentIndexChanged.connect(self.update_chart)
        
        chart_select_layout.addWidget(chart_label)
        chart_select_layout.addWidget(self.chart_combo)
        chart_select_layout.addStretch()
        
        layout.addLayout(chart_select_layout)
        
        # Chart widget
        self.chart_widget = ChartWidget()
        layout.addWidget(self.chart_widget)
        
        self.tabs.addTab(charts_widget, "📈 Charts")
    
    def update_dashboard(self):
        """Update dashboard statistics"""
        num_buses = len(self.bus_data)
        num_lines = len(self.line_data)
        
        self.buses_card.update_value(num_buses)
        self.lines_card.update_value(num_lines)
        
        # Find slack bus
        slack_bus = "Not Set"
        for bus in self.bus_data:
            if int(bus[1]) == 1:
                slack_bus = f"Bus {int(bus[0])}"
                break
        
        self.slack_card.update_value(slack_bus)
    
    def add_bus(self):
        """Add new bus"""
        dialog = BusDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
            if data:
                # Check if bus number already exists
                for bus in self.bus_data:
                    if int(bus[0]) == data[0]:
                        QMessageBox.warning(self, "Error", f"Bus {data[0]} already exists!")
                        return
                
                self.bus_data.append(data)
                self.refresh_bus_table()
                self.update_dashboard()
    
    def edit_bus(self):
        """Edit selected bus"""
        current_row = self.bus_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a bus to edit")
            return
        
        bus_data = self.bus_data[current_row]
        dialog = BusDialog(self, bus_data, edit_mode=True)
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
            if data:
                self.bus_data[current_row] = data
                self.refresh_bus_table()
                self.update_dashboard()
    
    def delete_bus(self):
        """Delete selected bus"""
        current_row = self.bus_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a bus to delete")
            return
        
        reply = QMessageBox.question(self, "Confirm", 
                                     "Are you sure you want to delete this bus?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            del self.bus_data[current_row]
            self.refresh_bus_table()
            self.update_dashboard()
    
    def clear_buses(self):
        """Clear all buses"""
        if not self.bus_data:
            return
        
        reply = QMessageBox.question(self, "Confirm",
                                     "Are you sure you want to clear all buses?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.bus_data.clear()
            self.refresh_bus_table()
            self.update_dashboard()
    
    def refresh_bus_table(self):
        """Refresh bus data table"""
        self.bus_table.setRowCount(len(self.bus_data))
        
        for i, bus in enumerate(self.bus_data):
            type_names = {1: "Slack", 2: "PV", 3: "PQ"}
            
            self.bus_table.setItem(i, 0, QTableWidgetItem(str(int(bus[0]))))
            self.bus_table.setItem(i, 1, QTableWidgetItem(f"{int(bus[1])} ({type_names.get(int(bus[1]), 'Unknown')})"))
            self.bus_table.setItem(i, 2, QTableWidgetItem(f"{bus[2]:.4f}"))
            self.bus_table.setItem(i, 3, QTableWidgetItem(f"{bus[3]:.2f}"))
            self.bus_table.setItem(i, 4, QTableWidgetItem(f"{bus[4]:.4f}"))
            self.bus_table.setItem(i, 5, QTableWidgetItem(f"{bus[5]:.4f}"))
            self.bus_table.setItem(i, 6, QTableWidgetItem(f"{bus[6]:.4f}"))
            self.bus_table.setItem(i, 7, QTableWidgetItem(f"{bus[7]:.4f}"))
            
            # Center align all items
            for j in range(8):
                self.bus_table.item(i, j).setTextAlignment(Qt.AlignCenter)
    
    def add_line(self):
        """Add new line"""
        if not self.bus_data:
            QMessageBox.warning(self, "Warning", "Please add buses first")
            return
        
        bus_numbers = [bus[0] for bus in self.bus_data]
        dialog = LineDialog(self, bus_numbers=bus_numbers)
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
            if data:
                # Check if line already exists
                for line in self.line_data:
                    if (int(line[0]) == data[0] and int(line[1]) == data[1]) or \
                       (int(line[0]) == data[1] and int(line[1]) == data[0]):
                        QMessageBox.warning(self, "Error", 
                                          f"Line between {data[0]} and {data[1]} already exists!")
                        return
                
                if data[0] == data[1]:
                    QMessageBox.warning(self, "Error", "From and To buses cannot be the same!")
                    return
                
                self.line_data.append(data)
                self.refresh_line_table()
                self.update_dashboard()
    
    def edit_line(self):
        """Edit selected line"""
        current_row = self.line_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a line to edit")
            return
        
        line_data = self.line_data[current_row]
        bus_numbers = [bus[0] for bus in self.bus_data]
        dialog = LineDialog(self, line_data, bus_numbers, edit_mode=True)
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
            if data:
                self.line_data[current_row] = data
                self.refresh_line_table()
    
    def delete_line(self):
        """Delete selected line"""
        current_row = self.line_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a line to delete")
            return
        
        reply = QMessageBox.question(self, "Confirm",
                                     "Are you sure you want to delete this line?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            del self.line_data[current_row]
            self.refresh_line_table()
            self.update_dashboard()
    
    def clear_lines(self):
        """Clear all lines"""
        if not self.line_data:
            return
        
        reply = QMessageBox.question(self, "Confirm",
                                     "Are you sure you want to clear all lines?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.line_data.clear()
            self.refresh_line_table()
            self.update_dashboard()
    
    def refresh_line_table(self):
        """Refresh line data table"""
        self.line_table.setRowCount(len(self.line_data))
        
        for i, line in enumerate(self.line_data):
            self.line_table.setItem(i, 0, QTableWidgetItem(str(int(line[0]))))
            self.line_table.setItem(i, 1, QTableWidgetItem(str(int(line[1]))))
            self.line_table.setItem(i, 2, QTableWidgetItem(f"{line[2]:.6f}"))
            self.line_table.setItem(i, 3, QTableWidgetItem(f"{line[3]:.6f}"))
            self.line_table.setItem(i, 4, QTableWidgetItem(f"{line[4]:.6f}"))
            
            # Center align all items
            for j in range(5):
                self.line_table.item(i, j).setTextAlignment(Qt.AlignCenter)
    
    def run_analysis(self):
        """Run load flow analysis"""
        # Validate input
        if not self.bus_data:
            QMessageBox.warning(self, "Error", "Please add at least one bus")
            return
        
        if not self.line_data:
            QMessageBox.warning(self, "Error", "Please add at least one line")
            return
        
        # Check for slack bus
        slack_count = sum(1 for bus in self.bus_data if int(bus[1]) == 1)
        if slack_count == 0:
            QMessageBox.warning(self, "Error", "Please specify exactly one slack bus (Type 1)")
            return
        elif slack_count > 1:
            QMessageBox.warning(self, "Error", "There can be only one slack bus (Type 1)")
            return
        
        # Prepare data
        bus_data_array = np.array(self.bus_data)
        line_data_array = np.array(self.line_data)
        
        try:
            # Select solver
            method = self.method_combo.currentText()
            
            if method == "Newton-Raphson":
                solver = NRLoadFlow(bus_data_array, line_data_array)
            else:  # Gauss-Seidel
                solver = GaussSeidel(bus_data_array, line_data_array)
            
            # Update status
            self.status_card.update_value("Running...")
            QApplication.processEvents()
            
            # Run solver
            V_final, iter_data = solver.solve()
            
            # Store results
            self.results = {
                'solver': solver,
                'V_final': V_final,
                'iter_data': iter_data,
                'method': method
            }
            
            # Display results
            self.display_results()
            
            # Update status
            self.status_card.update_value("✓ Converged")
            
            # Switch to results tab
            self.tabs.setCurrentIndex(3)
            
            QMessageBox.information(self, "Success", 
                                   f"{method} analysis completed successfully!\n"
                                   f"Converged in {len(iter_data)} iterations")
            
        except Exception as e:
            self.status_card.update_value("✗ Failed")
            QMessageBox.critical(self, "Error", 
                               f"An error occurred during analysis:\n{str(e)}")
    
    def display_results(self):
        """Display analysis results in modern format"""
        if not self.results:
            return
        
        solver = self.results['solver']
        V_final = self.results['V_final']
        iter_data = self.results['iter_data']
        method = self.results['method']
        
        # Update summary card
        self.summary_method.setText(f"Method: {method}")
        self.summary_iterations.setText(f"✓ Converged in {len(iter_data)} iterations")
        self.summary_status.setText("Status: ✓ Successful")
        self.summary_time.setText(f"Buses: {len(V_final)}")
        
        # Update voltage statistics
        V_mag = np.abs(V_final)
        V_min = np.min(V_mag)
        V_max = np.max(V_mag)
        V_avg = np.mean(V_mag)
        V_std = np.std(V_mag)
        
        # Find which card's value label to update
        self.vmin_card.findChild(QLabel, "value_label").setText(f"{V_min:.4f} pu")
        self.vmax_card.findChild(QLabel, "value_label").setText(f"{V_max:.4f} pu")
        self.vavg_card.findChild(QLabel, "value_label").setText(f"{V_avg:.4f} pu")
        self.vstd_card.findChild(QLabel, "value_label").setText(f"{V_std:.4f} pu")
        
        # Update bus voltages table
        self.bus_voltage_table.setRowCount(len(V_final))
        
        for i, v in enumerate(V_final):
            v_mag = abs(v)
            v_ang = np.degrees(np.angle(v))
            
            # Determine status
            if v_mag < 0.95:
                status = "⚠ Low"
                status_color = QColor(244, 67, 54)  # Red
            elif v_mag > 1.05:
                status = "⚠ High"
                status_color = QColor(255, 152, 0)  # Orange
            else:
                status = "✓ Normal"
                status_color = QColor(76, 175, 80)  # Green
            
            # Bus number
            bus_item = QTableWidgetItem(str(i + 1))
            bus_item.setTextAlignment(Qt.AlignCenter)
            self.bus_voltage_table.setItem(i, 0, bus_item)
            
            # Voltage magnitude
            vmag_item = QTableWidgetItem(f"{v_mag:.6f}")
            vmag_item.setTextAlignment(Qt.AlignCenter)
            self.bus_voltage_table.setItem(i, 1, vmag_item)
            
            # Voltage angle
            vang_item = QTableWidgetItem(f"{v_ang:.4f}")
            vang_item.setTextAlignment(Qt.AlignCenter)
            self.bus_voltage_table.setItem(i, 2, vang_item)
            
            # Status
            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setForeground(status_color)
            font = status_item.font()
            font.setBold(True)
            status_item.setFont(font)
            self.bus_voltage_table.setItem(i, 3, status_item)
        
        # Update iteration table
        df = pd.DataFrame(iter_data)
        self.iteration_table.setColumnCount(len(df.columns))
        self.iteration_table.setHorizontalHeaderLabels(df.columns.tolist())
        self.iteration_table.setRowCount(len(df))
        
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                value = df.iloc[i, j]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if abs(value) < 0.01 and value != 0:
                        item_text = f"{value:.2e}"
                    else:
                        item_text = f"{value:.6f}" if isinstance(value, float) else str(value)
                else:
                    item_text = str(value)
                
                item = QTableWidgetItem(item_text)
                item.setTextAlignment(Qt.AlignCenter)
                self.iteration_table.setItem(i, j, item)
        
        self.iteration_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        # Update Y-bus matrix
        self.ybus_text.setPlainText(solver.get_ybus_string())
        
        # Update charts
        self.update_chart()
    
    def update_chart(self):
        """Update selected chart"""
        if not self.results:
            return
        
        chart_type = self.chart_combo.currentText()
        V_final = self.results['V_final']
        iter_data = self.results['iter_data']
        
        if chart_type == "Voltage Profile":
            self.chart_widget.plot_voltage_profile(V_final)
        elif chart_type == "Voltage Angles":
            self.chart_widget.plot_voltage_angles(V_final)
        elif chart_type == "Convergence Plot":
            self.chart_widget.plot_convergence(iter_data)
        elif chart_type == "Power Distribution":
            bus_data_array = np.array(self.bus_data)
            self.chart_widget.plot_pq_distribution(bus_data_array)
    
    def copy_results(self):
        """Copy formatted results to clipboard"""
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to copy")
            return
        
        solver = self.results['solver']
        V_final = self.results['V_final']
        iter_data = self.results['iter_data']
        method = self.results['method']
        
        # Build formatted text
        text = f"{'='*80}\n"
        text += f"  {method} Load Flow Analysis Results\n"
        text += f"{'='*80}\n\n"
        text += f"Iterations: {len(iter_data)}\n"
        text += f"Buses: {len(V_final)}\n\n"
        
        text += "Bus Voltages:\n"
        for i, v in enumerate(V_final):
            text += f"Bus {i+1}: {abs(v):.6f} pu ∠ {np.degrees(np.angle(v)):.4f}°\n"
        
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "Success", "Results copied to clipboard!")

    def export_to_excel(self):
        """Export results to Excel file"""
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export to Excel", "", "Excel Files (*.xlsx);;All Files (*)"
        )
        
        if filename:
            try:
                V_final = self.results['V_final']
                iter_data = self.results['iter_data']
                
                # Create Excel writer
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    # Bus voltages sheet
                    bus_data = []
                    for i, v in enumerate(V_final):
                        bus_data.append({
                            'Bus': i + 1,
                            'Voltage (pu)': abs(v),
                            'Angle (degrees)': np.degrees(np.angle(v))
                        })
                    df_bus = pd.DataFrame(bus_data)
                    df_bus.to_excel(writer, sheet_name='Bus Voltages', index=False)
                    
                    # Iteration data sheet
                    df_iter = pd.DataFrame(iter_data)
                    df_iter.to_excel(writer, sheet_name='Iterations', index=False)
                
                QMessageBox.information(self, "Success", f"Results exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export:\n{str(e)}")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = ModernLoadFlowGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()