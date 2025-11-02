# ⚡ Load Flow Analysis System

A **modern, user-friendly Python application** for performing **electrical power system load flow analysis** using both **Newton–Raphson** and **Gauss–Seidel** methods. The system features a **beautiful PyQt5-based GUI** with real-time visualization and comprehensive result analysis.

![Load Flow Analysis GUI](https://github.com/user-attachments/assets/7d7f56df-3c28-4725-8441-bbcd55a98cab)

---

## ✨ Key Features

### 🔌 Analysis Methods
- **Newton–Raphson Method:** Fast convergence with a quadratic rate.
- **Gauss–Seidel Method:** Simple and robust iterative approach.
- **Supports All Bus Types:** Slack, PV, and PQ buses.

### 🎨 Modern GUI
- **Dark & Light Themes:** Clean and elegant interface with gradient headers.
- **Real-time Visualization:** Interactive charts and convergence graphs.
- **Responsive Layout:** Scales beautifully across different screen sizes.
- **Custom Widgets:** Animated controls, stat cards, and gradient buttons.

### 📊 Data Management
- **Bus Data Configuration:** Simple entry of voltage, power, and type.
- **Line Data Management:** Easy input of line parameters.
- **Input Validation:** Robust error checking and data consistency.
- **Import/Export:** Export data and results to Excel for documentation.

### 📈 Results & Visualization
- **Voltage Profile:** Color-coded bar charts showing bus voltages.
- **Convergence Plot:** Detailed visualization of iteration progress.
- **Power Distribution:** Active/reactive power flow plots.
- **Statistical Summary:** Key performance metrics and voltage statistics.
- **Y-Bus Matrix:** Complete admittance matrix display.

### 🔧 Technical Highlights
- **Sparse Matrix Support:** Optimized for large systems.
- **Adaptive Convergence Control:** Smart iteration and error handling.
- **Comprehensive Logging:** Step-by-step iteration details.
- **Robust Exception Handling:** Prevents crashes and ensures stability.

---

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Required Packages
```bash
pip install numpy pandas PyQt5 matplotlib scipy openpyxl
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/load-flow-analysis.git
cd load-flow-analysis
```

---

## 🎯 Usage

### Run the Application
```bash
python gui.py
```

### Basic Workflow
1. **Configure Bus Data:** Add buses and specify types (Slack, PV, PQ).
2. **Define Line Data:** Enter transmission line parameters.
3. **Select Method:** Choose Newton–Raphson or Gauss–Seidel.
4. **Run Analysis:** Execute and view load flow results.
5. **Visualize Results:** Inspect voltage, power flow, and convergence.
6. **Export Data:** Save results as Excel files for reports.

### Directory Structure
```
load-flow-analysis/
├── gui.py                 # Main GUI application
├── algorithms/
│   ├── newton_raphson.py  # Newton–Raphson method implementation
│   └── gauss_seidel.py    # Gauss–Seidel method implementation
├── requirements.txt       # Dependency list
└── README.md              # Documentation
```

---

## 🔬 Algorithms

### Newton–Raphson Method
- **Convergence:** Quadratic rate, efficient for large systems.
- **Implementation:** Sparse matrix optimization with adaptive step sizing.
- **Features:** Robust Jacobian handling and dynamic iteration control.

### Gauss–Seidel Method
- **Convergence:** Linear rate with acceleration factor support.
- **Simplicity:** Easy to understand and extend.
- **Features:** Q-limit management and flexible iteration limits.

---

## 📋 Input Data Format

### Bus Data
```python
[Bus#, Type, V_mag, V_ang, PG, QG, PL, QL]
```
- `Type`: 1 = Slack, 2 = PV, 3 = PQ  
- `V_mag`: Voltage magnitude (pu)  
- `V_ang`: Voltage angle (°)  
- `PG`, `QG`: Generation (pu)  
- `PL`, `QL`: Load (pu)

### Line Data
```python
[FromBus, ToBus, R, X, B]
```
- `R`: Resistance (pu)  
- `X`: Reactance (pu)  
- `B`: Susceptance (pu)

---

## 📊 Output Results

The program provides detailed analytical results including:
- Final bus voltages (magnitude & angle)
- Iteration convergence details
- Y-Bus admittance matrix
- Power flow between buses
- Voltage and power statistics
- Multiple real-time visualizations

---

## 🛠️ Development

### Contributing
1. **Fork** the repository.
2. **Create a branch:** `git checkout -b feature/amazing-feature`
3. **Commit changes:** `git commit -m 'Add amazing feature'`
4. **Push branch:** `git push origin feature/amazing-feature`
5. **Open a Pull Request** on GitHub.

### Testing
```bash
# Run the algorithm modules independently
python algorithms/newton_raphson.py
python algorithms/gauss_seidel.py
```

---

## 🧠 Credits
Developed by **Nishan Dananjaya** and contributors — Faculty of Engineering, South Eastern University of Sri Lanka.

> Empowering Power System Engineers with Modern Load Flow Tools ⚡

