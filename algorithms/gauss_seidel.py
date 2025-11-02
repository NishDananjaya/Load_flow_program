import numpy as np
import pandas as pd

class GaussSeidel:
    def __init__(self, bus_data, line_data, Sbase=1):
        self.bus_data = bus_data
        self.line_data = line_data
        self.Sbase = Sbase
        self.Ybus = None
        self.V_final = None
        self.iter_data = []

    def build_ybus(self):
        nb = len(self.bus_data)
        Ybus = np.zeros((nb, nb), dtype=complex)
        for line in self.line_data:
            i, j = int(line[0]-1), int(line[1]-1)
            r, x, b = line[2], line[3], line[4]
            z = complex(r, x)
            y = 1 / z
            b_shunt = 1j * b / 2
            Ybus[i, i] += y + b_shunt
            Ybus[j, j] += y + b_shunt
            Ybus[i, j] -= y
            Ybus[j, i] -= y
        self.Ybus = Ybus
        return Ybus

    def solve(self, tol=1e-6, max_iter=100, alpha=1.0):
        try:
            self.build_ybus()
            types = self.bus_data[:, 1].astype(int)
            
            # Initialize voltages
            V = np.ones(len(types), dtype=complex)
            for i in range(len(V)):
                if types[i] == 1:  # Slack bus
                    V[i] = self.bus_data[i, 2] * np.exp(1j * np.radians(self.bus_data[i, 3]))
                else:
                    # CORRECTED: Better initial guess for faster convergence
                    V[i] = complex(0.95, -0.05)
            
            # Process power values
            PG = self.bus_data[:, 4] / self.Sbase
            QG = self.bus_data[:, 5] / self.Sbase
            PL = self.bus_data[:, 6] / self.Sbase
            QL = self.bus_data[:, 7] / self.Sbase
            
            # Handle Q limits
            if self.bus_data.shape[1] >= 10:
                Qmin = self.bus_data[:, 8] / self.Sbase
                Qmax = self.bus_data[:, 9] / self.Sbase
            else:
                Qmin = np.full(len(self.bus_data), -np.inf)
                Qmax = np.full(len(self.bus_data), np.inf)

            P_spec = PG - PL
            Q_spec = QG - QL
            nb = len(V)
            converged = False

            for it in range(max_iter):
                V_prev = V.copy()
                max_mis = 0.0
                max_power_mis = 0.0

                for i in range(nb):
                    if types[i] == 1:  # Skip slack bus
                        continue

                    # CORRECTED: Compute sum Y_ik * V_k for k != i using latest voltages
                    # This is the key difference in Gauss-Seidel vs Jacobi method
                    sumYV = 0.0
                    for k in range(nb):
                        if k != i:
                            sumYV += self.Ybus[i, k] * V[k]

                    if types[i] == 3:  # PQ bus
                        S = complex(P_spec[i], Q_spec[i])
                        # CORRECTED: Avoid division by zero with better check
                        if abs(V[i]) < 1e-12:
                            V[i] = complex(0.95, -0.05)
                        
                        # CORRECTED: Standard Gauss-Seidel update formula
                        V_new = (1 / self.Ybus[i, i]) * ((S.conjugate() / V[i].conjugate()) - sumYV)
                        
                        # Apply acceleration factor (alpha=1.0 for standard GS)
                        V[i] = alpha * V_new + (1.0 - alpha) * V[i]
                        
                    elif types[i] == 2:  # PV bus
                        # CORRECTED: Compute current injection using latest voltages
                        I_inj = np.dot(self.Ybus[i, :], V)
                        print(V)
                        print("I_inj:", I_inj)
                        # CORRECTED: Proper complex power calculation
                        S_calc = V[i] * I_inj.conjugate()
                        Qi = np.imag(S_calc)
                        Qi = np.clip(Qi, Qmin[i], Qmax[i])
                     
                        
                        S = complex(P_spec[i], Qi)
                        # CORRECTED: Avoid division by zero
                        if abs(V[i]) < 1e-12:
                            V[i] = complex(self.bus_data[i, 2], 0.0)
                        
                        V_new = (1 / self.Ybus[i, i]) * ((S.conjugate() / V[i].conjugate()) - sumYV)
                        
                        # CORRECTED: Maintain specified voltage magnitude
                        if abs(V_new) > 1e-12:
                            V_new = self.bus_data[i, 2] * (V_new / abs(V_new))
                        
                        # Apply acceleration factor
                        V[i] = alpha * V_new + (1.0 - alpha) * V[i]

                    mis = abs(V[i] - V_prev[i])
                    max_mis = max(max_mis, mis)

                    # Compute power mismatch for monitoring
                    I_inj = np.dot(self.Ybus[i, :], V)
                    # CORRECTED: Proper power calculation
                    S_calc = V[i] * I_inj.conjugate()
                    if types[i] == 3:
                        P_mis = abs(P_spec[i] - np.real(S_calc))
                        Q_mis = abs(Q_spec[i] - np.imag(S_calc))
                        max_power_mis = max(max_power_mis, P_mis, Q_mis)
                    elif types[i] == 2:
                        P_mis = abs(P_spec[i] - np.real(S_calc))
                        max_power_mis = max(max_power_mis, P_mis)

                self.iter_data.append({
                    'Iteration': it + 1,
                    'Max Voltage Mismatch': max_mis,
                    'Max Power Mismatch': max_power_mis,
                    **{f"V{i+1} (pu)": f"{abs(V[i]):.4f}∠{np.degrees(np.angle(V[i])):.2f}°" for i in range(nb)}
                })

                # CORRECTED: Use voltage mismatch as primary convergence criterion
                if max_mis < tol:
                    converged = True
                    break

            if not converged:
                print("Warning: Gauss-Seidel did not converge within maximum iterations.")

            self.V_final = V
            return V, self.iter_data

        except Exception as e:
            raise RuntimeError(f"An error occurred during Gauss-Seidel load flow calculation: {str(e)}")

    def get_ybus_string(self):
        if self.Ybus is None:
            return "Y-bus not calculated yet."

        Ybus_str = "Y-bus Matrix (rectangular form):\n"
        for i in range(len(self.Ybus)):
            for j in range(len(self.Ybus)):
                real = self.Ybus[i, j].real
                imag = self.Ybus[i, j].imag
                Ybus_str += f"{real:.4f} {'+' if imag >= 0 else '-'} j{abs(imag):.4f}\t"
            Ybus_str += "\n"
        return Ybus_str

    def get_results_summary(self):
        if self.V_final is None or not self.iter_data:
            return "No results available. Run the load flow first."

        df = pd.DataFrame(self.iter_data)
        summary = "Iteration Summary:\n"
        summary += df.to_string(index=False)
        summary += "\n\nFinal Bus Voltages:\n"
        for i, v in enumerate(self.V_final):
            summary += f"Bus {i+1}: {abs(v):.4f}∠{np.degrees(np.angle(v)):.2f}°\n"
        return summary

# Example usage and test cases
if __name__ == "__main__":
    
    # Test case 2: 3-bus system  
    print("Testing 3-bus system:")
    bus_data_3bus = np.array([
        [1, 1, 1.05, 0.0, 0.0, 0.0, 0.0, 0.0],  # Bus 1: Slack bus
        [2, 2, 1.03, 0.0, 0.5, 0.0, 0.0, 0.0],  # Bus 2: PV bus
        [3, 3, 1.0, 0.0, 0.0, 0.0, 0.8, 0.4]   # Bus 3: PQ bus
    ])
    
    line_data_3bus = np.array([
        [1, 2, 0.02, 0.06, 0.06],  # Line 1-2
        [1, 3, 0.08, 0.24, 0.5],  # Line 1-3  
        [2, 3, 0.06, 0.18, 0.04]   # Line 2-3
    ])
    
    gs_3bus = GaussSeidel(bus_data_3bus, line_data_3bus)
    V_3bus, iter_data_3bus = gs_3bus.solve(tol=1e-6, max_iter=50)
    print(gs_3bus.get_results_summary())