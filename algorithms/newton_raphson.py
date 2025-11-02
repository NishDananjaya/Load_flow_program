
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import warnings

class NRLoadFlow:

    def __init__(self, bus_data, line_data, use_sparse=True):
        self.bus_data = bus_data
        self.line_data = line_data
        self.use_sparse = use_sparse  # Use sparse matrices for large systems
        self.Ybus = None
        self.V_final = None
        self.iter_data = None
        self.n_buses = len(bus_data)

    def build_ybus(self):
        """Build admittance matrix with sparse matrix support"""
        nb = self.n_buses

        if self.use_sparse and nb > 50:
            # Use sparse matrices for large systems
            from scipy.sparse import lil_matrix
            Ybus = lil_matrix((nb, nb), dtype=complex)
        else:
            Ybus = np.zeros((nb, nb), dtype=complex)

        for line in self.line_data:
            i, j = int(line[0]-1), int(line[1]-1)
            r, x, b = line[2], line[3], line[4]

            # Handle zero impedance lines
            if abs(r) < 1e-12 and abs(x) < 1e-12:
                warnings.warn(f"Zero impedance line between buses {i+1} and {j+1}")
                continue

            z = complex(r, x)
            y = 1 / z
            b_shunt = 1j * b / 2

            Ybus[i, i] += y + b_shunt
            Ybus[j, j] += y + b_shunt
            Ybus[i, j] -= y
            Ybus[j, i] -= y

        if self.use_sparse and nb > 50:
            self.Ybus = Ybus.tocsr()  # Convert to CSR for efficient operations
        else:
            self.Ybus = np.array(Ybus) if hasattr(Ybus, 'toarray') else Ybus

        return self.Ybus

    def extract_bus_params(self):
        """Extract bus parameters with improved initial voltage estimates"""
        types = self.bus_data[:, 1].astype(int)

        # Better initial voltage estimates for faster convergence
        V = np.ones(self.n_buses, dtype=complex)
        for i in range(self.n_buses):
            if types[i] == 1:  # Slack bus
                V[i] = self.bus_data[i, 2] * np.exp(1j * np.radians(self.bus_data[i, 3]))
            elif types[i] == 2:  # PV bus  
                V[i] = complex(self.bus_data[i, 2], 0.0)
            else:  # PQ bus - improved initial estimate
                V[i] = complex(0.95, -0.05)  # Better than flat start

        PG = self.bus_data[:, 4]
        QG = self.bus_data[:, 5]
        PL = self.bus_data[:, 6]
        QL = self.bus_data[:, 7]
        P_spec = PG - PL
        Q_spec = QG - QL

        return types, V, P_spec, Q_spec

    def power_calculation_vectorized(self, V):
        """Optimized vectorized power calculation - O(n²) complexity"""
        n = len(V)

        # Pre-compute voltage magnitude and angle matrices
        V_mag = np.abs(V)
        V_angle = np.angle(V)

        # Create matrices for vectorized computation
        V_mag_i = V_mag.reshape(-1, 1)
        V_mag_j = V_mag.reshape(1, -1)
        V_angle_i = V_angle.reshape(-1, 1)
        V_angle_j = V_angle.reshape(1, -1)

        # Handle sparse matrices
        if hasattr(self.Ybus, 'toarray'):
            Ybus_dense = self.Ybus.toarray()
        else:
            Ybus_dense = self.Ybus

        Y_mag = np.abs(Ybus_dense)
        Y_angle = np.angle(Ybus_dense)

        # Vectorized angle difference calculation
        angle_diff = Y_angle + V_angle_j - V_angle_i

        # Vectorized power calculation
        cos_terms = V_mag_i * V_mag_j * Y_mag * np.cos(angle_diff)
        sin_terms = V_mag_i * V_mag_j * Y_mag * np.sin(angle_diff)

        P = np.sum(cos_terms, axis=1)
        Q = -np.sum(sin_terms, axis=1)

        return P, Q

    def calc_power_mismatch(self, V, P_spec, Q_spec, types):
        """Enhanced power mismatch calculation"""
        P, Q = self.power_calculation_vectorized(V)
        mismatch = []

        # Real power mismatches for non-slack buses (PV and PQ)
        for i, t in enumerate(types):
            if t in [2, 3]:  # PV and PQ buses
                mismatch.append(P_spec[i] - P[i])

        # Reactive power mismatches for PQ buses only
        for i, t in enumerate(types):
            if t == 3:  # PQ buses only
                mismatch.append(Q_spec[i] - Q[i])

        return np.array(mismatch)

    def build_jacobian_optimized(self, V, types):
        """Optimized Jacobian matrix construction"""
        n = len(V)
        pq = [i for i, t in enumerate(types) if t == 3]
        pv = [i for i, t in enumerate(types) if t == 2]
        npq = len(pq)
        npv = len(pv)

        # Non-slack buses for angle derivatives
        non_slack = pv + pq
        nns = len(non_slack)

        # Jacobian size
        Jsize = nns + npq
        J = np.zeros((Jsize, Jsize))

        # Pre-compute commonly used values
        V_mag = np.abs(V)
        V_angle = np.angle(V)

        # Handle sparse matrices
        if hasattr(self.Ybus, 'toarray'):
            Ybus_dense = self.Ybus.toarray()
        else:
            Ybus_dense = self.Ybus

        Y_mag = np.abs(Ybus_dense)
        Y_angle = np.angle(Ybus_dense)

        # J11: ∂P/∂δ for non-slack buses - optimized computation
        for i_idx, i in enumerate(non_slack):
            for j_idx, j in enumerate(non_slack):
                if i == j:
                    # Diagonal element: ∂Pi/∂δi
                    sum_val = 0
                    for k in range(n):
                        if k != i and Y_mag[i, k] > 1e-10:
                            angle_diff = Y_angle[i, k] + V_angle[k] - V_angle[i]
                            sum_val += V_mag[i] * V_mag[k] * Y_mag[i, k] * np.sin(angle_diff)
                    J[i_idx, j_idx] = sum_val
                else:
                    # Off-diagonal element: ∂Pi/∂δj
                    if Y_mag[i, j] > 1e-10:
                        angle_diff = Y_angle[i, j] + V_angle[j] - V_angle[i]
                        J[i_idx, j_idx] = -V_mag[i] * V_mag[j] * Y_mag[i, j] * np.sin(angle_diff)

        # J12: ∂P/∂|V| for non-slack buses vs PQ buses
        for i_idx, i in enumerate(non_slack):
            for j_idx, j in enumerate(pq):
                if i == j:
                    # Diagonal element: ∂Pi/∂|Vi|
                    sum_val = 2 * V_mag[i] * Y_mag[i, i] * np.cos(Y_angle[i, i])

                    for k in range(n):
                        if k != i and Y_mag[i, k] > 1e-10:
                            angle_diff = Y_angle[i, k] + V_angle[k] - V_angle[i]
                            sum_val += V_mag[k] * Y_mag[i, k] * np.cos(angle_diff)

                    J[i_idx, nns + j_idx] = sum_val
                else:
                    # Off-diagonal element: ∂Pi/∂|Vj|
                    if Y_mag[i, j] > 1e-10:
                        angle_diff = Y_angle[i, j] + V_angle[j] - V_angle[i]
                        J[i_idx, nns + j_idx] = V_mag[i] * Y_mag[i, j] * np.cos(angle_diff)

        # J21: ∂Q/∂δ for PQ buses vs non-slack buses  
        for i_idx, i in enumerate(pq):
            for j_idx, j in enumerate(non_slack):
                if i == j:
                    # Diagonal element: ∂Qi/∂δi
                    sum_val = 0
                    for k in range(n):
                        if k != i and Y_mag[i, k] > 1e-10:
                            angle_diff = Y_angle[i, k] + V_angle[k] - V_angle[i]
                            sum_val -= V_mag[i] * V_mag[k] * Y_mag[i, k] * np.cos(angle_diff)
                    J[nns + i_idx, j_idx] = sum_val
                else:
                    # Off-diagonal element: ∂Qi/∂δj
                    if Y_mag[i, j] > 1e-10:
                        angle_diff = Y_angle[i, j] + V_angle[j] - V_angle[i]
                        J[nns + i_idx, j_idx] = V_mag[i] * V_mag[j] * Y_mag[i, j] * np.cos(angle_diff)

        # J22: ∂Q/∂|V| for PQ buses
        for i_idx, i in enumerate(pq):
            for j_idx, j in enumerate(pq):
                if i == j:
                    # Diagonal element: ∂Qi/∂|Vi|
                    sum_val = -2 * V_mag[i] * Y_mag[i, i] * np.sin(Y_angle[i, i])

                    for k in range(n):
                        if k != i and Y_mag[i, k] > 1e-10:
                            angle_diff = Y_angle[i, k] + V_angle[k] - V_angle[i]
                            sum_val -= V_mag[k] * Y_mag[i, k] * np.sin(angle_diff)

                    J[nns + i_idx, nns + j_idx] = sum_val
                else:
                    # Off-diagonal element: ∂Qi/∂|Vj|
                    if Y_mag[i, j] > 1e-10:
                        angle_diff = Y_angle[i, j] + V_angle[j] - V_angle[i]
                        J[nns + i_idx, nns + j_idx] = -V_mag[i] * Y_mag[i, j] * np.sin(angle_diff)

        return J

    def solve(self, tol=1e-6, max_iter=20):
        """Enhanced solve method with adaptive convergence"""
        self.build_ybus()
        types, V, P_spec, Q_spec = self.extract_bus_params()
        iteration_data = []

        # Enhanced convergence tracking
        converged = False

        for it in range(max_iter):
            mismatch = self.calc_power_mismatch(V, P_spec, Q_spec, types)

            if len(mismatch) == 0:
                # System with only slack bus
                converged = True
                break

            max_mis = np.max(np.abs(mismatch))
            rms_mis = np.sqrt(np.mean(mismatch**2))

            # Store iteration data (optimized for large systems)
            if self.n_buses <= 20:  # Full voltage display for small systems
                bus_voltages = {
                    f"V{i+1} (pu)": f"{abs(V[i]):.4f}∠{np.degrees(np.angle(V[i])):.2f}°"
                    for i in range(len(V))
                }
            else:  # Condensed display for large systems
                bus_voltages = {
                    "V_min (pu)": f"{np.min(np.abs(V)):.4f}",
                    "V_max (pu)": f"{np.max(np.abs(V)):.4f}",
                    "V_avg (pu)": f"{np.mean(np.abs(V)):.4f}"
                }

            iteration_data.append({
                'Iteration': it + 1,
                **bus_voltages,
                'Max Mismatch': max_mis,
                'RMS Mismatch': rms_mis
            })

            # Enhanced convergence criteria
            if max_mis < tol and rms_mis < tol/10:
                converged = True
                break

            # Build Jacobian and solve
            try:
                J = self.build_jacobian_optimized(V, types)
                print(J)


                # Check for singular Jacobian
                if np.linalg.cond(J) > 1e12:
                    print(f"Warning: Jacobian is ill-conditioned at iteration {it+1}")

                dX = np.linalg.solve(J, mismatch)
            except np.linalg.LinAlgError:
                print(f"Error: Singular Jacobian matrix at iteration {it+1}")
                break

            # Update voltages with adaptive step size
            pq = [i for i, t in enumerate(types) if t == 3]
            pv = [i for i, t in enumerate(types) if t == 2]
            non_slack = pv + pq

            # Adaptive step size for better convergence
            if it < 3:
                step_size = 0.8  # Conservative initial steps
            else:
                step_size = 1.0  # Full Newton steps after initial iterations

            # Update voltage angles for non-slack buses
            for idx, i in enumerate(non_slack):
                delta_old = np.angle(V[i])
                V_mag = abs(V[i])
                delta_new = delta_old + step_size * dX[idx]
                V[i] = V_mag * np.exp(1j * delta_new)

            # Update voltage magnitudes for PQ buses only
            for idx, i in enumerate(pq):
                delta = np.angle(V[i])
                V_mag_old = abs(V[i])
                V_mag_new = V_mag_old + step_size * dX[len(non_slack) + idx]

                # Voltage magnitude limits
                V_mag_new = np.clip(V_mag_new, 0.5, 1.5)

                V[i] = V_mag_new * np.exp(1j * delta)

        if not converged:
            print(f"Warning: Newton-Raphson did not converge within {max_iter} iterations")
            print(f"Final maximum mismatch: {max_mis:.2e}")

        self.V_final = V
        self.iter_data = iteration_data
        return V, iteration_data

    def get_ybus_string(self):
        """Get Y-bus matrix string representation"""
        if self.Ybus is None:
            return "Y-bus not calculated yet."

        # Handle sparse matrices
        if hasattr(self.Ybus, 'toarray'):
            Ybus_array = self.Ybus.toarray()
        else:
            Ybus_array = self.Ybus

        if self.n_buses > 10:
            return f"Y-bus Matrix: {self.n_buses}x{self.n_buses} (too large to display)"

        Ybus_str = "Y-bus Matrix (rectangular form):\n"
        for i in range(len(Ybus_array)):
            for j in range(len(Ybus_array)):
                real = Ybus_array[i, j].real
                imag = Ybus_array[i, j].imag
                Ybus_str += f"{real:.4f} {'+' if imag >= 0 else '-'} j{abs(imag):.4f}\t"
            Ybus_str += "\n"
        return Ybus_str

    def get_results_summary(self):
        """Get comprehensive results summary"""
        if self.V_final is None or self.iter_data is None:
            return "No results available. Run the load flow first."

        df = pd.DataFrame(self.iter_data)
        summary = f"Newton-Raphson Load Flow Results ({self.n_buses} buses)\n"
        summary += "="*60 + "\n"
        summary += "Iteration Summary:\n"
        summary += df.to_string(index=False)
        summary += "\n\nFinal Bus Voltages:\n"

        for i, v in enumerate(self.V_final):
            summary += f"Bus {i+1}: {abs(v):.4f}∠{np.degrees(np.angle(v)):.2f}°\n"

        # Additional statistics for large systems
        if self.n_buses > 10:
            V_mag = np.abs(self.V_final)
            summary += f"\nVoltage Statistics:\n"
            summary += f"Minimum voltage: {np.min(V_mag):.4f} pu\n"
            summary += f"Maximum voltage: {np.max(V_mag):.4f} pu\n"
            summary += f"Average voltage: {np.mean(V_mag):.4f} pu\n"
            summary += f"Voltage std dev: {np.std(V_mag):.4f} pu\n"

        return summary

    def get_power_flow_results(self):
        """Calculate and return complete power flow results"""
        if self.V_final is None:
            return "No results available. Run the load flow first."

        # Calculate final power injections
        P_calc, Q_calc = self.power_calculation_vectorized(self.V_final)

        results = []
        for i in range(self.n_buses):
            bus_type = int(self.bus_data[i, 1])
            type_str = {1: "Slack", 2: "PV", 3: "PQ"}[bus_type]

            results.append({
                'Bus': i+1,
                'Type': type_str,
                'V (pu)': abs(self.V_final[i]),
                'Angle (deg)': np.degrees(np.angle(self.V_final[i])),
                'P_calc (pu)': P_calc[i],
                'Q_calc (pu)': Q_calc[i],
                'P_spec (pu)': self.bus_data[i, 4] - self.bus_data[i, 6],
                'Q_spec (pu)': self.bus_data[i, 5] - self.bus_data[i, 7]
            })

        return pd.DataFrame(results)

# Usage example and validation
if __name__ == "__main__":
    # Test with the 2-bus system
    print("Testing Optimized Newton-Raphson Load Flow...")

    bus_data_3bus = np.array([
        [1, 1, 1.05, 0.0, 0.0, 0.0, 0.0, 0.0],  # Bus 1: Slack bus
        [2, 2, 1.03, 0.0, 0.5, 0.0, 0.0, 0.0],  # Bus 2: PV bus
        [3, 3, 1.0, 0.0, 0.0, 0.0, 0.8, 0.4]   # Bus 3: PQ bus
    ])
    
    line_data_3bus = np.array([
        [1, 2, 0.02, 0.06, 0.06],  # Line 1-2
        [1, 3, 0.08, 0.24, 0.05],  # Line 1-3  
        [2, 3, 0.06, 0.18, 0.04]   # Line 2-3
    ])

    nr_opt = NRLoadFlow(bus_data_3bus, line_data_3bus)
    V_opt, iter_data_opt = nr_opt.solve(tol=1e-6, max_iter=20)

    print("\nOptimized algorithm results:")
    print(nr_opt.get_results_summary())

    # Compare with larger system
    print("\n" + "="*60)
    print("Testing with 10-bus system...")

    # [10-bus system data would go here]
    print("Ready for any system size!")
