import numpy as np
import warnings

class FastDecoupled:
    def __init__(self, bus_data, line_data, Sbase=1):
        self.bus_data = bus_data
        self.line_data = line_data
        self.Sbase = Sbase
        self.Ybus = None
        self.V_final = None
        self.iter_data = []
        self.converged = False

#     def build_ybus(self):
#         nb = len(self.bus_data)
#         Ybus = np.zeros((nb, nb), dtype=complex)
#         for line in self.line_data:
#             i, j = int(line[0]-1), int(line[1]-1)
#             r, x, b = line[2], line[3], line[4]
#             z = complex(r, x)
#             y = 1 / z
#             b_shunt = 1j * b / 2
#             Ybus[i, i] += y + b_shunt
#             Ybus[j, j] += y + b_shunt
#             Ybus[i, j] -= y
#             Ybus[j, i] -= y
#         self.Ybus = Ybus
#         return Ybus

#     def _extract_bus_params(self):
#         types = self.bus_data[:, 1].astype(int)
#         V = self.bus_data[:, 2].astype(complex)
#         V_angle = np.radians(self.bus_data[:, 3])
#         V = V * np.exp(1j * V_angle)
        
#         # Convert to per unit
#         PG = self.bus_data[:, 4] / self.Sbase
#         QG = self.bus_data[:, 5] / self.Sbase
#         PL = self.bus_data[:, 6] / self.Sbase
#         QL = self.bus_data[:, 7] / self.Sbase
        
#         P_spec = PG - PL
#         Q_spec = QG - QL
        
#         # Handle Q limits
#         if self.bus_data.shape[1] >= 10:
#             Qmin = self.bus_data[:, 8] / self.Sbase
#             Qmax = self.bus_data[:, 9] / self.Sbase
#         else:
#             Qmin = np.full(len(self.bus_data), -np.inf)
#             Qmax = np.full(len(self.bus_data), np.inf)
            
#         return types, V, P_spec, Q_spec, Qmin, Qmax

#     def solve(self, tol=1e-6, max_iter=10):
#         self.build_ybus()
#         types, V, P_spec, Q_spec, Qmin, Qmax = self._extract_bus_params()
        
#         # Identify bus types
#         slack = np.where(types == 1)[0]
#         pv = np.where(types == 2)[0]
#         pq = np.where(types == 3)[0]
        
#         # Build B' and B'' matrices
#         B_prime = -np.imag(self.Ybus[np.ix_(np.concatenate((pv, pq)), np.concatenate((pv, pq)))])
#         B_dprime = -np.imag(self.Ybus[np.ix_(pq, pq)])
        
#         # Add shunt elements to diagonal
#         np.fill_diagonal(B_prime, -np.sum(np.imag(self.Ybus[np.ix_(np.concatenate((pv, pq)), np.concatenate((pv, pq)))]), axis=1))
#         np.fill_diagonal(B_dprime, -np.sum(np.imag(self.Ybus[np.ix_(pq, pq)]), axis=1))
        
#         # Factor matrices
#         B_prime_inv = np.linalg.inv(B_prime)
#         B_dprime_inv = np.linalg.inv(B_dprime)
        
#         for it in range(max_iter):
#             try:
#                 # Calculate power mismatches
#                 with warnings.catch_warnings():
#                     warnings.simplefilter("ignore")
#                     S = V * np.conj(self.Ybus @ V)
                
#                 P = S.real
#                 Q = S.imag
                
#                 # P-θ iteration
#                 delta_P = P_spec[np.concatenate((pv, pq))] - P[np.concatenate((pv, pq))]
#                 delta_P /= np.abs(V[np.concatenate((pv, pq))])
#                 delta_theta = B_prime_inv @ delta_P
                
#                 # Update angles
#                 V[np.concatenate((pv, pq))] *= np.exp(1j * delta_theta)
                
#                 # Q-V iteration
#                 delta_Q = Q_spec[pq] - Q[pq]
#                 delta_Q /= np.abs(V[pq])
#                 delta_V = B_dprime_inv @ delta_Q
                
#                 # Update voltages
#                 new_V_mag = np.abs(V[pq]) + delta_V
#                 V[pq] = new_V_mag * np.exp(1j * np.angle(V[pq]))
                
#                 # Check Q limits for PV buses
#                 for bus in pv:
#                     Qcalc = -np.imag(V[bus].conj() * (self.Ybus[bus, :] @ V))
#                     if Qcalc < Qmin[bus] or Qcalc > Qmax[bus]:
#                         # Convert to PQ bus
#                         types[bus] = 3
#                         pq = np.append(pq, bus)
#                         pv = np.delete(pv, np.where(pv == bus))
                        
#                         # Rebuild B' and B''
#                         B_prime = -np.imag(self.Ybus[np.ix_(np.concatenate((pv, pq)), np.concatenate((pv, pq)))])
#                         np.fill_diagonal(B_prime, -np.sum(np.imag(self.Ybus[np.ix_(np.concatenate((pv, pq)), np.concatenate((pv, pq)))]), axis=1))
#                         B_prime_inv = np.linalg.inv(B_prime)
                
#                 # Calculate maximum mismatch
#                 max_mis = max(np.max(np.abs(delta_P)), np.max(np.abs(delta_Q)))
                
#                 # Store iteration data
#                 self.iter_data.append({
#                     'Iteration': it + 1,
#                     'Max Mismatch': max_mis,
#                     **{f"V{i+1}": f"{abs(V[i]):.4f}∠{np.degrees(np.angle(V[i])):.2f}°" for i in range(len(V))}
#                 })
                
#                 if max_mis < tol:
#                     self.converged = True
#                     break
                    
#             except Exception as e:
#                 raise RuntimeError(f"Iteration {it+1} failed: {str(e)}")
        
#         self.V_final = V
#         return V, self.iter_data  # Now returns only 2 values

#     def get_results_summary(self):
#         if not hasattr(self, 'V_final') or self.V_final is None:
#             return "No results available. Run the load flow first."
        
#         summary = f"Converged: {self.converged}\n"
#         summary += "Iteration History:\n"
#         for iter_info in self.iter_data:
#             summary += f"Iter {iter_info['Iteration']}: Max Mismatch = {iter_info['Max Mismatch']:.6f}\n"
        
#         summary += "\nFinal Voltages:\n"
#         for i, v in enumerate(self.V_final):
#             summary += f"Bus {i+1}: {abs(v):.4f} pu ∠ {np.degrees(np.angle(v)):.2f}°\n"
        
#         return summary


# if __name__ == "__main__":
#     bus_data = np.array([
#         [1, 1, 1.05, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [2, 2, 1.03, 0.0, 0.5, 0.0, 0.0, 0.0],
#         [3, 3, 1.0, 0.0, 0.0, 0.0, 0.8, 0.4]
#     ])
#     line_data = np.array([
#         [1, 2, 0.02, 0.06, 0.06],
#         [1, 3, 0.08, 0.24, 0.05],
#         [2, 3, 0.06, 0.18, 0.04]
#     ])

#     gs = FastDecoupled(bus_data, line_data)
#     V, iter_data = gs.solve(tol=1e-6, max_iter=100)
#     print(gs.get_results_summary())