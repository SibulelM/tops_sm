#Creating a fault similar to the one created in the Digsilent report Fig 8
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import numpy as np

if __name__ == '__main__':

    # Load model
    import tops.ps_models.ieee39 as model_data
    model = model_data.load()

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    t_end = 30
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    sc_bus_idx = 3   #Corresponds with Bus 4 of the IEEE 39 bus system
    
    #Load Step
    Padd   = 100     #in MW

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Short circuit
        if t >= 1:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = Padd/ps.s_n #Equivalent to 100 MW at v = 1pu

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        # Constant power loads: update the modified admittance matrix
            
        Igen_4_3  = -1*ps.y_bus_red_full[3,2]*(v[3] - v[2])
        Igen_4_5  = -1*ps.y_bus_red_full[3,4]*(v[3] - v[4])
        Igen_4_14 = -1*ps.y_bus_red_full[3,13]*(v[3] - v[13])


        s_4 = v[3]*np.conj(Igen_4_3+Igen_4_5+Igen_4_14)  #Compute VA power at Bus 4

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    print('P_4',np.real(s_4)*ps.s_n)
    print('Q_4',np.imag(s_4)*ps.s_n)

    plt.figure()
    plt.plot(res['t'], res['gen_speed'])
    plt.xlabel('Time [s]')
    plt.legend(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10'])
    plt.ylabel('Gen. speed')
    plt.show()