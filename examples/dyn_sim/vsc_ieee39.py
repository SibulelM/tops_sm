import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import numpy as np
#import importlib
#importlib.reload(dps)
#import importlib

if __name__ == '__main__':

    # Load model
    import tops.ps_models.ieee39 as model_data
    #importlib.reload(model_data)
    model = model_data.load()

    model['vsc'] = {'VSC': [
        ['name',    'T_pll',    'T_i',  'bus',  'P_K_p',    'P_K_i',    'Q_K_p',    'Q_K_i',    'P_setp',   'Q_setp',   ],
        ['VSC1',    0.1,        1,      '4',   0.1,        0.1,        0.1,        0.1,        100,          0],
    ]}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.ode_fun(0, ps.x_0))))

    x0 = ps.x_0
    v0 = ps.v_0

    t_end = 30
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        if t > 1:
            ps.vsc['VSC'].set_input('P_setp', -105)

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t
        v = sol.v

        Igen_4_3  = -1*ps.y_bus_red_full[3,2]*(v[3] - v[2])
        Igen_4_5  = -1*ps.y_bus_red_full[3,4]*(v[3] - v[4])
        Igen_4_14 = -1*ps.y_bus_red_full[3,13]*(v[3] - v[13])
        s_4 = v[3]*np.conj(Igen_4_3+Igen_4_5+Igen_4_14)  #Compute VA power at Bus 4

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy()) # extract the speed of the generators
       
    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

H = ps.gen['GEN'].par['H'] # Inertia of the generators
COI = res['gen_speed']@H/np.sum(H)

print(' ')
print('P_4',np.real(s_4)*ps.s_n)
print('Q_4',np.imag(s_4)*ps.s_n)

print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

# Speed of all the generators
plt.figure(1)
plt.plot(res['t'], res['gen_speed'])
plt.xlabel('Time [s]')
plt.ylabel('Gen. speed [pu]')
plt.legend(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10'])
plt.title('Speed of the generators')

# Center of Inertia (COI) frequency
plt.figure(2)
plt.plot(res['t'], COI)
plt.xlabel('Time [s]')
plt.ylabel('COI freq [pu]')
plt.title('Center of Inertia (COI) frequency')
plt.show()