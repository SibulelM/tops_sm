from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
importlib.reload(dps)


if __name__ == '__main__':

    # Load model
    import dynpssimpy.ps_models.k2a_base_case_with_AVRs_and_GOVs as model_data
    model = model_data.load()

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    # ps.use_numba = True
    # Power flow calculation
    ps.power_flow()
    # Initialization
    ps.init_dyn_sim()
    #
    np.max(ps.ode_fun(0.0, ps.x0))
    # Specify simulation time
    #
    t_end = 10
    x0 = ps.x0.copy()
    # Add small perturbation to initial angle of first generator
    # x0[ps.gen_mdls['GEN'].state_idx['angle'][0]] += 1
    #
    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=5e-3)

    # Define other variables to plot
    P_e_stored = []
    E_f_stored = []
    Igen= 0,0
    I_stored = []
    v_bus = []
    # Initialize simulation
    t = 0
    result_dict = defaultdict(list)
    t_0 = time.time()
    # ps.build_y_bus_red(ps.buses['name'])
    ps.build_y_bus(['B8'])
    print('Ybus_full = ', ps.y_bus_red_full)
    print('Ybus_red = ', ps.y_bus_red)

    v_bus_mag = np.abs(ps.v_0)
    v_bus_angle = ps.v_0.imag / v_bus_mag
    #
    print(' ')
    print('Voltage magnitudes (p.u) = ', v_bus_mag)
    print(' ')
    print('Voltage angles     (rad) = ', v_bus_angle)
    print(' ')
    print('Voltage magnitudes  (kV) = ', v_bus_mag*[20, 20, 20, 20, 230, 230, 230, 230, 230, 230, 230])
    print(' ')
    # print(ps.v_n)
    print('v_vector = ', ps.v_0)
    print(' ')
    # print('Forskjell på red og full Ybus = ',ps.y_bus_red_full - ps.y_bus_red)
    #
    print('state description: ')
    print(ps.state_desc)
    print('Initial values on all state variables (G1 and IB) :')
    print(x0)
    print(' ')
    # Run simulation
    while t < t_end:
        # print(t)
        #v_bus_full = ps.red_to_full.dot(ps.v_red)
        # Simulate short circuit
        if 1 < t < 1.2:
          ps.y_bus_red_mod[4, 4] = 10000
        else:
           ps.y_bus_red_mod[4, 4] = 0
        # simulate a load change
        #if t > 2:
        #   ps.y_bus_red_mod[8, 8] = 0.1
        # else:
        #   ps.y_bus_red_mod[2, 2] = 0
        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t
        # Store result
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]
        # Igen = ps.y_bus_red_full[7,8]*(v[8] -v[7])
        Igen = v[8]
        # Legger til nye outputs
        P_e_stored.append(ps.gen['GEN'].P_e(x, v).copy())
        E_f_stored.append(ps.gen['GEN'].e_d_st(x, v).copy())
        I_stored.append(np.abs(Igen))

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))
    # # Linear analysis - not working
    #
    # ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    # ps_lin.linearize()
    # ps_lin.eigenvalue_decomposition()

    # Plot eigenvalues
    # dps_plt.plot_eigs(ps_lin.eigs)
    # plt.show()
    #
    # Convert dict to pandas dataframe
    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    # Plot angle and speed
    fig, ax = plt.subplots(3)
    fig.suptitle('Generator speed, angle and electric power')
    ax[0].plot(result[('Global', 't')], result.xs(key='speed', axis='columns', level=1))
    ax[0].set_ylabel('Speed (p.u.)')
    ax[1].plot(result[('Global', 't')], result.xs(key='angle', axis='columns', level=1))
    ax[1].set_ylabel('Angle (rad.)')
    ax[2].plot(result[('Global', 't')], np.array(P_e_stored)/[900, 900, 900, 900])
    ax[2].set_ylabel('Power (p.u.)')
    ax[2].set_xlabel('time (s)')

    plt.figure()
    plt.plot(result[('Global', 't')], np.array(E_f_stored))
    plt.xlabel('time (s)')
    plt.ylabel('E_d_st (p.u.)')

    plt.figure()
    plt.plot(result[('Global', 't')], np.array(I_stored))
    plt.xlabel('time (s)')
    plt.ylabel('V8 (magnitude p.u.)')

    plt.show()
