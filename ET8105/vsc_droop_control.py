# -*- coding: utf-8 -*-
"""
Created on Sun Nov  26 17:45:43 2023

@author: sbraa
"""

import sys
# import numpy as np
# import sympy as sp
# import control.matlab as control
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import numpy as np
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
importlib.reload(dps)
import importlib

if __name__ == '__main__':

    # Load model
    import dynpssimpy.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()

    model['pll'] = {'PLL1':[
        ['name', 'T_filter', 'bus'],
        *[[f'PLL{i}', 0.1, bus[0]] for i, bus in enumerate(model['buses'][1:])],
    ]}

    model['vsc'] = {'VSC': [
        ['name',    'T_pll',    'T_i',  'bus',  'P_K_p',    'P_K_i',    'Q_K_p',    'Q_K_i',    'P_setp',   'Q_setp'],
        # *[[f'VSC{i}', 0.1, 1, bus[0], 0.1, 0.1, 0.1, 0.1, 0.1, 0] for i, bus in enumerate(model['buses'][1:])],
        ['VSC1',    0.1,        1,      'B6',   0.01,        1e-12,        0.1,        0.1,        0,          0],
    ]}

    # import dynpssimpy.user_models.user_lib as user_lib
    import examples.user_models.user_lib as user_lib

    # Power system model
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.init_dyn_sim()
    print(max(abs(ps.ode_fun(0, ps.x_0))))

    x0 = ps.x_0
    v0 = ps.v_0

    t_end = 50
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()
    Pcontrol = 0.0
    Qcontrol = 36.0
    ps.vsc['VSC'].set_input('P_setp', Pcontrol)
    # Define variables to be stored
    Igen=0.0+0.0j
    Strans=0.0+0.0j
    v7_stored = []
    P_e_stored = []
    Ptrans_stored = []
    frequency_stored = []
    tcount=0
    event_flag=True
    # Run simulation with droop control
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))
        # Load change on bus B9
        #if t > 10:
        # ps.y_bus_red_mod[8, 8] = 0.3

        # Line outage between bus B5 and B6 (line no. 1)
        if t > 10 and event_flag:
            event_flag = False
            ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][0], 'disconnect')

        # BESS control
        ps.vsc['VSC'].set_input('P_setp', Pcontrol)
        ps.vsc['VSC'].set_input('Q_setp', Qcontrol)

        # Load change on bus B9
        # if t > 5:
        #   ps.y_bus_red_mod[8, 8] = 0.1
        # else:
        #   ps.y_bus_red_mod[8, 8] = 0

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t
        v = sol.v

        dx = ps.ode_fun(0, ps.x_0)

        for mdl in ps.dyn_mdls:
            mdl.reset_outputs()

        # Compute power transfer B8-->B9
        Igen = ps.y_bus_red_full[7,8]*(v[8] -v[7])
        Strans=v[7]*np.conj(Igen)
        # Store result
        res['t'].append(sol.t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['VSC_power'].append(ps.vsc['VSC'].P(x, v).copy())
        res['VSC_Q'].append(ps.vsc['VSC'].Q(x, v).copy())
        v7_stored.append(np.abs(v[7]))
        Ptrans_stored.append(np.real(Strans))
        P_e_stored.append(ps.gen['GEN'].P_e(x, v).copy())
        SpeedVector= ps.gen['GEN'].speed(x, v)
        

        # frequency=50+50*0.25* (SpeedVector[0]+SpeedVector[1]+SpeedVector[2]+SpeedVector[3])
        #if t<5:
        #    frequency=50+50*0.25* (SpeedVector[0]+SpeedVector[1]+SpeedVector[2]+SpeedVector[3])
        #elif t>5:
        frequency=50+50*0.33* (SpeedVector[1]+SpeedVector[2]+SpeedVector[3])
        frequency_stored.append(frequency)

        # Proportional control of VSC P and Q
        Pcontrol=-1000*(frequency-50)
        Qcontrol= 750*(1 - abs(v[7]))
        # Qcontrol = 0
        #if t > 5:
        #    Pcontrol = 300
        tcount+=1
        
    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))
    print('Line outage', ps.lines['Line'].par['name'][0])
    # print('bus_ref_spec', ps.vsc['VSC'].bus_ref_spec)
    
    #Separating each generator speed
    genspeed1=[]
    genspeed2=[]
    genspeed3=[]
    genspeed4=[]
    p=0
    while p < tcount:
        genspeed1.append(res['gen_speed'][p][0])
        genspeed2.append(res['gen_speed'][p][1])
        genspeed3.append(res['gen_speed'][p][2])
        genspeed4.append(res['gen_speed'][p][3])
        p+=1
        

    plt.figure()
    plt.plot(res['t'], genspeed1, label='GEN1')
    plt.plot(res['t'], genspeed2, label='GEN2')
    plt.plot(res['t'], genspeed3, label='GEN3')
    plt.plot(res['t'], genspeed4, label='GEN4')
    plt.xlabel('Time [s]')
    plt.ylabel('Speed deviation [pu]')
    plt.legend()
    # plt.ticklabel_format(useOffset=False, style='plain')
    # plt.show()
    
    plt.figure()
    plt.plot(res['t'], np.array(frequency_stored))
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    fmin = np.argmax(frequency_stored)
    x_min = res['t'][fmin]
    y_min = frequency_stored[fmin]
    #plt.plot(x_min, y_min, marker='o')
    #plt.ticklabel_format(useOffset=False, style='plain')
    # plt.show()

    # Plot active power
    fig, ax = plt.subplots(2)
    fig.suptitle('Generator and VSC active power')
    ax[0].plot(res['t'], np.array(res['VSC_power']))
    ax[0].set_ylabel('VSC power (MW)')
    # ax[1].plot(res['t'], np.array(P_e_stored) / [900, 900, 900, 900])
    ax[1].plot(res['t'], np.array(P_e_stored))
    ax[1].set_ylabel('Gen. power (MW)')
    ax[1].set_xlabel('time (s)')

    plt.figure()
    plt.plot(res['t'], np.array(Ptrans_stored))
    plt.xlabel('Time [s]')
    plt.ylabel('Power transfer [MW]')
    plt.ticklabel_format(useOffset=False, style='plain')
    # plt.show()

    plt.figure()
    plt.plot(res['t'], np.array(v7_stored))
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage @ B8 [pu]')
    #plt.ticklabel_format(useOffset=False, style='plain')
    #plt.show()

    plt.figure()
    plt.plot(res['t'], res['VSC_Q'])
    plt.xlabel('Time [s]')
    plt.ylabel('Q_VSC [MVar]')
    #plt.ticklabel_format(useOffset=False, style='plain')
    plt.legend()

    plt.show()
    