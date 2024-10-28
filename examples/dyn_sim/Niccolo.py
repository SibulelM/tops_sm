import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import numpy as np

# Load model
import tops.ps_models.ieee39 as model_data

model = model_data.load()

# Power system model
ps = dps.PowerSystemModel(model=model)
ps.init_dyn_sim() # initialize power flow on the network
print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

t_end = 30              # simulation time
t_event = 1             # time of the load step occurance
event_true = True       # boolean to activate
power_unbanlance = 1e2  # power unbalance in the generator bus [MW]
t_0 = time.time()
x_0 = ps.x_0.copy() # set the initial state as the one computes before

# Solver
sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

t = 0
res = defaultdict(list) # store the results

# event_load_bus_idx = ps.loads['Load'].bus_idx_red['terminal'][5] # bus index of the generator where the P step
event_load_bus_idx = 1
all_load_bus_idx = ps.loads['Load'].bus_idx_red['terminal'] # index of all the loads

s_const_old = (ps.loads['Load'].par['P'] + 1j * ps.loads['Load'].par['Q'])/ps.s_n # "old" apparent power
print(' ')
print('P4',np.real(s_const_old[event_load_bus_idx]*ps.s_n))
print('Q4',np.imag(s_const_old[event_load_bus_idx]*ps.s_n))
v_old = ps.v_0[all_load_bus_idx]

v_bus_mag = abs(ps.v_0)
# print(' ')
print('Bus voltage magnitudes (p.u) = ', v_bus_mag)
print(' ')
size = ps.v_0.shape
print(size)
# print(' ')
print('B = ', ps.lines['Line'].par['B'][4])

y_old = np.conj(s_const_old)/abs(v_old)**2 # admittance of the load
v = ps.v_0

#Extract admittance values of the lines



while t < t_end:
  sys.stdout.write("\r%d%%" % (t/(t_end)*100)) # print the percentage of the simulation completed

  # Implement the short circuit in the bus where the generator is connected
  if t > t_event and  event_true:
    s_const_old[event_load_bus_idx] += 0*power_unbanlance/ps.s_n
    event_true = False

  # Simulate next step
  result = sol.step() # integrate the system one step
  # Extract the information from the solution
  x = sol.y # state variables
  v = sol.v # complex node voltage
  t = sol.t

  # Constant power loads: update the modified admittance matrix
  Igen_4_3  = -1*ps.y_bus_red_full[3,2]*(v[3] - v[2])
  Igen_4_5  = -1*ps.y_bus_red_full[3,4]*(v[3] - v[4])
  Igen_4_14 = -1*ps.y_bus_red_full[3,13]*(v[3] - v[13])


  s_4 = v[3]*np.conj(Igen_4_3+Igen_4_5+Igen_4_14)  #Compute VA power at Bus 4
  v_load = v[all_load_bus_idx]
  y_new = np.conj(s_const_old)/abs(v_load)**2 # new admittance of the load
  ps.y_bus_red_mod[(all_load_bus_idx,) * 2] = y_new - y_old

  dx = ps.ode_fun(0, ps.x_0) # compute the derivative of the state variables (in case they are needed)

  # Store result
  res['t'].append(t)
  res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy()) # extract the speed of the generators
  res['P_4'].append(np.real(s_4)*ps.s_n) # computed active power of the generator 4
  res['P_4_setpoint'].append( np.real(s_const_old[event_load_bus_idx]*ps.s_n) ) # extract the apparent power of the generator 4
  res['p'].append(np.real(s_const_old)*ps.s_n)
  
print(' ')
print('New Bus voltage magnitudes (p.u) = ', abs(v))
print(' ')
print('S_increment_P',np.real(s_const_old[event_load_bus_idx])*ps.s_n)
print('S_increment_Q',np.imag(s_const_old[event_load_bus_idx])*ps.s_n)
print(' ')
#print('I4',I4)
print(' ')
print('P_4',np.real(s_4)*ps.s_n)
print('Q_4',np.imag(s_4)*ps.s_n)

H = ps.gen['GEN'].par['H'] # Inertia of the generators
COI = res['gen_speed']@H/np.sum(H)

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

# Power of the generator 4
plt.figure(4)
plt.plot(res['t'], res['P_4'])
plt.plot(res['t'], res['P_4_setpoint'])
plt.xlabel('Time [s]')
plt.ylabel('p4 [MW]')
plt.legend(['Computed power', 'Set point'])
plt.title('Power of the generator 4')

# Power of all the generators
plt.figure(5)
plt.plot(res['t'], res['p'])
plt.xlabel('Time [s]')
plt.ylabel('Gen. active power')
plt.legend(['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'L17', 'L18', 'L19' ])
plt.title('power of all the generators')


plt.show()