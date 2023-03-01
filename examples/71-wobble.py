import os
import glob
import numpy as np
import pandas as pd

from dhdt.generic.unit_conversion import deg2compass
from dhdt.input.read_sentinel2 import \
    get_flight_bearing_from_gnss_s2, \
    get_flight_orientation_s2, \
    get_flight_path_s2, get_raw_str_s2
from dhdt.generic.mapping_tools import ecef2llh
from dhdt.generic.attitude_tools import ordered_merge, quat_2_euler

dat_dir = '/Users/Alten005/jitter'

fname = ['MTD_DS-7N.xml', 'MTD_DS-7C.xml', 'MTD_DS-7S.xml', 'MTD_DS-7A.xml',
         'MTD_DS-7I.xml']
full_path = os.path.join(dat_dir, 'DSxml', fname[0])

#get_raw_str_s2(os.path.join(dat_dir, 'DSxml'), fname=fname[0])

t_0,q_0 = get_flight_orientation_s2(os.path.join(dat_dir, 'DSxml'),
                                    fname=fname[0])
t_1,q_1 = get_flight_orientation_s2(os.path.join(dat_dir, 'DSxml'),
                                    fname=fname[1])
t_2,q_2 = get_flight_orientation_s2(os.path.join(dat_dir, 'DSxml'),
                                    fname=fname[2])
t_3,q_3 = get_flight_orientation_s2(os.path.join(dat_dir, 'DSxml'),
                                    fname=fname[3])
t_4,q_4 = get_flight_orientation_s2(os.path.join(dat_dir, 'DSxml'),
                                    fname=fname[4])
print('.')

T_0,X_0,_,_ = get_flight_path_s2(os.path.join(dat_dir, 'DSxml'),
                                    fname=fname[0])
L_0 = ecef2llh(X_0)
T_1,X_1,_,_ = get_flight_path_s2(os.path.join(dat_dir, 'DSxml'),
                                    fname=fname[1])
L_1 = ecef2llh(X_1)
T_2,X_2,_,_ = get_flight_path_s2(os.path.join(dat_dir, 'DSxml'),
                                    fname=fname[2])
L_2 = ecef2llh(X_2)
T_3,X_3,_,_ = get_flight_path_s2(os.path.join(dat_dir, 'DSxml'),
                                    fname=fname[3])
L_3 = ecef2llh(X_3)
T_4,X_4,_,_ = get_flight_path_s2(os.path.join(dat_dir, 'DSxml'),
                                    fname=fname[4])
L_4 = ecef2llh(X_4)

import matplotlib.pyplot as plt

T,L = ordered_merge(T_0,T_3, L_0,L_3)
t,q = ordered_merge(t_0,t_3, q_0,q_3)
roll,pitch,yaw = quat_2_euler(q[:,0], q[:,1], q[:,2],q[:,3])
yaw = deg2compass(yaw)

timestamp = (t - t[0]).astype('timedelta64[ms]').astype(float)
Timestamp = (T - t[0]).astype('timedelta64[ms]').astype(float)
#f_r = np.poly1d(np.polyfit(timestamp, roll, deg=5))
#f_y = np.poly1d(np.polyfit(timestamp, yaw, deg=3))
#f_p = np.poly1d(np.polyfit(timestamp, pitch, deg=3))

from scipy import interpolate
s=.01
t_s,c_s,k_s = interpolate.splrep(timestamp, roll, s=s)
s_r = interpolate.BSpline(t_s,c_s,k_s)(timestamp)
t_s,c_s,k_s = interpolate.splrep(timestamp, pitch, s=s)
s_p = interpolate.BSpline(t_s,c_s,k_s)(timestamp)
t_s,c_s,k_s = interpolate.splrep(timestamp, yaw, s=s)
s_y = interpolate.BSpline(t_s,c_s,k_s)(timestamp)
t_s,c_s,k_s = interpolate.splrep(Timestamp, L[:,0], s=s)
s_l = interpolate.BSpline(t_s,c_s,k_s)(timestamp)

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

plt.rcParams['figure.figsize'] = [4.,6.]
fig, (ax1,ax2,ax3) = plt.subplots(1, 3, sharex='col', sharey='row')
ax1.scatter(roll-s_r, t, s=1, marker='.')

axins1 = zoomed_inset_axes(ax1, zoom=6, loc=1)
axins1.scatter(roll-s_r, t, s=1, marker='.', origin="lower")

ax2.scatter(pitch-s_p, t, s=1, marker='.')
ax3.scatter(yaw-s_y, t, s=1, marker='.')
ax1.invert_yaxis(), ax2.invert_yaxis(), ax3.invert_yaxis()
ax2.set_xticks(ax1.get_xticks()), ax3.set_xticks(ax1.get_xticks())
ax1.set_title('roll'), ax2.set_title('pitch'), ax3.set_title('yaw')
#ax1.set_yticks(ax1.get_yticks())
#ax1.set_yticklabels([f\"{int(item):,}\".replace(',', ' ') for item in ax1.get_yticks()])
plt.show()





plt.figure(),
plt.scatter(t, roll-s_r, s=1, marker='.')
plt.scatter(t, pitch-s_p, s=1, marker='.')
plt.scatter(t, yaw-s_y, s=1, marker='.')
plt.legend({'roll','pitch','yaw'})

fig,ax = plt.subplots()
plt.scatter(s_l, roll-s_r, s=1, marker='.')
plt.scatter(s_l, pitch-s_p, s=1, marker='.')
plt.scatter(s_l, yaw-s_y, s=1, marker='.')
plt.legend({'roll','pitch','yaw'})
ax.invert_xaxis()

plt.figure(),
plt.plot(t, s_r)
plt.plot(t, s_p)
plt.plot(t, s_y)
plt.legend({'roll','pitch','yaw'})


plt.figure(),
plt.plot(T,L[:,0])


#plt.plot(roll)
plt.plot(f_r(timestamp))

plt.plot(pitch)
plt.plot(f_p(timestamp))

t_s = np.hstack((t_0,t_3))
Idx = np.argsort(t_s)
t_s[Idx]
IN = (np.diff(t_s[Idx])/np.timedelta64(1, 's'))!=0


row_mask = np.append([True],np.any(np.diff(t_s[Idx],axis=0),1))

df_0 = pd.DataFrame(pd.DataFrame({'time':t_0, 'q_0':q_0[:,0]}))
df_3 = pd.DataFrame(pd.DataFrame({'time':t_3, 'q_0':q_3[:,0]}))

df_fl = pd.merge_ordered(df_0, df_3)

t_fl = df_fl['time'].to_numpy()
q_fl = df_fl['q_0'].to_numpy()

result = pd.concat([df_0, df_3])
