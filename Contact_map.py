import numpy as np
import numpy.linalg
import MDAnalysis as mda
from numpy import *
from MDAnalysis.analysis import contacts
import MDAnalysis.core.distances

from scipy.spatial import distance
#from MDAnalysis.tests.datafiles import PSF,DCD
# example trajectory (transition of AdK from closed to open)
top = '58ns.pdb'
traj = 'md_every1ns.trr'
u = mda.Universe(top,traj)

#load trajectory

timestep = 1
n_frames = len(u.trajectory)-1
start = 1

#select the regions to be measured against eachother from the trajecotry
protein_A = "(bynum  1:492)"
protein_B = "(bynum 493:1029)"
pA = u.select_atoms(protein_A)
pB = u.select_atoms(protein_B)
pA_res = np.unique(u.select_atoms(protein_A).resnums)
pB_res = np.unique(u.select_atoms(protein_B).resnums)
#determine dimensions of matrix
n1 = len(pA) #
n2 = len(pB)

#initialize array with zeros
contact_sum_atoms = numpy.zeros((n1, n2))

#define your max_distance (cutoff, example 5.0)
max_distance = 5.5

#contact matrix between atoms ####
#sum 1 or 0 of each matrix position across timesteps
for ts in u.trajectory[start::timestep]:
	ch1 = pA.positions
	ch2 = pB.positions
	ts_dist = distance.cdist(ch1, ch2, 'euclidean')
	ts_dist[ts_dist < max_distance] = 1
	ts_dist[ts_dist > max_distance] = 0
	contact_sum_atoms = ts_dist + contact_sum_atoms	
						
contact_ratio_atoms = contact_sum_atoms/n_frames


#==== contact matrix betwwen resid ======#
group_labels_x = pA.resids 
group_labels_y = pB.resids 
x_label_list = np.unique(group_labels_x).tolist()
y_label_list = np.unique(group_labels_y).tolist()
from itertools import groupby
repetitions_x = [len(list(group)) for key, group in groupby(group_labels_x)]
repetitions_y = [len(list(group)) for key, group in groupby(group_labels_y)]
resids_pA_range_ids = [0] + np.cumsum(repetitions_x).tolist()
resids_pB_range_ids = [0] + np.cumsum(repetitions_y).tolist()

x_dim = int((len(resids_pB_range_ids)-1)**2 )
y_dim = 1
res_i_j = np.zeros((x_dim, y_dim))
contact_mat_res_sum = numpy.zeros((len(resids_pA_range_ids)-1, len(resids_pB_range_ids)-1))

for ts in u.trajectory[start::timestep]:
    ch1 = pA.positions
    ch2 = pB.positions
    ts_dist = distance.cdist(ch1, ch2, 'euclidean')
    ts_dist[ts_dist < max_distance] = 1
    ts_dist[ts_dist > max_distance] = 0
    count = 0 
    for i in range(len(resids_pA_range_ids)-1):
        for j in range(len(resids_pB_range_ids)-1): 
            res_i_j[count, :] = np.mean(ts_dist[resids_pA_range_ids[i]:resids_pA_range_ids[i+1],\
                                      resids_pB_range_ids[j]:resids_pB_range_ids[j+1]])
    
            count+=1

    contacts_mat_res = res_i_j.reshape(len(resids_pA_range_ids) -1, len(resids_pB_range_ids)-1)
    contact_mat_res_sum =  contacts_mat_res + contact_mat_res_sum  
contact_ratio_resids = contact_mat_res_sum /n_frames


#---------------------------------------------------------------------------#
#basic plotting
#---------------------------------------------------------------------------#
from pylab import imshow, xlabel, ylabel, xlim, ylim, colorbar, cm, clf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
mpl.rc('text', usetex=True)


class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

#set x_min and y_min to the lowest residue index (example residue 50)
cr_shape = contact_ratio_atoms.shape
x_shift = cr_shape[1]
y_shift = cr_shape[0]
x_min = 1
y_min = 1
x_max = x_min + x_shift
y_max = y_min + y_shift

#had aspect= equal
fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.20)
im = plt.imshow(contact_ratio_atoms/np.max(contact_ratio_atoms),  vmin=0, vmax=1, aspect='equal', origin='lower', extent=[x_min,x_max, y_min, y_max] )

#im.set_cmap('hot')
im.set_cmap('YlGn')
#plt.grid(b=True, color='#737373')

im.set_interpolation('nearest')
plt.format_coord = Formatter(im)
delta = 1
plt.xlabel('Fcgr2b atom index', fontsize =20)
plt.ylabel('Dectin atom index', fontsize =20)
plt.xticks(np.arange(x_min, x_max, 49),rotation='vertical', fontsize =15)
plt.yticks(np.arange(y_min, y_max, 49),rotation='horizontal', fontsize =15)
cbar= fig.colorbar(im, ticks=[0,0.2,0.4, 0.6, 0.8, 1], )
cbar.ax.tick_params(labelsize=15)
plt.savefig('contact_map_atoms.png', dpi=300)





#set x_min and y_min to the lowest residue index (example residue 50)
cr_shape_res = contact_ratio_resids.shape
x_shift_res = cr_shape_res[1]
y_shift_res = cr_shape_res[0]
x_min_res = 1
y_min_res = 1
x_max_res = x_min_res + x_shift_res
y_max_res = y_min_res + y_shift_res

#had aspect= equal
fig2, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.20)
#plt.tight_layout()
im2 = plt.imshow(contact_ratio_resids/np.max(contact_ratio_resids), vmin=0, vmax=1, aspect='equal', origin='lower', extent=[x_min_res,x_max_res, y_min_res, y_max_res] )

#im.set_cmap('cool_r')
im2.set_cmap('YlGn')
plt.grid(b=True, color='#838383')

im.set_interpolation('nearest')
plt.format_coord = Formatter(im)
delta = 1
plt.xlabel('Fcgr2b', fontsize =20)
plt.ylabel('Dectin', fontsize = 20 )


list_AA_pA = (u.select_atoms('segid A and name CA').resnames).tolist()
list_AA_pB = (u.select_atoms('segid B and name CA').resnames).tolist()

plt.xticks(np.arange(2,35), list_AA_pB, rotation='vertical', fontsize = 7.5)
plt.yticks(np.arange(2,35), list_AA_pA, rotation='horizontal', fontsize= 7.5)
cbar= fig2.colorbar(im2, ticks=[0,0.2,0.4, 0.6, 0.8, 1])
cbar.ax.tick_params(labelsize=15)
plt.savefig('contact_map_resid.png', dpi=300)







