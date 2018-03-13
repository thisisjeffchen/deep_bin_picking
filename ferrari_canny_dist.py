import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from env.scene import ScenePopulator, Scene

import matplotlib.ticker as ticker



import dexnet

DEX_NET_PATH = '../dex-net/'
DB_NAME = 'dexnet_2.hdf5'
GRIPPER_NAME = 'yumi_metal_spline'
GRIPPER_REL_PATH = 'data/grippers/'
GRASP_METRIC = 'ferrari_canny'

scene = Scene (show_gui = False)
scene_pop = ScenePopulator (scene)
total_grasps = 0
metrics_all = []
dn = dexnet.DexNet()
dn.open_database(DEX_NET_PATH + DB_NAME, create_db=True)
dn.open_dataset('3dnet')

for item_name in scene_pop.item_database:
	
    grasps, metrics = dn.get_grasps(item_name, GRIPPER_NAME, GRASP_METRIC)
    l = len(grasps)
    print "num_grasps: " + str(l) + " ---" + "max: " + str(metrics[0]) + " ---min: " + str(metrics[-1]) + " ---" + item_name
    total_grasps += l
    metrics_all += metrics


print "total grasps: " + str(total_grasps)
print "mean: " + str(np.mean (metrics_all))
print "median: " + str(np.median (metrics_all))
print "mode: " + str(stats.mode (metrics_all))
print "stdev: " + str(np.std (metrics_all))

ninety = int (total_grasps* 0.9)
metrics_all.sort()
print "90th percent fc score: " + str(metrics_all[ninety])

'''
SMALL_SIZE = 20
MEDIUM_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)

'''
plt.figure(figsize=(4,3))

plt.hist(metrics_all)

plt.xlabel ("Grasp Quality (Ferrari-Canny)")
plt.ylabel ("Frequency")

#x_space = .008
#y_space = 1000
#plt.xaxis.set_major_locator(ticker.MultipleLocator(x_space))
#plt.yaxis.set_major_locator(ticker.MultipleLocator(y_space))

#xmarks=[i for i in range(1,length+1,1)]

plt.xticks([0, .008, .016])
plt.yticks([0, 1000,2000,3000])

#plt.show ()

plt.savefig('grasp_dist.png', bbox_inches='tight')
