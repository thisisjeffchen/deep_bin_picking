#import dexnet
#import mayavi.mlab

import copy
import json
import IPython
import logging
import numpy as np
import os
import trimesh

try:
        import mayavi.mlab as mv
        import mayavi.mlab as mlab
except:
        logging.info('Failed to import mayavi')

indir= "/home/dealmaker/projects/dex-net/meshes"

for root, dirs, filenames in os.walk (indir):
    for f in filenames:
        print f
        mesh = trimesh.load (os.path.join(root, f))
        p = os.path.join(root, "urdf", f.split(".")[0], "")
        trimesh.io.export.export_urdf(mesh, str(p))        

print ("hello now")
