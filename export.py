"""Converts obj export models to urdf models."""

import logging
import os

import trimesh

logging.basicConfig(level=logging.INFO)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BULLET_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(ROOT_DIR))))
DEXNET_DIR = os.path.join(os.path.dirname(BULLET_DIR), 'dex-net')
IMPORT_DIR = os.path.join(DEXNET_DIR, 'meshes')
EXPORT_DIR = os.path.join(ROOT_DIR, 'meshes', 'urdf')

logging.info(DEXNET_DIR)
logging.info(EXPORT_DIR)
for (root, dirs, filenames) in os.walk(IMPORT_DIR):
    for f in filenames:
        logging.info('Loading {}'.format(f))
        mesh = trimesh.load(os.path.join(root, f))
        p = os.path.join(EXPORT_DIR, os.path.splitext(f)[0], '')
        logging.info('Exporting to {}'.format(p))
        os.makedirs(p)
        trimesh.io.export.export_urdf(mesh, p)
logging.info('Completed!')
