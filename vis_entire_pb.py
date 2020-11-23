import os
import sys
import argparse
import tensorflow as tf
import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
from keras import backend as K
from tqdm import tqdm
import sys
import glob
from io import BytesIO
import numpy as np
from math import ceil
from PIL import Image, ImageDraw
import imageio

if './' not in sys.path:
    sys.path.append('./')

import pretrained_networks
import run_projector
import dream_projector
import uuid
import shutil
import cv2

# ===========================================================
# CONSTANTS
# ===========================================================

NETWORK_PKL = "gdrive:networks/stylegan2-ffhq-config-f.pkl"
N_ITERATIONS = 300
WORK_DIR = 'temp/'

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('pb', help='The path to the pb model file')
    parser.add_argument('--save-dir', '-d', help='The path to the storage folder', default='results')
    parser.add_argument('--layers', '-l', help='Path to the txt file containing the layers\'s names', default='relu_layer_info.txt')
    return parser.parse_args(args)

def dream_project(gan_net, pb_path, layer_name, neuron_index, prefix):
    proj = dream_projector.DreamProjector()
    proj.set_network(gan_net, pb_path, layer_name, neuron_index)
    run_projector.dream_project(proj, prefix+'/', N_ITERATIONS)

def save_video(work_dir, save_path):
    imgs = sorted(glob.glob(os.path.join(work_dir, '*.jpg')))
    with imageio.get_writer(save_path, mode='I') as writer:
        for filepath in imgs:
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            writer.append_data(img)

def vis_layers(gan_net, pb_path, layer_info, work_dir, save_dir):
    for layer_name, n_channels in tqdm(layer_info):
        sub_folder = os.path.join(save_dir, layer_name.split('/')[0])
        os.makedirs(sub_folder, exist_ok=True)
        for i in range(n_channels):
            # Define paths and skip if needed
            src_file = os.path.join(work_dir, 'step0300.jpg')
            dst_path = os.path.join(sub_folder, str(i)+'.jpg')
            movie_path = dst_path.replace('.jpg', '.mp4')
            if os.path.isfile(dst_path) and os.path.isfile(movie_path):
                print('Skipping {}-{}'.format(layer_name, i))
                continue

            # Run, save image, save video
            dream_project(gan_net, pb_path, layer_name, i, work_dir)
            shutil.copy(src_file, dst_path)
            save_video(work_dir, movie_path)

            # Delete all temp files
            files = glob.glob(os.path.join(work_dir, '*'))
            for f in files:
                os.remove(f)


def get_layer_info(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        layer_name, n_channels = line.split(' ')
        n_channels = int(n_channels)
        data.append([layer_name, n_channels])
    return data[::-1] #Start with logits

def create_dir(save_dir, subfolder=None):
    os.makedirs(save_dir, exist_ok=True)
    if subfolder is not None:
        save_dir = os.path.join(save_dir, subfolder)
        os.makedirs(save_dir, exist_ok=True)
    return save_dir

def set_sess():
    # Run this to save memory space
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

def run(pb_path, save_dir, layer_info_path):

    # Allocating minimal memory
    set_sess()

    # Create save_dir
    filename = os.path.split(pb_path)[1][:-3]
    save_dir = create_dir(save_dir, subfolder=filename)

    # Working dir
    temp_folder = str(uuid.uuid4())
    work_dir = os.path.join(WORK_DIR, temp_folder)
    os.makedirs(work_dir, exist_ok=True)

    # Read in layer info
    layer_info = get_layer_info(layer_info_path)

    # Get GAN network
    gan_net = pretrained_networks.load_networks(NETWORK_PKL)[2]

    # Run visualization
    vis_layers(gan_net, pb_path, layer_info, work_dir, save_dir)
    
    # Remove temp dir
    shutil.rmtree(work_dir)



def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    run(args.pb, args.save_dir, args.layers)

if __name__ == '__main__':
    main()