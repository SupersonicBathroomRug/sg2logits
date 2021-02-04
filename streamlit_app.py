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
import streamlit as st

if './' not in sys.path:
    sys.path.append('./')

import pretrained_networks
import run_projector
import dream_projector
import uuid
import shutil
import cv2
import vis_entire_pb

TL_GAN_HASH_FUNCS = {
    tf.Session : id
}

# ===========================================================
# CONSTANTS
# ===========================================================

NETWORK_PKL = "gdrive:networks/stylegan2-ffhq-config-f.pkl"
N_ITERATIONS = 2
WORK_DIR = 'temp/'
pb_path='./adam.pb' #this probably doesn't work in this form, possibly needs something along the lines of os.get_current_directory_path()
STEPS=100
gen_vid=False
feature_names=["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald",        "Bangs","Big_Lips","Big_Nose","Black_Hair","Blond_Hair",
                "Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin",                    "Eyeglasses","Goatee","Gray_Hair", "Heavy_Makeup","High_Cheekbones",
                "Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard",                 "Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks",
                "Sideburns","Smiling","Straight_Hair","Wavy_Hair","Wearing_Earrings",             "Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"]

TL_GAN_HASH_FUNCS = {
    tf.Session : id
}

def main():
    save_dir,work_dir,gan_net=setup()

    #======================================================================================================
    st.sidebar.title('Features')
    seed = 42
    # If the user doesn't want to select which features to control, these will be used.
    default_control_features = ['Young','Smiling','Male']

    if st.sidebar.checkbox('Show advanced options'):
        # Randomly initialize feature values. 
        features = get_random_features(feature_names, seed)
        
        # Let the user pick which features to control with sliders.
        control_features = st.sidebar.multiselect( 'Control which features?',
            sorted(features), default_control_features)
    else:
        features = get_random_features(feature_names, seed)
        # Don't let the user pick feature values to control.
        control_features = default_control_features
    
    # Insert user-controlled values from sliders into the feature vector.
    for feature in control_features:
        features[feature] = st.sidebar.slider(feature, -1.0, 1.0, 0, 0.1)


    st.sidebar.title('Note')
    st.sidebar.write(
        """Playing with the sliders, you _will_ find **biases** that exist in this model.
        """
    )
    st.sidebar.write(
        """For example, moving the `Smiling` slider can turn a face from masculine to feminine or from lighter skin to darker. 
        """
    )
    st.sidebar.write(
        """Apps like these that allow you to visually inspect model inputs help you find these biases so you can address them in your model _before_ it's put into production.
        """
    )

    active_goals=tf.convert_to_tensor([1 if name in control_features else 0 for name in feature_names],dtype=tf.float32)
    logit_goals=tf.convert_to_tensor([features[name] for name in feature_names],dtype=tf.float32)
    logit_weights=tf.constant([[[[1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1]]]],dtype=tf.float32)


    # Generate a new image from this feature vector
    vis_wrapper(gan_net, pb_path, work_dir, save_dir,active_goals,logit_goals,logit_weights,gen_vid,STEPS)
    image_out=mpimg.imread('./pictures/adam/Logits/op.jpg') #this may or may not be usable in this format
    st.image(image_out, use_column_width=True)


@st.cache(allow_output_mutation=True, hash_funcs=TL_GAN_HASH_FUNCS)
def setup():
    vis_entire_pb.set_sess()

    # Create save_dir
    filename = os.path.split(pb_path)[1][:-3]
    save_dir = vis_entire_pb.create_dir("pictures", subfolder=filename)

    # Working dir
    temp_folder = str(uuid.uuid4())
    work_dir = os.path.join(WORK_DIR, temp_folder)
    os.makedirs(work_dir, exist_ok=True)

    # Read in layer info

    # Get GAN network
    gan_net = pretrained_networks.load_networks(NETWORK_PKL)[2]
    return save_dir,work_dir,gan_net

@st.cache(show_spinner=False, hash_funcs=TL_GAN_HASH_FUNCS)
def vis_wrapper(gan_net, pb_path, work_dir, save_dir,active_goals,logit_goals,logit_weights,gen_vid,STEPS):
    vis_entire_pb.alt_vis_layers(gan_net, pb_path, work_dir, save_dir,active_goals,logit_goals,logit_weights,gen_vid,STEPS)

def get_random_features(feature_names,seed):
    """
    Return a random dictionary from feature names to feature
    values within the range [-1.0,1.0] (out of [-1.0,1.0]).
    """
    np.random.seed(seed)
    features = dict((name, -1.0+np.random.randint(0,21)/10.0) for name in feature_names)
    return features

if __name__ == "__main__":
    main()
