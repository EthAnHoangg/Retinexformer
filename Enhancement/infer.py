import numpy as np
import os
import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import img_as_ubyte

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from basicsr.models import create_model
from basicsr.utils.options import parse

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


parser = argparse.ArgumentParser(description='Single image inference using Retinexformer')
parser.add_argument('--input', required=True, type=str, help='Path to input image or folder')
parser.add_argument('--output_dir', default='./results/infer', type=str, help='Directory to save output')
parser.add_argument('--opt', required=True, type=str, help='Path to option YAML file')
parser.add_argument('--weights', required=True, type=str, help='Path to model weights')
parser.add_argument('--gpus', type=str, default='0', help='GPU devices')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

# Load model
opt = parse(args.opt, is_train=False)
opt['dist'] = False

x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
x['network_g'].pop('type')

model = create_model(opt).net_g
checkpoint = torch.load(args.weights)
try:
    model.load_state_dict(checkpoint['params'])
except Exception:
    new_checkpoint = {'module.' + k: v for k, v in checkpoint['params'].items()}
    model.load_state_dict(new_checkpoint)

model.cuda()
model = nn.DataParallel(model)
model.eval()
print(f"Loaded weights: {args.weights}")

os.makedirs(args.output_dir, exist_ok=True)

# Collect image paths
input_path = args.input
if os.path.isdir(input_path):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_paths = [
        os.path.join(input_path, f)
        for f in sorted(os.listdir(input_path))
        if f.lower().endswith(exts)
    ]
else:
    image_paths = [input_path]

factor = 4

with torch.inference_mode():
    for inp_path in image_paths:
        img = np.float32(utils.load_img(inp_path)) / 255.
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()

        h, w = img_tensor.shape[2], img_tensor.shape[3]
        H = ((h + factor) // factor) * factor
        W = ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        img_tensor = F.pad(img_tensor, (0, padw, 0, padh), 'reflect')

        restored = model(img_tensor)
        restored = restored[:, :, :h, :w]
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        out_name = os.path.splitext(os.path.basename(inp_path))[0] + '.png'
        out_path = os.path.join(args.output_dir, out_name)
        utils.save_img(out_path, img_as_ubyte(restored))
        print(f"Saved: {out_path}")
