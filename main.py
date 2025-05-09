from extract_features import run
from resnet import i3_res50
import os
from pathlib import Path
import shutil
import argparse
import numpy as np
import time
# import ffmpeg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
import subprocess


def generate(datasetpath, outputpath, pretrainedpath, frequency, batch_size, sample_mode):
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    # temppath = outputpath+ "/temp/"
    rootdir = Path(datasetpath)
    frame_dirs = [f for f in rootdir.iterdir() if f.is_dir()]


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	i3d = i3_res50(400, pretrainedpath)
	i3d.to(device)
	i3d.train(False)  # Set model to evaluate mode
	for frame_dirs in frame_dirs:
    	folder_name = frame_dirs.name
		start_time = time.time()
		print(f"Generating features for {folder_name}")

# ffmpeg.input(video).output('{}%d.jpg'.format(temppath),start_number=0).global_args('-loglevel', 'quiet').run()
# ffmpeg.input(video).output(os.path.join(temppath, '%d.jpg'), start_number=0).global_args('-loglevel',

        print("Preprocessing done..")
        features = run(i3d, frequency, str(frame_dir), batch_size, sample_mode)
        np.save(os.path.join(outputpath, folder_name), features)
        print(f"Obtained features of size: {features.shape}")
        print(f"Done in {time.time() - start_time:.2f} seconds.")
#######

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath', type=str, default="/kaggle/working/train")
    parser.add_argument('--outputpath', type=str, default="/kaggle/working/feauter_train")
    parser.add_argument('--pretrainedpath', type=str,
                        default="/kaggle/working/I3D_Feature_Extraction_resnet/pretrained/i3d_r50_kinetics.pth")
    parser.add_argument('--frequency', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--sample_mode', type=str, default="oversample")
    args = parser.parse_args()
    generate(args.datasetpath, str(args.outputpath), args.pretrainedpath, args.frequency, args.batch_size,
             args.sample_mode)
