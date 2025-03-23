from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import ffmpeg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from extract_features import run
from resnet import i3_res50
import os


def generate(framepath, labelpath, outputpath, pretrainedpath, frequency, batch_size, sample_mode):
	Path(outputpath).mkdir(parents=True, exist_ok=True)

	# Get all frame directories
	root_frames = Path(framepath)
	frame_dirs = [f for f in root_frames.iterdir() if f.is_dir()]

	# setup the model
	i3d = i3_res50(400, pretrainedpath)
	i3d.cuda()
	i3d.train(False)  # Set model to evaluate mode

	for frame_dir in frame_dirs:
		folder_name = frame_dir.name
		start_time = time.time()
		print(f"Generating features for {folder_name}")

		# No need for temp directory since images are already extracted
		# Run feature extraction directly on the jpg files in the folder
		features = run(i3d, frequency, str(frame_dir), batch_size, sample_mode)

		# Save features with the same name as the folder
		np.save(os.path.join(outputpath, folder_name), features)
		print(f"Obtained features of size: {features.shape}")
		print(f"Done in {time.time() - start_time:.2f} seconds.")
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--framepath', type=str, default="/kaggle/input/shanghaitec-vad-train/frames", help="Path to directory containing frame folders")
	parser.add_argument('--labelpath', type=str, default="/kaggle/input/shanghaitec-vad-train/label",
						help="Path to directory containing label files (.npy)")
	parser.add_argument('--outputpath', type=str, default="output")
	parser.add_argument('--pretrainedpath', type=str, default="pretrained/i3d_r50_kinetics.pth")
	parser.add_argument('--frequency', type=int, default=16)
	parser.add_argument('--batch_size', type=int, default=20)
	parser.add_argument('--sample_mode', type=str, default="oversample")
	args = parser.parse_args()

	generate(args.framepath, args.labelpath, str(args.outputpath), args.pretrainedpath,
			 args.frequency, args.batch_size, args.sample_mode)


##################################


