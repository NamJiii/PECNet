import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader
import argparse
import copy
sys.path.append("../utils/")
import matplotlib.pyplot as plt
import numpy as np
from models import *
from social_utils import *
import yaml

parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--load_file', '-lf', default="run7.pt")
parser.add_argument('--num_trajectories', '-nt', default=20) #number of trajectories to sample
parser.add_argument('--verbose', '-v', action='store_true')
parser.add_argument('--root_path', '-rp', default="./")

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(args.gpu_index)
print(device)


checkpoint = torch.load('../saved_models/{}'.format(args.load_file), map_location=device)
hyper_params = checkpoint["hyper_params"]

print(hyper_params)
def inference(test_dataset, model, best_of_n = 1):
	model.eval()
	assert best_of_n >= 1 and type(best_of_n) == int
	test_loss = 0

	with torch.no_grad():
		for i, (traj, mask, initial_pos) in enumerate(zip(test_dataset.trajectory_batches, test_dataset.mask_batches, test_dataset.initial_pos_batches)):
			traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
			x = traj[:, :hyper_params["past_length"], :]#2829*20*2 => 2829*8*2
			sample_data = [[[0.0,0.0],
							[710.0,10.0],
							[620.0,20.0],
							[530.0,30.0],
							[440.0,40.0],
							[350.0,50.0],
							[260.0,60.0],
							[170.0,70.0]],
						   [[0.0,0.0],
							[10.0,1.0],
							[20.0,2.0],
							[30.0,3.0],
							[40.0,4.0],
							[50.0,5.0],
							[60.0,6.0],
							[70.0,7.0]]]#2*5*2
			#print('x : ',x[:10,:,:])
			x = x.contiguous().view(-1, x.shape[1]*x.shape[2])#2829*8*2 => 2829*16
			#print('x : ', x[:10, :])
			sample_data = torch.DoubleTensor(sample_data).to(device)
			sample_data = sample_data.contiguous().view(-1,sample_data.shape[1]*sample_data.shape[2])
			sample_data = sample_data.to(device)
			x = x.to(device)

			all_guesses = []

			#print('x.shape : ',x)# x는 이중 배열  2829*16
			#print('initial_pos.shape : ', initial_pos)  # x는 이중 배열 2829*2
			init = []
			dest_recon = model.forward(sample_data, init, device=device)#********'''initial_pos[:100,:]''',
			#print('dest_recon : ',dest_recon)#2829*2
			dest_recon = dest_recon.cpu().numpy()

			#print('initial_pos : ',initial_pos[:10,:])
			print('dest_recon : ',dest_recon)

	return dest_recon

def test(test_dataset, model, best_of_n = 1):
	print('==start test==')
	model.eval()
	assert best_of_n >= 1 and type(best_of_n) == int
	test_loss = 0

	with torch.no_grad():
		for i, (traj, mask, initial_pos) in enumerate(zip(test_dataset.trajectory_batches, test_dataset.mask_batches, test_dataset.initial_pos_batches)):
			traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
			#traj는 정답 mask는 문제
			x = traj[:, :hyper_params["past_length"], :]
			y = traj[:, hyper_params["past_length"]:, :]
			y = y.cpu().numpy()
			# reshape the data
			#print(x.shape)
			x = x.contiguous().view(-1, x.shape[1]*x.shape[2])#2829*8*2 => 2829*16
			x = x.to(device)
			#print('--start with x--')
			#print(x.shape)

			future = y[:, :-1, :]
			dest = y[:, -1, :]
			all_l2_errors_dest = []
			all_guesses = []
			for index in range(best_of_n):
				print('x.shape : ',x)# x는 이중 배열  2829*16
				print('initial_pos.shape : ', initial_pos)  # x는 이중 배열 2829*2

				dest_recon = model.forward(x, initial_pos, device=device)#********
				print('dest_recon : ',dest_recon)#2829*2
				dest_recon = dest_recon.cpu().numpy()
				all_guesses.append(dest_recon)

				l2error_sample = np.linalg.norm(dest_recon - dest, axis = 1)#dest_recon은 추측값, dest는 진짜임
				all_l2_errors_dest.append(l2error_sample)
			#추측과 에러내기를 여러번 함
			all_l2_errors_dest = np.array(all_l2_errors_dest)
			all_guesses = np.array(all_guesses)
			# average error
			l2error_avg_dest = np.mean(all_l2_errors_dest)

			# choosing the best guess
			indices = np.argmin(all_l2_errors_dest, axis = 0)

			best_guess_dest = all_guesses[indices,np.arange(x.shape[0]),  :]

			# taking the minimum error out of all guess
			l2error_dest = np.mean(np.min(all_l2_errors_dest, axis = 0))

			# back to torch land
			best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)

			# using the best guess for interpolation
			interpolated_future = model.predict(x, best_guess_dest, mask, initial_pos)
			interpolated_future = interpolated_future.cpu().numpy()
			best_guess_dest = best_guess_dest.cpu().numpy()

			# final overall prediction
			predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis = 1)
			predicted_future = np.reshape(predicted_future, (-1, hyper_params["future_length"], 2))

			print('**REAL :',y,' |Predicted :',predicted_future,'**')

			# ADE error
			l2error_overall = np.mean(np.linalg.norm(y - predicted_future, axis = 2))

			l2error_overall /= hyper_params["data_scale"]
			l2error_dest /= hyper_params["data_scale"]
			l2error_avg_dest /= hyper_params["data_scale"]

			print('Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format(l2error_dest, l2error_avg_dest))
			print('Test time error overall (ADE) best: {:0.3f}'.format(l2error_overall))

	return l2error_overall, l2error_dest, l2error_avg_dest

def main():
	N = args.num_trajectories #number of generated trajectories
	model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
	model = model.double().to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	test_dataset = SocialDataset(set_name="test", b_size=hyper_params["test_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=args.verbose)
	#print(test_dataset.trajectory_batches[0][22])#batch_num X datadict[key] X 20(prev8+post20) by ID pid fid x y
	#return 0
	for traj in test_dataset.trajectory_batches:
		#print('-----',type(traj))
		traj -= traj[:, :1, :] #위치정보를 변위로 바꿈
		#print('=====', traj)
		traj *= hyper_params["data_scale"] #1.86배, 상수임
		#print('=-=-=', traj)

	#average ade/fde for k=20 (to account for variance in sampling)
	num_samples = 1
	average_ade, average_fde = 0, 0
	for i in range(num_samples):
		predicted= inference(test_dataset, model, best_of_n = N)
main()
