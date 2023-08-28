
import torch
import torchreid
import torchvision

from torchvision.models import resnet50, densenet121, inception_v3
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, RandomErasing, RandomHorizontalFlip, ColorJitter
from torchvision.transforms import RandomCrop, functional, GaussianBlur
from torch.nn import Module, Dropout, BatchNorm1d, Linear, AdaptiveAvgPool2d, CrossEntropyLoss, ReLU, AvgPool2d, AdaptiveMaxPool2d
from torch.nn import functional as F
from torch import nn

import os
import copy

from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import numpy as np
import time
import argparse
import joblib

from DCNNs import getDCNN, getEnsembles, calculateMetrics, validate
from datasetUtils import load_dataset

from random import shuffle
from termcolor import colored

from sklearn.cluster import DBSCAN, KMeans

from featureExtraction import extractFeatures
from torch.backends import cudnn

import faiss
from faiss_utils import search_index_pytorch, search_raw_array_pytorch, index_init_gpu, index_init_cpu

transform = Compose([Resize((256, 128), interpolation=functional.InterpolationMode.BICUBIC), ToTensor(), 
						Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


#transform_augmentation = Compose([RandomHorizontalFlip(p=0.5), 
#								Resize((256, 128), interpolation=functional.InterpolationMode.BICUBIC), 
#								ToTensor(), RandomErasing(p=0.5), 	
#								Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) # =============> It works!

#transform_augmentation = Compose([RandomHorizontalFlip(p=0.5), 
#								Resize((256, 128), interpolation=functional.InterpolationMode.BICUBIC), 
#								ToTensor(), 	
#								Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#								RandomErasing(p=1.0)]) # ===============> It also works!

transform_augmentation = Compose([Resize((256, 128), interpolation=functional.InterpolationMode.BICUBIC), 
                     #RandomCrop((256, 128), padding=10), 
                     RandomHorizontalFlip(p=0.5), 
                     ColorJitter(brightness=0.4, contrast=0.0, saturation=0.0, hue=0.0), 
                     ToTensor(), 
                     RandomErasing(p=1.0), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

np.random.seed(12)
torch.manual_seed(12)
cudnn.deterministic = True

def train(selected_images, pseudo_labels, sampling, optimizer, P, K, perc, tau, beta, lambda_hard, number_of_iterations, 
																						model_online, model_momentum, gpu_indexes):

	
		centers_labels = np.unique(pseudo_labels)
		num_classes = len(centers_labels)

		event_dataset = samplePKBatches(selected_images, pseudo_labels, K=K, perc=perc) 	
		batchLoader = DataLoader(event_dataset, batch_size=min(P, num_classes), num_workers=8, 
													pin_memory=True, shuffle=True, drop_last=True, collate_fn=collate_fn_PK)

		keys = list(model_online.state_dict().keys())
		size = len(keys) 

		model_online.train()
		model_momentum.eval()

		num_batches_computed = 0

		for inner_iter in np.arange(number_of_iterations):
		#while reach_max_number_iterations == False:

			model_online.eval()
			selected_feature_vectors = extractFeatures(selected_images, model_online, 500, gpu_index=gpu_indexes[0])

			centers = []
			for label in centers_labels:

				if sampling == "mean":
					center = torch.mean(selected_feature_vectors[pseudo_labels == label], dim=0, keepdim=True)
				elif sampling == "random":
					N_samples = np.sum(pseudo_labels == label)
					selected_sample_idx = np.random.choice(N_samples) 
					center = selected_feature_vectors[pseudo_labels == label][selected_sample_idx:(selected_sample_idx+1)]

				if len(centers) == 0:
					centers = center
				else:
					centers = torch.cat((centers, center), dim=0)

			centers = centers/torch.norm(centers, dim=1, keepdim=True)
			centers = centers.cuda(gpu_indexes[0])

			model_online.train()

			iteration_loss = 0.0
			iteration_center = 0.0
			iteration_hard = 0.0

			total_corrects = 0
			total_batch_size = 0

			for batch_idx, batch in enumerate(batchLoader):

				initilized = False
				for imgs, labels, pids in batch:

					if initilized:
						batch_imgs = torch.cat((batch_imgs, imgs), dim=0)
						batch_labels = torch.cat((batch_labels, labels), dim=0)
						batch_pids = np.concatenate((batch_pids, pids), axis=0)
					else:
						batch_imgs = imgs
						batch_labels = labels
						batch_pids = pids
						initilized = True

				batch_imgs = batch_imgs.cuda(gpu_indexes[0])

				if batch_imgs.shape[0] <= 2:
					continue

				# Feature Vectors of Original Images
				batch_fvs = model_online(batch_imgs)
				batch_fvs = batch_fvs/torch.norm(batch_fvs, dim=1, keepdim=True)

				batch_center_loss = BatchCenterLoss(batch_fvs, batch_labels, centers, centers_labels, tau=tau, 
																									gpu_index=gpu_indexes[0])
				batch_hard_loss, corrects, total_number_triplets = BatchSoftmaxTripletLoss(batch_fvs, batch_labels, 
																										batch_pids, tau=tau, 
																										gpu_index=gpu_indexes[0])

				total_corrects += corrects
				total_batch_size += total_number_triplets

				batch_loss = batch_center_loss + lambda_hard*batch_hard_loss

				iteration_center += batch_center_loss.item()
				iteration_hard += batch_hard_loss.item()
				iteration_loss += batch_loss.item()

				optimizer.zero_grad()
				batch_loss.backward()
				optimizer.step()

				#centers = UpdateCenters(batch_fvs, batch_labels, centers, centers_labels, alpha=alpha)
				model_online.eval()
				model_online_weights = model_online.state_dict()
				model_momentum_weights = model_momentum.state_dict()

				for i in range(size):	
					model_momentum_weights[keys[i]] =  beta*model_momentum_weights[keys[i]] + (1-beta)*model_online_weights[keys[i]].detach()
					
				model_momentum.load_state_dict(model_momentum_weights)
				model_online.train()

				num_batches_computed += 1
				#if total_number_of_batches >= number_of_iterations:
				#	reach_max_number_iterations = True
				#	break

			TTPR = total_corrects/total_batch_size

			iteration_loss = iteration_loss/(batch_idx+1)
			iteration_center = iteration_center/(batch_idx+1)
			iteration_hard = iteration_hard/(batch_idx+1)


			print(colored("Batches computed: %d, Tau value: %.3f" % (num_batches_computed, tau), "cyan"))
			print(colored("Mean Loss: %.7f, Mean Center Loss: %.7f, Mean Hard Triplet Loss: %.7f" % (iteration_loss, 
																									iteration_center, 
																									iteration_hard), "yellow"))
			
			
			print(colored("Percetage of correct triplets: %.2f" % TTPR, "blue", attrs=['blink']))
			#progress_loss.append(iteration_loss)
			
		model_online.eval()
		model_momentum.eval()

		return model_online, model_momentum, optimizer
		
def BatchCenterLoss(batch_fvs, batch_labels, centers, centers_labels, tau=0.1, gpu_index=0):

	# Calculating Similarity
	S = torch.matmul(batch_fvs, centers.T)
	centers_labels = torch.Tensor(centers_labels)
	
	# Calcuating Loss
	batch_loss = torch.tensor(0.0).cuda(gpu_index)

	for si in range(batch_fvs.shape[0]):
		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]

		#print(colored("###====== Jesus ======###", "blue"))
		#print(fvs_similarities)
		
		# Proxy Loss
		positive_similarity = fvs_similarities[centers_labels == pseudo_label][0]
		#print(positive_similarity)

		pos_sim = torch.exp(positive_similarity/tau)
		all_sim = torch.exp(fvs_similarities/tau).sum()
		batch_loss += -torch.log(pos_sim/all_sim)
		#print(-torch.log(pos_sim/all_sim))

	batch_loss = batch_loss/batch_fvs.shape[0]	

	return batch_loss

def BatchSoftmaxTripletLoss(batch_fvs, batch_labels, batch_pids, tau=0.1, gpu_index=0):
  
	S = torch.mm(batch_fvs, batch_fvs.T)
	batch_loss = torch.tensor(0.0).cuda(gpu_index)
	corrects = 0
	total_number_triplets = 0

	for si in np.arange(S.shape[0]):

		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]
		true_label = batch_pids[si] # Only for reference! It is NOT used on optmization!!

		positive_similarities = fvs_similarities[batch_labels == pseudo_label]
		negative_similarities = fvs_similarities[batch_labels != pseudo_label]

		#print(positive_similarities.shape, negative_similarities.shape)

		p, pos_idx = torch.topk(positive_similarities, k=1, largest=False)
		q, neg_idx = torch.topk(negative_similarities, k=1, largest=True)

		#print(p, pos_idx, q, neg_idx)

		p = torch.exp(p[0]/tau)
		q = torch.exp(q[0]/tau)

		sample_loss = -torch.log(p/(p+q))
		batch_loss += sample_loss

		pos_pid = batch_pids[batch_labels == pseudo_label][pos_idx[0]]
		neg_pid = batch_pids[batch_labels != pseudo_label][neg_idx[0]]

		#print(true_label, pos_pid, neg_pid)

		if (true_label == pos_pid) and (true_label != neg_pid):
			corrects += 1
		total_number_triplets += 1

	loss = batch_loss/S.shape[0]
	return loss, corrects, total_number_triplets


def collate_fn(batch):
	return torch.cat(batch, dim=0)

def collate_fn_PK(batch):
	return batch

class samplePKBatches(Dataset):
    
	def __init__(self, images, pseudo_labels, K=4, perc=0.5):

		self.images_names = images[:,0]
		self.true_ids = images[:,1]
		self.pseudo_labels = pseudo_labels
		self.labels_set = np.unique(pseudo_labels)
		self.K = K

		np.random.shuffle(self.labels_set)
		self.num_of_pseudo_identities_on_epoch = int(round(len(self.labels_set)*perc))
		
	def __getitem__(self, idx):

		pseudo_identity = self.labels_set[idx]
		images_identity = self.images_names[self.pseudo_labels == pseudo_identity]
		true_identities = self.true_ids[self.pseudo_labels == pseudo_identity]

		selected_images_idx = np.random.choice(images_identity.shape[0], size=min(images_identity.shape[0], self.K), replace=False)
		selected_images = images_identity[selected_images_idx]
		selected_true_identities = true_identities[selected_images_idx]
		
		batch_images = []
		for img_name in selected_images:
			imgPIL = torchreid.utils.tools.read_image(img_name)
			augmented_img = torch.stack([transform_augmentation(imgPIL)])

			if len(batch_images) == 0:
				batch_images = augmented_img
			else:
				batch_images = torch.cat((batch_images, augmented_img), dim=0)
	
		batch_labels = torch.ones(batch_images.shape[0])*pseudo_identity
		return batch_images, batch_labels, selected_true_identities
	
	def __len__(self):
		return self.num_of_pseudo_identities_on_epoch