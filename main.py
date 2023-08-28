
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
from trainer import train

from random import shuffle
from termcolor import colored

from sklearn.cluster import DBSCAN, KMeans

from featureExtraction import extractFeatures
from torch.backends import cudnn

import faiss
from faiss_utils import search_index_pytorch, search_raw_array_pytorch, index_init_gpu, index_init_cpu

np.random.seed(12)
torch.manual_seed(12)
cudnn.deterministic = True


def main(gpu_ids, base_lr, P, K, tau, beta, k1, sampling, lambda_hard, number_of_iterations, momentum_on_feature_extraction, 
														target, dir_to_save, dir_to_save_metrics, version, eval_freq):


	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
	os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

	num_gpus = torch.cuda.device_count()
	print("Num GPU's:", num_gpus)

	if len(gpu_ids) > 1:
		gpu_indexes = np.arange(num_gpus)[1:].tolist()
	else:
		gpu_indexes = [0]

	print("Allocated GPU's for model:", gpu_indexes)

	
	model_online_resnet50, model_momentum_resnet50, model_online_osnet, model_momentum_osnet, model_online_densenet121, model_momentum_densenet121 = getEnsembles(gpu_indexes)

	#### Load target dataset ####
	train_images_target, gallery_images_target, queries_images_target = load_dataset(target)

	print("Training Size:", train_images_target.shape)
	print("Gallery Size:", gallery_images_target.shape)
	print("Query Size:", queries_images_target.shape)
	
	query_size = queries_images_target.shape[0]

	print("Validating ResNet50 on %s ..." % target)
	cmc, mAP, distmat_resnet50 = validate(queries_images_target, gallery_images_target, model_online_resnet50, gpu_index=gpu_indexes[0])

	print("Validating OSNet on %s ..." % target)
	cmc, mAP, distmat_osnet = validate(queries_images_target, gallery_images_target, model_online_osnet, gpu_index=gpu_indexes[0])

	print("Validating DenseNet121 on %s ..." % target)
	cmc, mAP, distmat_densenet121 = validate(queries_images_target, gallery_images_target, model_online_densenet121, gpu_index=gpu_indexes[0])

	distmat_ensembled = (distmat_resnet50 + distmat_osnet + distmat_densenet121)/3
	calculateMetrics(distmat_ensembled, queries_images_target, gallery_images_target)
	
	#np.random.shuffle(train_images_target)

	cmc_progress = []
	mAP_progress = []
	reliable_data_progress = []
	lr_values = []

	number_of_epoches = 30
	perc = 1.0

	'''
	UNCOMENT HERE!!!
	base_lr_values01 = np.ones(19)*base_lr
	base_lr_values02 = np.linspace(base_lr, base_lr/10, num=11)
	base_lr_values = np.concatenate((base_lr_values01, base_lr_values02))
	'''

	base_lr_values01 = np.linspace(base_lr/10, base_lr, num=10)
	base_lr_values02 = np.linspace(base_lr, base_lr, num=30)
	base_lr_values03 = np.linspace(base_lr/10, base_lr/10, num=10)
	base_lr_values = np.concatenate((base_lr_values01, base_lr_values02, base_lr_values03))

	optimizer_resnet50 = torch.optim.Adam(model_online_resnet50.parameters(), lr=base_lr, weight_decay=5e-4)
	optimizer_osnet = torch.optim.Adam(model_online_osnet.parameters(), lr=base_lr, weight_decay=5e-4)
	optimizer_densenet121 = torch.optim.Adam(model_online_densenet121.parameters(), lr=base_lr, weight_decay=5e-4)

	total_feature_extraction_reranking_time = 0
	total_clustering_time = 0
	total_finetuning_time = 0

	t0_pipeline = time.time()
	for pipeline_iter in range(1, number_of_epoches+1):

		t0 = time.time()
		print("###============ Iteration number %d/%d ============###" % (pipeline_iter, number_of_epoches))

		if momentum_on_feature_extraction == False:

			print("Extracting Online Features for ResNet50 ...")
			train_fvs_resnet50 = extractFeatures(train_images_target, model_online_resnet50, 500, gpu_index=gpu_indexes[0])
			train_fvs_resnet50 = train_fvs_resnet50/torch.norm(train_fvs_resnet50, dim=1, keepdim=True)

			print("Extracting Online Features for OSNet ...")
			train_fvs_osnet = extractFeatures(train_images_target, model_online_osnet, 500, gpu_index=gpu_indexes[0])
			train_fvs_osnet = train_fvs_osnet/torch.norm(train_fvs_osnet, dim=1, keepdim=True)

			print("Extracting Online Features for DenseNet121 ...")
			train_fvs_densenet121 = extractFeatures(train_images_target, model_online_densenet121, 500, gpu_index=gpu_indexes[0])
			train_fvs_densenet121 = train_fvs_densenet121/torch.norm(train_fvs_densenet121, dim=1, keepdim=True)

		else:

			print("Extracting Ensembled Features for ResNet50 ...")
			train_fvs_resnet50 = extractFeatures(train_images_target, model_momentum_resnet50, 500, gpu_index=gpu_indexes[0])
			train_fvs_resnet50 = train_fvs_resnet50/torch.norm(train_fvs_resnet50, dim=1, keepdim=True)

			print("Extracting Ensembled Features for Osnet ...")
			train_fvs_osnet = extractFeatures(train_images_target, model_momentum_osnet, 500, gpu_index=gpu_indexes[0])
			train_fvs_osnet = train_fvs_osnet/torch.norm(train_fvs_osnet, dim=1, keepdim=True)

			print("Extracting Ensembled Features for DenseNet121 ...")
			train_fvs_densenet121 = extractFeatures(train_images_target, model_momentum_densenet121, 500, gpu_index=gpu_indexes[0])
			train_fvs_densenet121 = train_fvs_densenet121/torch.norm(train_fvs_densenet121, dim=1, keepdim=True)
		
		
		distances_resnet50 = compute_jaccard_distance(train_fvs_resnet50, k1=k1)
		distances_resnet50 = np.abs(distances_resnet50)

		distances_osnet = compute_jaccard_distance(train_fvs_osnet, k1=k1)
		distances_osnet = np.abs(distances_osnet)

		distances_densenet121 = compute_jaccard_distance(train_fvs_densenet121, k1=k1)
		distances_densenet121 = np.abs(distances_densenet121)

		distances = (distances_resnet50 + distances_osnet + distances_densenet121)/3
		tf = time.time()
		dt_feature_extraction_reranking = tf - t0
		total_feature_extraction_reranking_time += dt_feature_extraction_reranking

		t0 = time.time()
		selected_images, pseudo_labels, ratio_of_reliable_data = multiClustering(train_images_target, distances)
		tf = time.time()
		dt_clustering = tf - t0
		total_clustering_time += dt_clustering

		del train_fvs_resnet50, train_fvs_osnet, train_fvs_densenet121
		del distances_resnet50, distances_osnet, distances_densenet121, distances

		num_classes = len(np.unique(pseudo_labels))
		calculatePurityAndCameras(selected_images, pseudo_labels)
		print("Number of classes: %d" % num_classes)

		t0 = time.time()
		#lr_value = base_lr_values[pipeline_iter-1]*ratio_of_reliable_data ### UNCOMMENT HERE
		lr_value = base_lr_values[pipeline_iter-1]
		lr_values.append(lr_value)

		print(colored("Learning Rate: %f" % lr_value, "cyan"))
		lambda_lr_warmup(optimizer_resnet50, lr_value)
		lambda_lr_warmup(optimizer_osnet, lr_value)
		lambda_lr_warmup(optimizer_densenet121, lr_value)
	
		print(colored("Training ResNet50 ...", "green"))
		model_online_resnet50, model_momentum_resnet50, optimizer_resnet50 = train(selected_images, pseudo_labels, sampling, 
																					optimizer_resnet50, 
																					P, K, perc, tau, beta, lambda_hard, 
																					number_of_iterations, 
																					model_online_resnet50, 
																					model_momentum_resnet50, gpu_indexes)

		print(colored("Training OSNet ...", "green"))
		model_online_osnet, model_momentum_osnet, optimizer_osnet = train(selected_images, pseudo_labels, sampling, 
																					optimizer_osnet, 
																					P, K, perc, tau, beta, lambda_hard, 
																					number_of_iterations, 
																					model_online_osnet, 
																					model_momentum_osnet, gpu_indexes)

		print(colored("Training DenseNet121 ...", "green"))
		model_online_densenet121, model_momentum_densenet121, optimizer_densenet121 = train(selected_images, pseudo_labels,
																					sampling, 
																					optimizer_densenet121, 
																					P, K, perc, tau, beta, lambda_hard, 
																					number_of_iterations, 
																					model_online_densenet121, 
																					model_momentum_densenet121, gpu_indexes)
		tf = time.time()
		dt_finetuning = tf - t0
		total_finetuning_time += dt_finetuning

		if pipeline_iter % eval_freq == 0:
			print(colored("Validating online ResNet50 ...", "yellow"))
			cmc, mAP, distmat_online_resnet50 = validate(queries_images_target, gallery_images_target, 
																		model_online_resnet50, gpu_index=gpu_indexes[0])

			print(colored("Validating momentum ResNet50 ...", "yellow"))
			cmc, mAP, distmat_momentum_resnet50 = validate(queries_images_target, gallery_images_target, 
																		model_momentum_resnet50, gpu_index=gpu_indexes[0])

			print(colored("Validating online OSNet ...", "yellow"))
			cmc, mAP, distmat_online_osnet = validate(queries_images_target, gallery_images_target, 
																		model_online_osnet, gpu_index=gpu_indexes[0])

			print(colored("Validating momentum OSNet ...", "yellow"))
			cmc, mAP, distmat_momentum_osnet = validate(queries_images_target, gallery_images_target, 
																		model_momentum_osnet, gpu_index=gpu_indexes[0])

			print(colored("Validating online DenseNet121 ...", "yellow"))
			cmc, mAP, distmat_online_densenet121 = validate(queries_images_target, gallery_images_target, 
																		model_online_densenet121, gpu_index=gpu_indexes[0])

			print(colored("Validating momentum DenseNet121 ...", "yellow"))
			cmc, mAP, distmat_momentum_densenet121 = validate(queries_images_target, gallery_images_target, 
																		model_momentum_densenet121, gpu_index=gpu_indexes[0])
			
			distmat_ensembled_online = (distmat_online_resnet50 + distmat_online_osnet + distmat_online_densenet121)/3
			distmat_ensembled_momentum = (distmat_momentum_resnet50 + distmat_momentum_osnet + distmat_momentum_densenet121)/3

			print(colored("Validating ensembled with online models ...", "yellow"))
			cmc, mAP = calculateMetrics(distmat_ensembled_online, queries_images_target, gallery_images_target)

			print(colored("Validating ensembled with momentum models ...", "yellow"))
			cmc, mAP = calculateMetrics(distmat_ensembled_momentum, queries_images_target, gallery_images_target)
			
			cmc_progress.append(cmc)
			mAP_progress.append(mAP)

			joblib.dump(cmc_progress, "%s/CMC_%s_%s" % (dir_to_save_metrics, "To" + target, version))
			joblib.dump(mAP_progress, "%s/mAP_%s_%s" % (dir_to_save_metrics, "To" + target, version))

		reliable_data_progress.append(ratio_of_reliable_data)

		torch.save(model_online_resnet50.state_dict(), "%s/model_online_%s_%s_%s.h5" % (dir_to_save, "To" + target, "resnet50", version))
		torch.save(model_momentum_resnet50.state_dict(), "%s/model_momentum_%s_%s_%s.h5" % (dir_to_save, "To" + target, "resnet50", version))

		torch.save(model_online_osnet.state_dict(), "%s/model_online_%s_%s_%s.h5" % (dir_to_save, "To" + target, "osnet", version))
		torch.save(model_momentum_osnet.state_dict(), "%s/model_momentum_%s_%s_%s.h5" % (dir_to_save, "To" + target, "osnet", version))

		torch.save(model_online_densenet121.state_dict(), "%s/model_online_%s_%s_%s.h5" % (dir_to_save, "To" + target, "densenet121", version))
		torch.save(model_momentum_densenet121.state_dict(), "%s/model_momentum_%s_%s_%s.h5" % (dir_to_save, "To" + target, "densenet121", version))

		joblib.dump(reliable_data_progress, "%s/reliability_progress_%s_%s" % (dir_to_save_metrics, "To" + target, version))
		joblib.dump(lr_values, "%s/lr_progress_%s_%s" % (dir_to_save_metrics, "To" + target, version))

		#joblib.dump(progress_loss, "%s/loss_progress_%s_%s_%s" % (dir_to_save_metrics, "To" + target, model_name, version))
		#joblib.dump(number_of_clusters, "%s/number_clusters_%s_%s_%s" % (dir_to_save_metrics, "To" + target, model_name, version))

	tf_pipeline = time.time()
	total_pipeline_time = tf_pipeline - t0_pipeline

	mean_feature_extraction_reranking_time = total_feature_extraction_reranking_time/number_of_epoches
	mean_clustering_time = total_clustering_time/number_of_epoches
	mean_finetuning_time = total_finetuning_time/number_of_epoches

	print(total_feature_extraction_reranking_time, total_clustering_time, total_finetuning_time)
	print("Mean Feature Extraction and Reranking Time: %f" % mean_feature_extraction_reranking_time)
	print("Mean Clustering Time: %f" % mean_clustering_time)
	print("Mean Finetuning Time: %f" % mean_finetuning_time)
	print("Total pipeline Time:  %f" % total_pipeline_time)

def multiClustering(images, distances):

	N = distances.shape[0]
	eps_values = np.linspace(0.5, 0.7, num=5)
	pseudo_labels_along_clustering = []

	for eps in eps_values:
		clusterer = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
		clusterer.fit(distances)
		proposed_labels = clusterer.labels_

		pseudo_labels_along_clustering.append(proposed_labels)

	pseudo_labels_along_clustering = np.array(pseudo_labels_along_clustering)	
	last_clustering_pseudo_labels = pseudo_labels_along_clustering[-1]

	selecting_mask = np.ones(N)
	for i in range(N):
		if -1 in pseudo_labels_along_clustering[:,i]:
			selecting_mask[i] = 0

	# To apply mask, we let all labels between 0 <= label <= N (outliers are 0 after transformation)
	last_clustering_pseudo_labels = last_clustering_pseudo_labels + 1
	# Apply mask letting only samples that have never been outliers
	last_clustering_pseudo_labels = last_clustering_pseudo_labels*selecting_mask
	# Let labels again in -1 <= label <= N-1 range where -1 are again the outliers
	last_clustering_pseudo_labels = last_clustering_pseudo_labels - 1

	#print(pseudo_labels_along_clustering[:,:50])
	#print(selecting_mask[:50])
	#print(last_clustering_pseudo_labels[:50])
	selected_images = images[last_clustering_pseudo_labels != -1]
	pseudo_labels = last_clustering_pseudo_labels[last_clustering_pseudo_labels != -1]

	ratio_of_reliable_data = pseudo_labels.shape[0]/last_clustering_pseudo_labels.shape[0]
	print("Reliability: %.3f" % ratio_of_reliable_data)

	return selected_images, pseudo_labels, ratio_of_reliable_data

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

def calculatePurityAndCameras(images, pseudo_labels):

	unique_pseudo_labels = np.unique(pseudo_labels)

	if unique_pseudo_labels[0] == -1:
		j = 1
	else:
		j = 0

	H = 0
	cameras_quantities = []

	for lab in unique_pseudo_labels[j:]:
		#print(colored("Statistics on Cluster %d" % lab, "green"))
		cluster_images = images[pseudo_labels == lab]
		#print("Cluster Images:", cluster_images)
		cluster_true_ids = cluster_images[:,1]
		true_ids, freq_ids = np.unique(cluster_true_ids, return_counts=True)
		#print("ID's and Frequencies:", true_ids, freq_ids)
		ratio_freq_ids = freq_ids/np.sum(freq_ids)
		#print("Frequencies in ratio:", ratio_freq_ids)
		cluster_purity = np.sum(-1*ratio_freq_ids*np.log2(ratio_freq_ids))
		#print("Cluster Purity", cluster_purity)
		H += cluster_purity

		cluster_cameras, cameras_freqs = np.unique(cluster_images[:,2], return_counts=True)
		#print("Cameras and Frequencies:", cluster_cameras, cameras_freqs)
		#print("There are %d cameras" % len(cameras_freqs))
		cameras_quantities.append(len(cameras_freqs))


	#print("Total number of clusters:", len(unique_pseudo_labels[j:]))
	mean_purity = H/len(unique_pseudo_labels[j:])
	print(colored("Mean Purity: %.5f" % mean_purity, "green"))

	numbers_of_cameras, cameras_freqs = np.unique(cameras_quantities, return_counts=True)

	for i in range(len(numbers_of_cameras)):
		print(colored("There are %d clusters with %d cameras" % (cameras_freqs[i], numbers_of_cameras[i]), "blue"))

def lambda_lr_warmup(optimizer, lr_value):

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr_value

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]


def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()
    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    if (search_option==0):
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==1):
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==2):
        # GPU
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)


    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]]+np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1-temp_min/(2-temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Define the UDA parameters')
	
	parser.add_argument('--gpu_ids', type=str, default="7", help='GPU IDs')
	parser.add_argument('--lr', type=float, default=3.5e-4, help='Learning Rate')
	parser.add_argument('--P', type=int, default=16, help='Number of Persons')
	parser.add_argument('--K', type=int, default=4, help='Number of samples per person')
	parser.add_argument('--tau', type=float, default=0.05, help='tau value used on softmax triplet loss')
	parser.add_argument('--beta', type=float, default=0.999, help='beta used on self-Ensembling')
	parser.add_argument('--k1', type=int, default=30, help='k on k-Reciprocal Encoding')
	parser.add_argument('--sampling', type=str, default="mean", help='Mean or Random feature vectors to be prototype')
	parser.add_argument('--lambda_hard', type=float, default=0.5, help='tuning prameter of Softmax Triplet Loss')	
	parser.add_argument('--num_iter', type=int, default=400, help='Number of iterations on an epoch')
	parser.add_argument('--momentum_on_feature_extraction', type=int, default=0, 
																		help='If it is the momentum used on feature extraction')	
	parser.add_argument('--target', type=str, help='Name of target dataset')
	parser.add_argument('--path_to_save_models', type=str, help='Path to save models')
	parser.add_argument('--path_to_save_metrics', type=str, help='Path to save metrics (mAP, CMC, ...)')
	parser.add_argument('--version', type=str, help='Path to save models')
	parser.add_argument('--eval_freq', type=int, help='Evaluation Frequency along training')
	
	args = parser.parse_args()

	gpu_ids = args.gpu_ids
	base_lr = args.lr
	P = args.P
	K = args.K
	
	tau = args.tau
	beta = args.beta
	k1 = args.k1
	sampling  = args.sampling
	
	lambda_hard = args.lambda_hard
	number_of_iterations = args.num_iter

	momentum_on_feature_extraction = bool(args.momentum_on_feature_extraction)

	target = args.target
	dir_to_save = args.path_to_save_models
	dir_to_save_metrics = args.path_to_save_metrics
	version = args.version
	eval_freq = args.eval_freq

	main(gpu_ids, base_lr, P, K, tau, beta, k1, sampling, lambda_hard, number_of_iterations, momentum_on_feature_extraction, 
															target, dir_to_save, dir_to_save_metrics, version, eval_freq)




