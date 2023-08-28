
import torch
import torchreid
import torchvision

from torchvision.models import resnet50, densenet121, inception_v3
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, RandomErasing, RandomHorizontalFlip, ColorJitter
from torchvision.transforms import RandomCrop, functional, GaussianBlur
from torch.nn import Module, Dropout, BatchNorm1d, Linear, AdaptiveAvgPool2d, CrossEntropyLoss, ReLU, AvgPool2d, AdaptiveMaxPool2d
from torch.nn import functional as F
from torch import nn

from transformers import BertTokenizer, GPT2Tokenizer, T5Tokenizer
from transformers import BertModel, GPT2Model, T5Model, RobertaModel
from transformers import AutoModel, AutoTokenizer
from transformers import BertConfig, RobertaConfig, T5Config

import os
import copy

from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import numpy as np
import time
import argparse
import joblib
import logging
import pickle
import json

from DCNNs import getDCNN, getEnsembles, calculateMetrics, validate_text
from datasetUtils import load_text_dataset
from trainer_text import train

from random import shuffle
from termcolor import colored

from sklearn.cluster import DBSCAN, KMeans

from featureExtraction import extractTextFeatures
from torch.backends import cudnn

import faiss
from faiss_utils import search_index_pytorch, search_raw_array_pytorch, index_init_gpu, index_init_cpu

np.random.seed(12)
torch.manual_seed(12)
cudnn.deterministic = True

tokenizer_t5 = T5Tokenizer.from_pretrained("t5-small")


def main(gpu_ids, pretrained, base_lr, P, K, num_concatenated_tweets, tau, beta, k1, sampling, lambda_hard, number_of_iterations, 
						momentum_on_feature_extraction, base_dir, authors_set, dir_to_save, dir_to_save_metrics, version, eval_freq):

	
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
	os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

	num_gpus = torch.cuda.device_count()
	print("Num GPU's:", num_gpus)

	if len(gpu_ids) > 1:
		gpu_indexes = np.arange(num_gpus)[1:].tolist()
	else:
		gpu_indexes = [0]

	print("Allocated GPU's for model:", gpu_indexes)


	# Loading BERT
	if pretrained:
		model_online_bert = BertModel.from_pretrained(
				"bert-base-cased", # Use the 12-layer BERT model, with an cased vocab.
				output_attentions = False, # Whether the model returns attentions weights.
				output_hidden_states = False, # Whether the model returns all hidden-states.
		)

		model_momentum_bert = BertModel.from_pretrained(
				"bert-base-cased", # Use the 12-layer BERT model, with an cased vocab.
				output_attentions = False, # Whether the model returns attentions weights.
				output_hidden_states = False, # Whether the model returns all hidden-states.
		)

	else:

		configuration = BertModel.from_pretrained("bert-base-cased", output_attentions = False, 
																			output_hidden_states = False).config
		model_online_bert = BertModel(configuration)
		model_momentum_bert = BertModel(configuration)


	model_online_bert = BertFeatures(model_online_bert)
	#model_online_bert = DistilBertFeatures(model_online_bert)
	model_online_bert = nn.DataParallel(model_online_bert, device_ids=gpu_indexes)

	#model_momentum_bert = DistilBertFeatures(model_momentum_bert)
	model_momentum_bert = BertFeatures(model_momentum_bert)
	model_momentum_bert = nn.DataParallel(model_momentum_bert, device_ids=gpu_indexes)

	model_momentum_bert.load_state_dict(model_online_bert.state_dict())

	model_online_bert = model_online_bert.cuda(gpu_indexes[0])
	model_online_bert = model_online_bert.eval()

	model_momentum_bert = model_momentum_bert.cuda(gpu_indexes[0])
	model_momentum_bert = model_momentum_bert.eval()



	'''
	# Loading GPT-2
	model_online_gpt2 = GPT2Model.from_pretrained(
			"gpt2", 
			output_attentions = False, # Whether the model returns attentions weights.
			output_hidden_states = False, # Whether the model returns all hidden-states.
	)

	model_momentum_gpt2 = GPT2Model.from_pretrained(
			"gpt2", 
			output_attentions = False, # Whether the model returns attentions weights.
			output_hidden_states = False, # Whether the model returns all hidden-states.
	)

	model_online_gpt2 = GPT2Features(model_online_gpt2)
	model_online_gpt2 = nn.DataParallel(model_online_gpt2, device_ids=gpu_indexes)

	model_momentum_gpt2 = GPT2Features(model_momentum_gpt2)
	model_momentum_gpt2 = nn.DataParallel(model_momentum_gpt2, device_ids=gpu_indexes)

	model_momentum_gpt2.load_state_dict(model_online_gpt2.state_dict())

	model_online_gpt2 = model_online_gpt2.cuda(gpu_indexes[0])
	model_online_gpt2 = model_online_gpt2.eval()

	model_momentum_gpt2 = model_momentum_gpt2.cuda(gpu_indexes[0])
	model_momentum_gpt2 = model_momentum_gpt2.eval()
	'''

	# Loading tweet-BERT
	if pretrained:
		model_online_tweetBert = AutoModel.from_pretrained(
				"vinai/bertweet-base", # Use the 12-layer BERT model, with an cased vocab.
				output_attentions = False, # Whether the model returns attentions weights.
				output_hidden_states = False, # Whether the model returns all hidden-states.
		)

		model_momentum_tweetBert = AutoModel.from_pretrained(
				"vinai/bertweet-base", # Use the 12-layer BERT model, with an cased vocab.
				output_attentions = False, # Whether the model returns attentions weights.
				output_hidden_states = False, # Whether the model returns all hidden-states.
		)

	else:

		configuration = AutoModel.from_pretrained("vinai/bertweet-base", output_attentions = False, 
																					output_hidden_states = False).config

		#configuration = RobertaConfig()
		model_online_tweetBert = RobertaModel(configuration)
		model_momentum_tweetBert = RobertaModel(configuration)



	model_online_tweetBert = BertFeatures(model_online_tweetBert)
	#model_online_tweetBert = DistilBertFeatures(model_online_tweetBert)
	model_online_tweetBert = nn.DataParallel(model_online_tweetBert, device_ids=gpu_indexes)

	#model_momentum_bert = DistilBertFeatures(model_momentum_bert)
	model_momentum_tweetBert = BertFeatures(model_momentum_tweetBert)
	model_momentum_tweetBert = nn.DataParallel(model_momentum_tweetBert, device_ids=gpu_indexes)

	model_momentum_tweetBert.load_state_dict(model_online_tweetBert.state_dict())

	model_online_tweetBert = model_online_tweetBert.cuda(gpu_indexes[0])
	model_online_tweetBert = model_online_tweetBert.eval()

	model_momentum_tweetBert = model_momentum_tweetBert.cuda(gpu_indexes[0])
	model_momentum_tweetBert = model_momentum_tweetBert.eval()

	# Loading T5
	if pretrained:
		model_online_t5 = T5Model.from_pretrained("t5-small", pad_token_id=tokenizer_t5.pad_token_id, output_attentions = False, 
																									output_hidden_states = False)

		model_momentum_t5 = T5Model.from_pretrained("t5-small", pad_token_id=tokenizer_t5.pad_token_id, output_attentions = False, 
																									output_hidden_states = False)

	else:

		configuration = T5Model.from_pretrained("t5-small", pad_token_id=tokenizer_t5.pad_token_id, 
													output_attentions = False, output_hidden_states = False).config
		model_online_t5 = T5Model(configuration)
		model_momentum_t5 = T5Model(configuration)



	model_online_t5 = T5Features(model_online_t5)
	model_online_t5 = nn.DataParallel(model_online_t5, device_ids=gpu_indexes)

	model_momentum_t5 = T5Features(model_momentum_t5)
	model_momentum_t5 = nn.DataParallel(model_momentum_t5, device_ids=gpu_indexes)

	model_momentum_t5.load_state_dict(model_online_t5.state_dict())

	model_online_t5 = model_online_t5.cuda(gpu_indexes[0])
	model_online_t5 = model_online_t5.eval()

	model_momentum_t5 = model_momentum_t5.cuda(gpu_indexes[0])
	model_momentum_t5 = model_momentum_t5.eval()
	
	### Load text models ###
	#model_online_bert, model_momentum_bert, model_online_osnet, model_momentum_osnet, model_online_densenet121, model_momentum_densenet121 = getEnsembles(gpu_indexes)
	train_text, gallery_text, query_text = load_text_dataset(base_dir, authors_set)

	print("Training Size:", train_text.shape)
	print("Gallery Size:", gallery_text.shape)
	print("Query Size:", query_text.shape)

	print("Validating BERT ...")
	cmc, mAP, distmat_bert = validate_text(query_text, gallery_text, model_online_bert, 'bert', gpu_index=gpu_indexes[0])

	print("Validating tweet-BERT ...")
	cmc, mAP, distmat_tweetBert = validate_text(query_text, gallery_text, model_online_tweetBert, 'tweet-bert', gpu_index=gpu_indexes[0])

	#print("Validating GPT-2 ...")
	#cmc, mAP, distmat_gpt2 = validate_text(query_text, gallery_text, model_online_gpt2, 'gpt2', gpu_index=gpu_indexes[0])

	print("Validating T5 ...")
	cmc, mAP, distmat_t5 = validate_text(query_text, gallery_text, model_online_t5, 't5', gpu_index=gpu_indexes[0])

	#print("Validating Custom ...")
	#cmc, mAP, distmat_custom = validate_text(query_text, gallery_text, model_online_custom, 'custom', gpu_index=gpu_indexes[0])
	
	#distmat_ensembled = (distmat_bert + distmat_gpt2 + distmat_t5)/3
	distmat_ensembled = (distmat_bert + distmat_tweetBert + distmat_t5)/3

	calculateMetrics(distmat_ensembled, query_text, gallery_text)
	
	cmc_progress = []
	mAP_progress = []

	cmc_bert_progress = []
	cmc_tweetBert_progress = []
	cmc_t5_progress = []
	cmc_gpt2_progress = []
	cmc_custom_progress = []

	reliable_data_progress = []

	number_of_epoches = 15
	perc = 1.0

	base_lr_values01 = np.linspace(base_lr/10, base_lr, num=10)
	base_lr_values02 = np.linspace(base_lr, base_lr, num=30)
	base_lr_values03 = np.linspace(base_lr/10, base_lr/10, num=10)
	base_lr_values = np.concatenate((base_lr_values01, base_lr_values02, base_lr_values03))

	optimizer_bert = torch.optim.Adam(model_online_bert.parameters(), lr=base_lr, weight_decay=5e-4)
	optimizer_tweetBert = torch.optim.Adam(model_online_tweetBert.parameters(), lr=base_lr, weight_decay=5e-4)
	#optimizer_gpt2 = torch.optim.Adam(model_online_gpt2.parameters(), lr=base_lr, weight_decay=5e-4)
	optimizer_t5 = torch.optim.Adam(model_online_t5.parameters(), lr=base_lr, weight_decay=5e-4)
	#optimizer_custom = torch.optim.Adam(model_online_custom.parameters(), lr=base_lr, weight_decay=0.0)

	for pipeline_iter in range(1, number_of_epoches+1):

		print("###============ Iteration number %d/%d ============###" % (pipeline_iter, number_of_epoches))

		if momentum_on_feature_extraction == False:

			print("Extracting Online Features for BERT ...")
			train_fvs_bert = extractTextFeatures(train_text, model_online_bert, 500, 'bert', gpu_index=gpu_indexes[0])
			train_fvs_bert = train_fvs_bert/torch.norm(train_fvs_bert, dim=1, keepdim=True)

			print("Extracting Online Features for tweet-BERT ...")
			train_fvs_tweetBert = extractTextFeatures(train_text, model_online_tweetBert, 500, 'tweet-bert', gpu_index=gpu_indexes[0])
			train_fvs_tweetBert = train_fvs_tweetBert/torch.norm(train_fvs_tweetBert, dim=1, keepdim=True)

			#print("Extracting Online Features for GPT-2 ...")
			#train_fvs_gpt2 = extractTextFeatures(train_text, model_online_gpt2, 500, 'gpt2', gpu_index=gpu_indexes[0])
			#train_fvs_gpt2 = train_fvs_gpt2/torch.norm(train_fvs_gpt2, dim=1, keepdim=True)

			print("Extracting Online Features for T5...")
			train_fvs_t5 = extractTextFeatures(train_text, model_online_t5, 500, 't5', gpu_index=gpu_indexes[0])
			train_fvs_t5 = train_fvs_t5/torch.norm(train_fvs_t5, dim=1, keepdim=True)

			#print("Extracting Online Features for Custom ...")
			#train_fvs_custom = extractTextFeatures(train_text, model_online_custom, 500, 'custom', gpu_index=gpu_indexes[0])
			#train_fvs_custom = train_fvs_custom/torch.norm(train_fvs_custom, dim=1, keepdim=True)
			
				
		distances_bert = compute_jaccard_distance(train_fvs_bert, k1=k1)
		distances_bert = np.abs(distances_bert)

		distances_tweetBert = compute_jaccard_distance(train_fvs_tweetBert, k1=k1)
		distances_tweetBert = np.abs(distances_tweetBert)

		#distances_gpt2 = compute_jaccard_distance(train_fvs_gpt2, k1=k1)
		#distances_gpt2 = np.abs(distances_gpt2)
		
		distances_t5 = compute_jaccard_distance(train_fvs_t5, k1=k1)
		distances_t5 = np.abs(distances_t5)

		#distances_custom = compute_jaccard_distance(train_fvs_custom, k1=k1)
		#distances_custom = np.abs(distances_custom)

		#distances = (distances_bert + distances_gpt2 + distances_t5)/3
		#distances = (distances_bert + distances_t5 + distances_custom)/3
		distances = (distances_bert + distances_tweetBert + distances_t5)/3
		selected_tweets, pseudo_labels, ratio_of_reliable_data = multiClustering(train_text, distances)

		#del train_fvs_bert, train_fvs_gpt2, train_fvs_t5, #train_fvs_custom
		#del distances_bert, distances_gpt2, distances_t5, distances#distances_custom, distances

		del train_fvs_t5, train_fvs_bert, train_fvs_tweetBert
		del distances_t5, distances_bert, distances_tweetBert, distances

		num_classes = len(np.unique(pseudo_labels))
		calculatePurity(selected_tweets, pseudo_labels)
		print("Number of classes: %d" % num_classes)

		lr_value = base_lr_values[pipeline_iter-1]

		print(colored("Learning Rate: %f" % lr_value, "cyan"))
		lambda_lr_warmup(optimizer_bert, lr_value)
		lambda_lr_warmup(optimizer_tweetBert, lr_value)
		#lambda_lr_warmup(optimizer_gpt2, lr_value)
		lambda_lr_warmup(optimizer_t5, lr_value)
		#lambda_lr_warmup(optimizer_custom, lr_value)
	
		print(colored("Training BERT ...", "green"))
		model_online_bert, model_momentum_bert, optimizer_bert = train(selected_tweets, pseudo_labels, 'bert', sampling, 
																					optimizer_bert, 
																					P, K, num_concatenated_tweets, 
																					perc, tau, beta, lambda_hard, 
																					number_of_iterations, 
																					model_online_bert, 
																					model_momentum_bert, gpu_indexes)

		print(colored("Training tweet-BERT ...", "green"))
		model_online_tweetBert, model_momentum_tweetBert, optimizer_tweetBert = train(selected_tweets, pseudo_labels, 
																					'tweet-bert', sampling, 
																					optimizer_tweetBert, 
																					P, K, num_concatenated_tweets, 
																					perc, tau, beta, lambda_hard, 
																					number_of_iterations, 
																					model_online_tweetBert, 
																					model_momentum_tweetBert, gpu_indexes)

		'''
		print(colored("Training GPT-2 ...", "green"))
		model_online_gpt2, model_momentum_gpt2, optimizer_gpt2 = train(selected_tweets, pseudo_labels, 'gpt2', sampling, 
																					optimizer_gpt2, 
																					P, K, num_concatenated_tweets, 
																					perc, tau, beta, lambda_hard, 
																					number_of_iterations, 
																					model_online_gpt2, 
																					model_momentum_gpt2, gpu_indexes)
		'''

		print(colored("Training T5 ...", "green"))
		model_online_t5, model_momentum_t5, optimizer_t5 = train(selected_tweets, pseudo_labels, 't5', sampling, 
																					optimizer_t5, 
																					P, K, num_concatenated_tweets,
																					perc, tau, beta, lambda_hard, 
																					number_of_iterations, 
																					model_online_t5, 
																					model_momentum_t5, gpu_indexes)

		
		'''
		print(colored("Training Custom ...", "green"))
		model_online_custom, model_momentum_custom, optimizer_custom = train(selected_tweets, pseudo_labels, 'custom',
																					sampling, 
																					optimizer_custom, 
																					P, K, num_concatenated_tweets, 
																					perc, tau, beta, lambda_hard, 
																					number_of_iterations, 
																					model_online_custom, 
																					model_momentum_custom, gpu_indexes)
		'''
		

		if pipeline_iter % eval_freq == 0:
			print(colored("Validating online BERT ...", "yellow"))
			cmc, mAP, distmat_online_bert = validate_text(query_text, gallery_text, 
																		model_online_bert, 'bert', gpu_index=gpu_indexes[0])

			print(colored("Validating momentum BERT ...", "yellow"))
			cmc, mAP, distmat_momentum_bert = validate_text(query_text, gallery_text, 
																		model_momentum_bert, 'bert', gpu_index=gpu_indexes[0])

			cmc_bert_progress.append(cmc)


			print(colored("Validating online tweet-BERT ...", "yellow"))
			cmc, mAP, distmat_online_tweetBert = validate_text(query_text, gallery_text, 
																		model_online_tweetBert, 'tweet-bert', gpu_index=gpu_indexes[0])

			print(colored("Validating momentum tweet-BERT ...", "yellow"))
			cmc, mAP, distmat_momentum_tweetBert = validate_text(query_text, gallery_text, 
																		model_momentum_tweetBert, 'tweet-bert', gpu_index=gpu_indexes[0])

			cmc_tweetBert_progress.append(cmc)

			#print(colored("Validating online GPT-2 ...", "yellow"))
			#cmc, mAP, distmat_online_gpt2 = validate_text(query_text, gallery_text, 
			#															model_online_gpt2, 'gpt2', gpu_index=gpu_indexes[0])

			#print(colored("Validating momentum GPT-2 ...", "yellow"))
			#cmc, mAP, distmat_momentum_gpt2 = validate_text(query_text, gallery_text, 
			#															model_momentum_gpt2, 'bert', gpu_index=gpu_indexes[0])

			#cmc_gpt2_progress.append(cmc)

			
			print(colored("Validating online T5 ...", "yellow"))
			cmc, mAP, distmat_online_t5 = validate_text(query_text, gallery_text, 
																		model_online_t5, 't5', gpu_index=gpu_indexes[0])

			print(colored("Validating momentum T5 ...", "yellow"))
			cmc, mAP, distmat_momentum_t5 = validate_text(query_text, gallery_text, 
																		model_momentum_t5, 't5', gpu_index=gpu_indexes[0])

			cmc_t5_progress.append(cmc)
			
			'''
			print(colored("Validating online Custom ...", "yellow"))
			cmc, mAP, distmat_online_custom = validate_text(query_text, gallery_text, 
																		model_online_custom, 'custom', gpu_index=gpu_indexes[0])

			print(colored("Validating momentum Custom ...", "yellow"))
			cmc, mAP, distmat_momentum_custom = validate_text(query_text, gallery_text, 
																		model_momentum_custom, 'custom', gpu_index=gpu_indexes[0])
			
			cmc_custom_progress.append(cmc)
			'''

			distmat_ensembled_online = (distmat_online_bert + distmat_online_tweetBert + distmat_online_t5)/3
			distmat_ensembled_momentum = (distmat_momentum_bert + distmat_momentum_tweetBert + distmat_momentum_t5)/3

			#distmat_ensembled_online = (distmat_online_bert + distmat_online_gpt2 + distmat_online_t5)/3
			#distmat_ensembled_momentum = (distmat_momentum_bert + distmat_momentum_gpt2 + distmat_momentum_t5)/3

			print(colored("Validating ensembled with online models ...", "yellow"))
			cmc, mAP = calculateMetrics(distmat_ensembled_online, query_text, gallery_text)

			print(colored("Validating ensembled with momentum models ...", "yellow"))
			cmc, mAP = calculateMetrics(distmat_ensembled_momentum, query_text, gallery_text)
			
			cmc_progress.append(cmc)
			mAP_progress.append(mAP)

			joblib.dump(cmc_bert_progress, "%s/CMC_text_BERT_%s" % (dir_to_save_metrics, version))
			joblib.dump(cmc_tweetBert_progress, "%s/CMC_text_tweetBERT_%s" % (dir_to_save_metrics, version))
			#joblib.dump(cmc_gpt2_progress, "%s/CMC_text_GPT2_%s" % (dir_to_save_metrics, version))
			joblib.dump(cmc_t5_progress, "%s/CMC_text_T5_%s" % (dir_to_save_metrics, version))
			#joblib.dump(cmc_custom_progress, "%s/CMC_text_Custom_%s" % (dir_to_save_metrics, version))

			joblib.dump(cmc_progress, "%s/CMC_text_%s" % (dir_to_save_metrics, version))
			joblib.dump(mAP_progress, "%s/mAP_text_%s" % (dir_to_save_metrics, version))

		reliable_data_progress.append(ratio_of_reliable_data)

		#torch.save(model_online_bert.state_dict(), "%s/model_online_text_%s_%s.h5" % (dir_to_save, "BERT", version))
		#torch.save(model_momentum_bert.state_dict(), "%s/model_momentum_text_%s_%s.h5" % (dir_to_save, "BERT", version))

		#torch.save(model_online_osnet.state_dict(), "%s/model_online_%s_%s_%s.h5" % (dir_to_save, "To" + target, "osnet", version))
		#torch.save(model_momentum_osnet.state_dict(), "%s/model_momentum_%s_%s_%s.h5" % (dir_to_save, "To" + target, "osnet", version))

		#torch.save(model_online_custom.state_dict(), "%s/model_online_%s_%s_%s.h5" % (dir_to_save, "To" + target, "custom", version))
		#torch.save(model_momentum_custom.state_dict(), "%s/model_momentum_%s_%s_%s.h5" % (dir_to_save, "To" + target, "custom", version))

		#joblib.dump(reliable_data_progress, "%s/reliability_progress_%s_%s" % (dir_to_save_metrics, "BERT", version))

		#joblib.dump(progress_loss, "%s/loss_progress_%s_%s_%s" % (dir_to_save_metrics, "To" + target, model_name, version))
		#joblib.dump(number_of_clusters, "%s/number_clusters_%s_%s_%s" % (dir_to_save_metrics, "To" + target, model_name, version))


def multiClustering(tweets, distances):

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
	selected_tweets = tweets[last_clustering_pseudo_labels != -1]
	pseudo_labels = last_clustering_pseudo_labels[last_clustering_pseudo_labels != -1]

	ratio_of_reliable_data = pseudo_labels.shape[0]/last_clustering_pseudo_labels.shape[0]
	print("Reliability: %.3f" % ratio_of_reliable_data)

	return selected_tweets, pseudo_labels, ratio_of_reliable_data

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

def calculatePurity(tweets, pseudo_labels):

	unique_pseudo_labels = np.unique(pseudo_labels)

	if unique_pseudo_labels[0] == -1:
		j = 1
	else:
		j = 0

	H = 0
	cameras_quantities = []

	for lab in unique_pseudo_labels[j:]:
		#print(colored("Statistics on Cluster %d" % lab, "green"))
		cluster_tweets = tweets[pseudo_labels == lab]
		#print("Cluster Images:", cluster_images)
		cluster_true_ids = cluster_tweets[:,1]
		true_ids, freq_ids = np.unique(cluster_true_ids, return_counts=True)
		#print("ID's and Frequencies:", true_ids, freq_ids)
		ratio_freq_ids = freq_ids/np.sum(freq_ids)
		#print("Frequencies in ratio:", ratio_freq_ids)
		cluster_purity = np.sum(-1*ratio_freq_ids*np.log2(ratio_freq_ids))
		#print("Cluster Purity", cluster_purity)
		H += cluster_purity

	#print("Total number of clusters:", len(unique_pseudo_labels[j:]))
	mean_purity = H/len(unique_pseudo_labels[j:])
	print(colored("Mean Purity: %.5f" % mean_purity, "green"))

def lambda_lr_warmup(optimizer, lr_value):

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr_value

## New Definition for BERT-12 without classification layer
class BertFeatures(Module):
    
	def __init__(self, model_base):
		super(BertFeatures, self).__init__()

		self.bert = model_base #.bert
		self.last_bn = BatchNorm1d(768)

	def forward(self, input_token_ids, attention_mask, token_type_ids=None, return_dict=True):
		
		x = self.bert(input_token_ids, token_type_ids=token_type_ids, 
                  attention_mask=attention_mask, return_dict=return_dict)

		#output = self.last_bn(x[1])
		x = x['last_hidden_state'][:,0,:] # Get the first vector from each sentece
		output = self.last_bn(x)
		return output

## New Definition for Tweet-BERT without classification layer
class BertFeatures(Module):
    
	def __init__(self, model_base):
		super(BertFeatures, self).__init__()

		self.bert = model_base #.bert
		self.last_bn = BatchNorm1d(768)

	def forward(self, input_token_ids, attention_mask, token_type_ids=None, return_dict=True):
		
		x = self.bert(input_token_ids, token_type_ids=token_type_ids, 
                  attention_mask=attention_mask, return_dict=return_dict)

		#output = self.last_bn(x[1])
		x = x['last_hidden_state'][:,0,:] # Get the first vector from each sentece
		output = self.last_bn(x)
		return output


## New Definition for DistiliBert without classification layer
class DistilBertFeatures(Module):
    
	def __init__(self, model_base):
		super(DistilBertFeatures, self).__init__()

		self.bert = model_base #.bert
		self.last_bn = BatchNorm1d(768)

	def forward(self, input_token_ids, attention_mask, token_type_ids=None, return_dict=True):
		
		x = self.bert(input_token_ids, attention_mask=attention_mask, return_dict=return_dict)

		#output = self.last_bn(x[1])
		x = x['last_hidden_state'][:,0,:] # Get the first vector from each sentece
		output = self.last_bn(x)
		return output

## New Definition for GPT-2 without classification layer
class GPT2Features(Module):
    
  def __init__(self, model_base):
    super(GPT2Features, self).__init__()

    self.gpt2 = model_base
    self.last_bn = BatchNorm1d(768)

  def forward(self, input_token_ids, attention_mask, token_type_ids=None, return_dict=True):
    
    x = self.gpt2(input_token_ids, token_type_ids=token_type_ids, 
                  attention_mask=attention_mask, return_dict=return_dict)

    #x = x['last_hidden_state'][:,0,:] # Get the first vector from each sentece
    x = torch.mean(x['last_hidden_state'], dim=1) # Get the first vector from each sentece
    output = self.last_bn(x)
    return output

## New Definition for GPT-2 without classification layer
class T5Features(Module):
    
  def __init__(self, model_base):
    super(T5Features, self).__init__()

    self.t5 = model_base
    self.last_bn = BatchNorm1d(512)

  def forward(self, input_token_ids, attention_mask, token_type_ids=None, return_dict=True):
    
    x = self.t5(input_token_ids, decoder_input_ids=input_token_ids, attention_mask=attention_mask, return_dict=return_dict)

    #x = x['last_hidden_state'][:,0,:] # Get the first vector from each sentece
    x = torch.mean(x['encoder_last_hidden_state'], dim=1) # Get the first vector from each sentece
    output = self.last_bn(x)
    return output

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
	parser.add_argument('--pretrained', type=int, default=1, help='Define if the text model is pretrained')
	parser.add_argument('--lr', type=float, default=3.5e-4, help='Learning Rate')
	parser.add_argument('--P', type=int, default=16, help='Number of Persons')
	parser.add_argument('--K', type=int, default=4, help='Number of samples per person')
	parser.add_argument('--num_concatenated_tweets', type=int, default=1, help='Number of concatenated tweets on training')
	parser.add_argument('--tau', type=float, default=0.05, help='tau value used on softmax triplet loss')
	parser.add_argument('--beta', type=float, default=0.999, help='beta used on self-Ensembling')
	parser.add_argument('--k1', type=int, default=30, help='k on k-Reciprocal Encoding')
	parser.add_argument('--sampling', type=str, default="random", help='Mean or Random feature vectors to be prototype')
	parser.add_argument('--lambda_hard', type=float, default=0.5, help='tuning prameter of Softmax Triplet Loss')	
	parser.add_argument('--num_iter', type=int, default=7, help='Number of iterations on an epoch')
	parser.add_argument('--momentum_on_feature_extraction', type=int, default=0, 
																		help='If it is the momentum used on feature extraction')	
	parser.add_argument('--base_dir', type=str, help='base dir where tweets are stored')
	parser.add_argument('--authors_set', type=str, default="50authors", help='500 or 50 authors set')
	parser.add_argument('--path_to_save_models', type=str, help='Path to save models')
	parser.add_argument('--path_to_save_metrics', type=str, help='Path to save metrics (mAP, CMC, ...)')
	parser.add_argument('--version', type=str, help='Path to save models')
	parser.add_argument('--eval_freq', type=int, help='Evaluation Frequency along training')
	
	args = parser.parse_args()

	gpu_ids = args.gpu_ids
	pretrained = bool(args.pretrained)
	print(pretrained)

	base_lr = args.lr
	P = args.P
	K = args.K
	num_concatenated_tweets = args.num_concatenated_tweets
	
	tau = args.tau
	beta = args.beta
	k1 = args.k1
	sampling  = args.sampling
	
	lambda_hard = args.lambda_hard
	number_of_iterations = args.num_iter

	momentum_on_feature_extraction = bool(args.momentum_on_feature_extraction)

	base_dir = args.base_dir
	authors_set = args.authors_set
	dir_to_save = args.path_to_save_models
	dir_to_save_metrics = args.path_to_save_metrics
	version = args.version
	eval_freq = args.eval_freq

	main(gpu_ids, pretrained, base_lr, P, K, num_concatenated_tweets, tau, beta, k1, sampling, lambda_hard, number_of_iterations, 
															momentum_on_feature_extraction, base_dir, authors_set, dir_to_save, 
																					dir_to_save_metrics, version, eval_freq)




