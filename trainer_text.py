import torch
import torchreid

from torch.nn import Module, BatchNorm1d
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
import json
import pickle

from DCNNs import getDCNN, getEnsembles, calculateMetrics, validate
from datasetUtils import load_dataset

from random import shuffle
from termcolor import colored

from featureExtraction import extractTextFeatures
from torch.backends import cudnn

import faiss
from faiss_utils import search_index_pytorch, search_raw_array_pytorch, index_init_gpu, index_init_cpu

from transformers import BertTokenizer, GPT2Tokenizer, T5Tokenizer
from transformers import AutoTokenizer

MAX_LENGTH = 128
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer_tweetBert = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
tokenizer_gpt = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer_gpt.add_special_tokens({'pad_token': '<|endoftext|>'})
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-small")

SIZE_BERT = len(tokenizer_bert)
SIZE_GPT2 = len(tokenizer_gpt)
SIZE_T5 = len(tokenizer_t5)

'''
vocabulary_filename = '/work/antonio_gabriel/microblog_authorship_attribution/datasets/twitter_56K-authors_130MM-tweets_2019-01_preprocessed/ngrams_vocabulary/ngrams_vocabulary_7000001.pkl'
with open(vocabulary_filename, mode='rb') as fd:
    vocabulary = pickle.load(fd)

VOCAB_SIZE = len(vocabulary)
print(VOCAB_SIZE)
'''


np.random.seed(12)
torch.manual_seed(12)
cudnn.deterministic = True

def train(selected_tweets, pseudo_labels, model_name, sampling, optimizer, P, K, num_concatenated_tweets, 
																		perc, tau, beta, lambda_hard, number_of_iterations, 
																		model_online, model_momentum, gpu_indexes):

	
		centers_labels = np.unique(pseudo_labels)
		num_classes = len(centers_labels)

		event_dataset = samplePKTextBatches(selected_tweets, pseudo_labels, model_name, K=K, 
																num_concatenated_tweets=num_concatenated_tweets, perc=perc) 

		batchLoader = DataLoader(event_dataset, batch_size=min(P, num_classes), num_workers=8, 
													pin_memory=True, shuffle=True, drop_last=True, collate_fn=collate_fn_PK_text)

		keys = list(model_online.state_dict().keys())
		size = len(keys) 

		model_online.train()
		
		if model_name == 'bert' or model_name == 'tweet-bert':
			model_online.eval()
			for param in model_online.parameters():
				param.requires_grad = False
				#print(param.requires_grad)

			#model_online.module.bert.transformer.layer[5].training = True
			model_online.module.bert.encoder.layer[11].training = True
			for param in model_online.module.bert.encoder.layer[11].parameters():
				param.requires_grad = True

		
			
		elif model_name == 'gpt2':
			model_online.eval()
			for param in model_online.parameters():
				param.requires_grad = False
				#print(param.requires_grad)

			#model_online.module.bert.transformer.layer[5].training = True
			for layer_idx in range(10,12):
				model_online.module.gpt2.h[layer_idx].training = True
				for param in model_online.module.gpt2.h[layer_idx].parameters():
					param.requires_grad = True
			
		

		else:
			model_online.train()
		


		model_momentum.eval()

		num_batches_computed = 0

		for inner_iter in np.arange(number_of_iterations):
		#while reach_max_number_iterations == False:

			model_online.eval()
			selected_feature_vectors = extractTextFeatures(selected_tweets, model_online, 500, model_name, gpu_index=gpu_indexes[0])

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
			
			if model_name == 'bert' or model_name == 'tweet-bert':
				model_online.eval()
				for param in model_online.parameters():
					param.requires_grad = False
					#print(param.requires_grad)

				model_online.module.bert.encoder.layer[11].training = True
				for param in model_online.module.bert.encoder.layer[11].parameters():
					param.requires_grad = True

			
			elif model_name == 'gpt2':
				model_online.eval()
				for param in model_online.parameters():
					param.requires_grad = False
					#print(param.requires_grad)

				#model_online.module.bert.transformer.layer[5].training = True
				for layer_idx in range(10,12):
					model_online.module.gpt2.h[layer_idx].training = True
					for param in model_online.module.gpt2.h[layer_idx].parameters():
						param.requires_grad = True
			
			else:
				model_online.train()
			

			iteration_loss = 0.0
			iteration_center = 0.0
			iteration_hard = 0.0

			total_corrects = 0
			total_pos_corrects = 0
			total_neg_corrects = 0
			total_batch_size = 0

			for batch_idx, batch in enumerate(batchLoader):

				initilized = False
				for input_token_ids, attention_masks, labels, pids in batch:

					if initilized:
						batch_input_tokens_ids = torch.cat((batch_input_tokens_ids, input_token_ids), dim=0)
						batch_attention_masks = torch.cat((batch_attention_masks, attention_masks), dim=0)
						batch_labels = torch.cat((batch_labels, labels), dim=0)
						batch_pids = np.concatenate((batch_pids, pids), axis=0)
					else:
						batch_input_tokens_ids = input_token_ids
						batch_attention_masks = attention_masks
						batch_labels = labels
						batch_pids = pids
						initilized = True

				batch_input_tokens_ids = batch_input_tokens_ids.cuda(gpu_indexes[0])
				batch_attention_masks = batch_attention_masks.cuda(gpu_indexes[0])

				if batch_input_tokens_ids.shape[0] <= 2:
					continue

				# Feature Vectors of Original Images
				if model_name != 'custom':
					batch_fvs = model_online(batch_input_tokens_ids, attention_mask=batch_attention_masks, 
																	token_type_ids=None, return_dict=True)
				else:
					batch_fvs = model_online(batch_input_tokens_ids)

				batch_fvs = batch_fvs/torch.norm(batch_fvs, dim=1, keepdim=True)

				batch_center_loss = BatchCenterLoss(batch_fvs, batch_labels, centers, centers_labels, tau=tau, 
																									gpu_index=gpu_indexes[0])

				batch_hard_loss, corrects, pos_corrects, neg_corrects, total_number_triplets = BatchSoftmaxTripletLoss(batch_fvs, 
																										batch_labels, 
																										batch_pids, tau=tau, 
																										gpu_index=gpu_indexes[0])
				total_pos_corrects += pos_corrects
				total_neg_corrects += neg_corrects
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

				
				if model_name == 'bert' or model_name == 'tweet-bert':
					model_online.eval()
					for param in model_online.parameters():
						param.requires_grad = False
						#print(param.requires_grad)

					model_online.module.bert.encoder.layer[11].training = True
					for param in model_online.module.bert.encoder.layer[11].parameters():
						param.requires_grad = True

				
				elif model_name == 'gpt2':
					model_online.eval()
					for param in model_online.parameters():
						param.requires_grad = False
						#print(param.requires_grad)

					#model_online.module.bert.transformer.layer[5].training = True
					for layer_idx in range(10,12):
						model_online.module.gpt2.h[layer_idx].training = True
						for param in model_online.module.gpt2.h[layer_idx].parameters():
							param.requires_grad = True
				
				else:
					model_online.train()
				

				num_batches_computed += 1
				#if total_number_of_batches >= number_of_iterations:
				#	reach_max_number_iterations = True
				#	break

			total_correct_positive_rate = total_pos_corrects/total_batch_size
			total_correct_negative_rate = total_neg_corrects/total_batch_size
			TTPR = total_corrects/total_batch_size

			iteration_loss = iteration_loss/(batch_idx+1)
			iteration_center = iteration_center/(batch_idx+1)
			iteration_hard = iteration_hard/(batch_idx+1)


			print(colored("Batches computed: %d, Tau value: %.3f" % (num_batches_computed, tau), "cyan"))
			print(colored("Mean Loss: %.7f, Mean Center Loss: %.7f, Mean Hard Triplet Loss: %.7f" % (iteration_loss, 
																									iteration_center, 
																									iteration_hard), "yellow"))
			
			
			print(colored("Percetage of correct positive pairs on triplets: %.2f" % total_correct_positive_rate, "blue", attrs=['blink']))
			print(colored("Percetage of correct negative pairs on triplets: %.2f" % total_correct_negative_rate, "blue", attrs=['blink']))
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
	pos_corrects = 0
	neg_corrects = 0
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
		if true_label == pos_pid:
			pos_corrects += 1
		if true_label != neg_pid:
			neg_corrects += 1
		if (true_label == pos_pid) and (true_label != neg_pid):
			corrects += 1
		total_number_triplets += 1

	loss = batch_loss/S.shape[0]
	return loss, corrects, pos_corrects, neg_corrects, total_number_triplets


def collate_fn(batch):
	return torch.cat(batch, dim=0)

def collate_fn_PK_text(batch):
	return batch

class samplePKTextBatches(Dataset):
    
	def __init__(self, tweets, pseudo_labels, model_name, K=4, num_concatenated_tweets=2, perc=1.0):

		self.tweets = tweets[:,0]
		self.true_authors = tweets[:,1]
		self.tweets_ids = tweets[:,2]

		self.model_name = model_name
		self.pseudo_labels = pseudo_labels
		self.labels_set = np.unique(pseudo_labels)
		self.K = K
		self.num_concatenated_tweets = num_concatenated_tweets

		np.random.shuffle(self.labels_set)
		self.num_of_pseudo_identities_on_epoch = int(round(len(self.labels_set)*perc))
		
	def __getitem__(self, idx):

		pseudo_identity = self.labels_set[idx]

		tweets_identity = self.tweets[self.pseudo_labels == pseudo_identity]
		true_identities = self.true_authors[self.pseudo_labels == pseudo_identity]
		tweets_identity_ids = self.tweets_ids[self.pseudo_labels == pseudo_identity]

		size = min(tweets_identity.shape[0], self.K)
		selected_tweets_idx = np.random.choice(tweets_identity.shape[0], size=size, replace=False)

		selected_tweets = tweets_identity[selected_tweets_idx]
		selected_true_identities = true_identities[selected_tweets_idx] # Only for reference! Not used in pipeline! 
		selected_tweets_ids = tweets_identity_ids[selected_tweets_idx]
		
		ratio_of_masked_samples = np.random.uniform(low=0.1, high=0.2, size=1)[0]

		batch_input_tokens_ids = []
		batch_attention_masks = []

		for tweet_idx in np.arange(len(selected_tweets)):

			tweet_name = selected_tweets[tweet_idx]
			tweet_id = selected_tweets_ids[tweet_idx]

			author_tweets = json.load(open(tweet_name,))

			for tweet in author_tweets:
				if str(tweet['id']) == tweet_id:
					tweet_text = tweet['text']
					break

			if self.model_name == 'bert':
				encoded_dict = tokenizer_bert.encode_plus(
						tweet_text, #.lower(), # Sentence to encode.
						add_special_tokens = True, # Add '[CLS]' and '[SEP]'
						max_length = MAX_LENGTH,           # Pad & truncate all sentences.
						padding = 'max_length',
						truncation=True, 
						return_attention_mask = True, # Construct attn. masks.
						return_tensors = 'pt',     # Return pytorch tensors.
				)

				input_tokens_ids = encoded_dict['input_ids']
				attention_mask = encoded_dict['attention_mask']

				non_pad_tokens = torch.where(input_tokens_ids[0] != tokenizer_bert.pad_token_id)[0]
				num_tokens = non_pad_tokens.shape[0]

				if num_tokens > 5:
					num_masked_tokens = int(num_tokens*ratio_of_masked_samples) 
					selected_tokens_idx_tobe_masked = np.random.choice(non_pad_tokens, size=num_masked_tokens, replace=False)
					input_tokens_ids[0][selected_tokens_idx_tobe_masked] = tokenizer_bert.mask_token_id
					#input_tokens_ids[0][selected_tokens_idx_tobe_masked] = torch.Tensor(np.random.choice(SIZE_BERT, 
					#																					size=num_masked_tokens, 
					#	


			elif self.model_name == 'tweet-bert':
				encoded_dict = tokenizer_tweetBert.encode_plus(
					tweet_text, #.lower(), # Sentence to encode.
					add_special_tokens = True, # Add '[CLS]' and '[SEP]'
					max_length = MAX_LENGTH,           # Pad & truncate all sentences.
					padding = 'max_length',
					truncation=True, 
					return_attention_mask = True, # Construct attn. masks.
					return_tensors = 'pt',     # Return pytorch tensors.
				)

				input_tokens_ids = encoded_dict['input_ids']
				attention_mask = encoded_dict['attention_mask']

				non_pad_tokens = torch.where(input_tokens_ids[0] != tokenizer_tweetBert.pad_token_id)[0]
				num_tokens = non_pad_tokens.shape[0]

				if num_tokens > 5:
					num_masked_tokens = int(num_tokens*ratio_of_masked_samples) 
					selected_tokens_idx_tobe_masked = np.random.choice(non_pad_tokens, size=num_masked_tokens, replace=False)
					input_tokens_ids[0][selected_tokens_idx_tobe_masked] = tokenizer_tweetBert.mask_token_id
					#input_tokens_ids[0][selected_tokens_idx_tobe_masked] = torch.Tensor(np.random.choice(SIZE_BERT, 
					#																					size=num_masked_tokens, 
					#	


			elif self.model_name == 'gpt2':
				encoded_dict = tokenizer_gpt.encode_plus(
						tweet_text, #.lower(), # Sentence to encode.
						add_special_tokens = True, # Add '[CLS]' and '[SEP]'
						max_length = MAX_LENGTH,           # Pad & truncate all sentences.
						padding = 'max_length',
						truncation=True, 
						return_attention_mask = True, # Construct attn. masks.
						return_tensors = 'pt',     # Return pytorch tensors.
				)

				input_tokens_ids = encoded_dict['input_ids']
				attention_mask = encoded_dict['attention_mask']

				non_pad_tokens = torch.where(input_tokens_ids[0] != tokenizer_gpt.pad_token_id)[0]
				num_tokens = non_pad_tokens.shape[0]

				if num_tokens > 5:
					num_masked_tokens = int(num_tokens*ratio_of_masked_samples)
					selected_tokens_idx_tobe_masked = np.random.choice(non_pad_tokens, size=num_masked_tokens, replace=False)
					input_tokens_ids[0][selected_tokens_idx_tobe_masked] = tokenizer_gpt.unk_token_id
					#input_tokens_ids[0][selected_tokens_idx_tobe_masked] = torch.Tensor(np.random.choice(SIZE_GPT2, 
					#																					size=num_masked_tokens,
					#																					replace=False)).long()


			elif self.model_name == 't5':
				encoded_dict = tokenizer_t5.encode_plus(
						tweet_text, #.lower(), # Sentence to encode.
						add_special_tokens = True, # Add '[CLS]' and '[SEP]'
						max_length = MAX_LENGTH,           # Pad & truncate all sentences.
						padding = 'max_length',
						truncation=True, 
						return_attention_mask = True, # Construct attn. masks.
						return_tensors = 'pt',     # Return pytorch tensors.
				)

				input_tokens_ids = encoded_dict['input_ids']
				attention_mask = encoded_dict['attention_mask']

				non_pad_tokens = torch.where(input_tokens_ids[0] != tokenizer_t5.pad_token_id)[0]
				num_tokens = non_pad_tokens.shape[0]

				if num_tokens > 5:
					num_masked_tokens = int(num_tokens*ratio_of_masked_samples)
					selected_tokens_idx_tobe_masked = np.random.choice(non_pad_tokens, size=num_masked_tokens, replace=False)
					input_tokens_ids[0][selected_tokens_idx_tobe_masked] = tokenizer_t5.unk_token_id
					#input_tokens_ids[0][selected_tokens_idx_tobe_masked] = torch.Tensor(np.random.choice(SIZE_T5, 
					#																					size=num_masked_tokens, 
					#																					replace=False)).long()

			elif self.model_name == 'custom':
				tweets_tokenized = tokenize([tweet_text], vocabulary)
				tweets_tokenized_tensor = tweets_tokenized[0].reshape(1, 1, -1)

				tokens_size = tweets_tokenized_tensor.shape[2]

				if tokens_size < MAX_LENGTH:
					pad_size = MAX_LENGTH - tokens_size
					pad_tensor = torch.zeros(1,1,pad_size)
					tweets_tokenized_tensor = torch.cat((tweets_tokenized_tensor, pad_tensor), dim=2).long()
				elif tokens_size > MAX_LENGTH:
					tweets_tokenized_tensor = tweets_tokenized_tensor[:,:,:MAX_LENGTH].long()
				else:
					tweets_tokenized_tensor = tweets_tokenized_tensor.long()

				input_tokens_ids = tweets_tokenized_tensor
				attention_mask = torch.zeros(tweets_tokenized_tensor.shape)


			if len(batch_input_tokens_ids) == 0:
				batch_input_tokens_ids = input_tokens_ids
				batch_attention_masks = attention_mask
			else:
				batch_input_tokens_ids = torch.cat((batch_input_tokens_ids, input_tokens_ids), dim=0)
				batch_attention_masks = torch.cat((batch_attention_masks, attention_mask), dim=0)
	
		batch_labels = torch.ones(batch_input_tokens_ids.shape[0])*pseudo_identity

		augmented_batch_input_tokens_ids = []
		augmented_batch_attention_masks = []

		range_idx = np.arange(size)
		
		# Roll the tweets to create pairs. E.g., [1,2,3,4] combined with [4,1,2,3] -> [(1,4), (2,1), (3,2), (4,3)].
		# We concatenate tweet 1 with 4, 2 with 1, 3 with 2 and 4 with 3. This case holds for num_concatenated_tweets == 2. 
		# If num_concatenated_tweets == 3, then we have [1,2,3,4] concatenated to [4,1,2,3] concatenated to [3,4,1,2] resulting in
		# [(1,4,3), (2,1,4), (3,2,1), (4,3,2)]
		for i in range(self.num_concatenated_tweets):
			ridx = np.roll(range_idx, i)
			augmented_batch_input_tokens_ids.append(batch_input_tokens_ids[ridx])
			augmented_batch_attention_masks.append(batch_attention_masks[ridx])

		augmented_batch_input_tokens_ids = torch.cat(augmented_batch_input_tokens_ids, dim=1)
		augmented_batch_attention_masks = torch.cat(augmented_batch_attention_masks, dim=1)
		
		#print(augmented_batch_input_tokens_ids.shape)
		#return batch_input_tokens_ids, batch_attention_masks, batch_labels, selected_true_identities
		return augmented_batch_input_tokens_ids, augmented_batch_attention_masks, batch_labels, selected_true_identities
	
	def __len__(self):
		return self.num_of_pseudo_identities_on_epoch