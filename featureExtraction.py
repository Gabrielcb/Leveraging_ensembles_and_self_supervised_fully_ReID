import torch
import time
import json
import pickle

import torchreid
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, GPT2Tokenizer, T5Tokenizer
from transformers import AutoTokenizer

transform = Compose([Resize((256, 128), interpolation=3), ToTensor(), 
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

MAX_LENGTH = 128

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer_tweetBert = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
tokenizer_gpt = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer_gpt.add_special_tokens({'pad_token': '<|endoftext|>'})
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-small")

'''
vocabulary_filename = '/work/antonio_gabriel/microblog_authorship_attribution/datasets/twitter_56K-authors_130MM-tweets_2019-01_preprocessed/ngrams_vocabulary/ngrams_vocabulary_7000001.pkl'
with open(vocabulary_filename, mode='rb') as fd:
    vocabulary = pickle.load(fd)

VOCAB_SIZE = len(vocabulary)
print(VOCAB_SIZE)
'''


class sample(Dataset):
    
    def __init__(self, Set):
        self.set = Set        
            
    def __getitem__(self, idx):
        
        sample = self.set[idx]
        imgPIL = torchreid.utils.tools.read_image(sample[0])
        img = torch.stack([transform(imgPIL)])
        return img[0]
                 
    def __len__(self):
        return self.set.shape[0]

class textSampler(Dataset):
    
    def __init__(self, Set, model_name):
        self.set = Set  
        self.model_name = model_name      
            
    def __getitem__(self, idx):
        
        sample = self.set[idx]
        author_tweets = json.load(open(sample[0],))

        for tweet in author_tweets:
            if str(tweet['id']) == sample[2]:
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

        elif self.model_name == 't5':
            encoded_dict = tokenizer_t5.encode_plus(
                    tweet_text,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = MAX_LENGTH,           # Pad & truncate all sentences.
                    padding = 'max_length',
                    truncation=True,
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )

        if self.model_name != 'custom':
            input_tokens_ids = encoded_dict['input_ids']
            attention_mask = encoded_dict['attention_mask']
        else:
            input_tokens_ids = tweets_tokenized_tensor
            attention_mask = torch.zeros(tweets_tokenized_tensor.shape)

        return input_tokens_ids, attention_mask
                 
    def __len__(self):
        return self.set.shape[0]


@torch.no_grad()
def getFVs(batch, model, gpu_index):
    batch_gpu = batch.cuda(gpu_index)
    fv = model(batch_gpu)
    fv_cpu = fv.data.cpu()
    return fv_cpu

@torch.no_grad()
def getFeatueMaps(batch, model, gpu_index):
    batch_gpu = batch.cuda(gpu_index)
    fv = model.featuremaps(batch_gpu)
    fv_cpu = fv.data.cpu()
    return fv_cpu


def extractFeatures(subset, model, batch_size, gpu_index=0, eval_mode=True):

    if eval_mode:
        model.eval()
    else:
        model.train()
        
    dataSubset = sample(subset)
    loader = DataLoader(dataSubset, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    start = time.time()
    subset_fvs = []
    for batch_idx, batch in enumerate(loader):

        #fvs = getFVs(batch, model, gpu_index)
        with torch.no_grad():
            batch_gpu = batch.cuda(gpu_index)
            fv = model(batch_gpu)
            fvs = fv.data.cpu()

        if len(subset_fvs) == 0:
            subset_fvs = fvs
        else:
            subset_fvs = torch.cat((subset_fvs, fvs), 0)
            
    end = time.time()
    print("Features extracted in %.2f seconds" % (end-start))

    return subset_fvs


def extractTextFeatures(subset, model, batch_size, model_name, gpu_index=0, eval_mode=True):

    if eval_mode:
        model.eval()
    else:
        model.train()
        
    dataSubset = textSampler(subset, model_name)
    loader = DataLoader(dataSubset, batch_size=batch_size, num_workers=8, pin_memory=True, collate_fn=collate_text)
    
    start = time.time()
    subset_fvs = []
    for batch_idx, batch in enumerate(loader):

        input_token_ids_gpu = batch[0].cuda(gpu_index) 
        attention_masks_gpu = batch[1].cuda(gpu_index)
        
        with torch.no_grad():

            if model_name != 'custom':
                fv = model(input_token_ids_gpu, attention_mask=attention_masks_gpu, token_type_ids=None, return_dict=True)
            else:
                fv = model(input_token_ids_gpu)

            fvs = fv.data.cpu()

        if len(subset_fvs) == 0:
            subset_fvs = fvs
        else:
            subset_fvs = torch.cat((subset_fvs, fvs), 0)
            
    end = time.time()
    print("Features extracted in %.2f seconds" % (end-start))

    return subset_fvs

def extractFeatureMaps(subset, model, batch_size=500):
    
    dataSubset = sample(subset)
    loader = DataLoader(dataSubset, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    start = time.time()
    initialized = False

    for batch_idx, batch in enumerate(loader):

        with torch.no_grad():
            batch_gpu = batch.cuda()
            feature_maps, fvs = model(batch_gpu)

            feature_maps_cpu = feature_maps.data.cpu()
            fvs_cpu = fvs.data.cpu()

        if not initialized:
            featmaps = feature_maps_cpu
            all_fvs = fvs_cpu
            initialized = True
        else:
            featmaps = torch.cat((featmaps, feature_maps_cpu), dim=0)
            all_fvs = torch.cat((all_fvs, fvs_cpu), dim=0)
            
    end = time.time()
    print("Features extracted in %.2f seconds" % (end-start))
    return featmaps, all_fvs

def collate_text(batch):

    input_tokens_ids = []
    attention_masks = []

    for token_id, attention_mask in batch:

        if len(input_tokens_ids) == 0:
            input_tokens_ids = token_id
            attention_masks = attention_mask
        else:
            input_tokens_ids = torch.cat((input_tokens_ids, token_id), dim=0)
            attention_masks = torch.cat((attention_masks, attention_mask), dim=0)

    return input_tokens_ids, attention_masks