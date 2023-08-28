import torch
import torchreid

from torchvision.models import resnet50, densenet121, inception_v3
from torch.nn import Module, BatchNorm1d, AdaptiveAvgPool2d, AdaptiveMaxPool2d
from torch.nn import functional as F

from torch import nn

from featureExtraction import extractFeatures
from datasetUtils import load_dataset

import argparse
import os
import time


def main(gpu_ids, resnet_path, osnet_path, densenet_path, target):

	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
	os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

	print(torch.cuda.device_count())

	model_resnet50 = resnet50(pretrained=True)
	model_resnet50 = ResNet50ReID(model_resnet50)
	model_resnet50 = nn.DataParallel(model_resnet50)
	model_resnet50.load_state_dict(torch.load(resnet_path, map_location='cuda:0'))
	model_resnet50 = model_resnet50.cuda()
	model_resnet50 = model_resnet50.eval()

	model_osnet = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
	model_osnet = OSNETReID(model_osnet)
	model_osnet = nn.DataParallel(model_osnet)
	model_osnet.load_state_dict(torch.load(osnet_path, map_location='cuda:0'))
	model_osnet = model_osnet.cuda()
	model_osnet = model_osnet.eval()

	model_densenet121 = densenet121(pretrained=True)
	model_densenet121 = DenseNet121ReID(model_densenet121)
	model_densenet121 = nn.DataParallel(model_densenet121)
	model_densenet121.load_state_dict(torch.load(densenet_path, map_location='cuda:0'))
	model_densenet121 = model_densenet121.cuda()
	model_densenet121 = model_densenet121.eval()

	#### Load target dataset ####
	train_images_target, gallery_images_target, queries_images_target = load_dataset(target)

	print("Training Size:", train_images_target.shape)
	print("Gallery Size:", gallery_images_target.shape)
	print("Query Size:", queries_images_target.shape)
	
	t0 = time.time()

	print("Validating ResNet50 on %s ..." % target)
	distmat_resnet = validate(queries_images_target, gallery_images_target, model_resnet50)

	t1 = time.time()
	dt_resnet = (t1 - t0)/queries_images_target.shape[0]
	print("Average time (ms) per query for ResNet50: %.2f" % (dt_resnet*1000))

	print("Validating OSNet on %s ..." % target)
	distmat_osnet = validate(queries_images_target, gallery_images_target, model_osnet)
	


	t2 = time.time()
	dt_osnet = (t2 - t1)/queries_images_target.shape[0]
	print("Average time (ms) per query for OSNet: %.2f" % (dt_osnet*1000))

	print("Validating DenseNet121 on %s ..." % target)
	distmat_densenet121 = validate(queries_images_target, gallery_images_target, model_densenet121)
	

	t3 = time.time()
	dt_densenet = (t3 - t2)/queries_images_target.shape[0]
	print("Average time (ms) per query for DenseNet121: %.2f" % (dt_densenet*1000))

	distamat_ensemble = (distmat_resnet + distmat_osnet + distmat_densenet121)/3
	calculateMetrics(distamat_ensemble, queries_images_target, gallery_images_target)

	t4 = time.time()
	dt_ensemble = (t3 - t4)/queries_images_target.shape[0]
	print("Average time (ms) per query for Ensemble: %.2f" % (dt_ensemble*1000))

	total_time_per_query = dt_resnet + dt_osnet + dt_densenet + max(0, dt_ensemble)
	print("Total average (ms) time per query: %.2f" % (total_time_per_query*1000))




def calculateMetrics(distmat, queries, gallery):

	#compute Ranks
	ranks = [1,5,10]
	print('Computing CMC and mAP ...')
	cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[:,1], gallery[:,1], 
														queries[:,2], gallery[:,2], use_metric_cuhk03=False)

	print('** Results **')
	print('mAP: {:.2%}'.format(mAP))
	print('Ranks:')
	for r in ranks:
		print('Rank-{:<3}: {:.2%}'.format(r, cmc[r-1]))

	
def validate(queries, gallery, model, rerank=False, gpu_index=0):
	model.eval()
	queries_fvs = extractFeatures(queries, model, 500, gpu_index)
	gallery_fvs = extractFeatures(gallery, model, 500, gpu_index)

	queries_fvs = queries_fvs/torch.norm(queries_fvs, dim=1, keepdim=True)
	gallery_fvs = gallery_fvs/torch.norm(gallery_fvs, dim=1, keepdim=True)

	distmat = torchreid.metrics.compute_distance_matrix(queries_fvs, gallery_fvs, metric="euclidean")
	distmat = distmat.numpy()

	if rerank:
		print("Applying Re-Ranking...")
		distmat_qq = torchreid.metrics.compute_distance_matrix(queries_fvs, queries_fvs, metric="euclidean")
		distmat_gg = torchreid.metrics.compute_distance_matrix(gallery_fvs, gallery_fvs, metric="euclidean")
		distmat = torchreid.utils.re_ranking(distmat, distmat_qq, distmat_gg)

	# Compute Ranks
	ranks = [1, 5, 10]
	print("Computing CMC and mAP ...")
	cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[:,1], gallery[:,1], 
														queries[:,2], gallery[:,2], use_metric_cuhk03=False)

	print('**Results**')
	print('mAP: {:.2%}'.format(mAP))
	print('Ranks:')
	for r in ranks:
		print('Rank-{:<3}: {:.2%}'.format(r, cmc[r-1]))

	return distmat

## New Model Definition for ResNet50
class ResNet50ReID(Module):
    
	def __init__(self, model_base):
		super(ResNet50ReID, self).__init__()


		self.conv1 = model_base.conv1
		self.bn1 = model_base.bn1
		self.maxpool = model_base.maxpool
		self.layer1 = model_base.layer1
		self.layer2 = model_base.layer2
		self.layer3 = model_base.layer3
		self.layer4 = model_base.layer4

		self.layer4[0].conv2.stride = (1,1)
		self.layer4[0].downsample[0].stride = (1,1)

		self.global_avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
		self.global_maxpool = AdaptiveMaxPool2d(output_size=(1, 1))
		self.last_bn = BatchNorm1d(2048)
		

	def forward(self, x):
		
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x_avg = self.global_avgpool(x)
		x_max = self.global_maxpool(x)
		x = x_avg + x_max
		x = x.view(x.size(0), -1)

		output = self.last_bn(x)
		return output 

## New Model Definition for DenseNet121
class DenseNet121ReID(Module):
    
	def __init__(self, model_base):
		super(DenseNet121ReID, self).__init__()

		self.model_base = model_base.features
		self.gap = AdaptiveAvgPool2d(1)
		self.gmp = AdaptiveMaxPool2d(output_size=(1, 1))
		self.last_bn = BatchNorm1d(2048)
			
		
	def forward(self, x):
		
		x = self.model_base(x)
		x = F.relu(x, inplace=True)

		x_avg = self.gap(x)
		x_max = self.gmp(x)
		x = x_avg + x_max
		x = torch.cat([x,x], dim=1)

		x = x.view(x.size(0), -1)
		output = self.last_bn(x)
		
		return output 

## New Definition for OSNET
class OSNETReID(Module):
    
	def __init__(self, model_base):
		super(OSNETReID, self).__init__()

		self.conv1 = model_base.conv1
		self.maxpool = model_base.maxpool
		self.conv2 = model_base.conv2
		self.conv3 = model_base.conv3
		self.conv4 = model_base.conv4
		self.conv5 = model_base.conv5
		self.avgpool = model_base.global_avgpool
		self.maxpool02 = AdaptiveMaxPool2d(output_size=(1, 1))
		#self.fc = model_base.fc
		self.last_bn = BatchNorm1d(512)

	def forward(self, x):
		
		x = self.conv1(x)
		x = self.maxpool(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)

		v_avg = self.avgpool(x)
		v_max = self.maxpool02(x)
		v = v_max + v_avg
		v = v.view(v.size(0), -1)
		output = self.last_bn(v)

		#output = self.fc(v)
		return output

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Define Ensemble Parameters')
	parser.add_argument('--gpu_ids', type=str, default='7', help='GPU IDs')
	parser.add_argument('--resnet_path', type=str, help='Path to the resnet50 weights')
	parser.add_argument('--osnet_path', type=str, help='Path to the inceptionV3 weights')
	parser.add_argument('--densenet_path', type=str, help='Path to the densenet121 weights')
	parser.add_argument('--target', type=str, help='Target dataset')

	args = parser.parse_args()
	gpu_ids = args.gpu_ids
	resnet_path = args.resnet_path
	osnet_path = args.osnet_path
	densenet_path = args.densenet_path
	target = args.target

	main(gpu_ids, resnet_path, osnet_path, densenet_path, target)