import torch
import torchreid
import torchvision
from torchvision.models import resnet50, densenet121, inception_v3
from torch.nn import Module, Dropout, BatchNorm1d, Linear, AdaptiveAvgPool2d, CrossEntropyLoss, Softmax, ReLU, AdaptiveMaxPool2d
from torch.nn import functional as F
from torch import nn

from featureExtraction import extractFeatures, extractTextFeatures

import warnings

try:
    from torchreid.metrics.rank_cylib.rank_cy import evaluate_cy
    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )

def getDCNN(gpu_indexes, model_name):

	if model_name == "resnet50":
		# loading ResNet50
		model_source = resnet50(pretrained=True)
		model_source = ResNet50ReID(model_source)

		model_momentum = resnet50(pretrained=True)
		model_momentum = ResNet50ReID(model_momentum)

		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		model_momentum.load_state_dict(model_source.state_dict())

		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	elif model_name == "osnet":

		# loading OSNet	
		model_source = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
		model_source = OSNETReID(model_source)

		model_momentum = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
		model_momentum = OSNETReID(model_momentum)

		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		model_momentum.load_state_dict(model_source.state_dict())

		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	elif model_name == "densenet121":
		# loading DenseNet121
		model_source = densenet121(pretrained=True)
		model_source = DenseNet121ReID(model_source)

		model_momentum = densenet121(pretrained=True)
		model_momentum = DenseNet121ReID(model_momentum)

		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		model_momentum.load_state_dict(model_source.state_dict())

		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	return model_source, model_momentum



def getEnsembles(gpu_indexes):

	# loading ResNet50
	model_source_resnet50 = resnet50(pretrained=True)
	model_source_resnet50 = ResNet50ReID(model_source_resnet50)

	model_momentum_resnet50 = resnet50(pretrained=True)
	model_momentum_resnet50 = ResNet50ReID(model_momentum_resnet50)

	model_source_resnet50 = nn.DataParallel(model_source_resnet50, device_ids=gpu_indexes)
	model_momentum_resnet50 = nn.DataParallel(model_momentum_resnet50, device_ids=gpu_indexes)

	model_momentum_resnet50.load_state_dict(model_source_resnet50.state_dict())

	model_source_resnet50 = model_source_resnet50.cuda(gpu_indexes[0])
	model_source_resnet50 = model_source_resnet50.eval()

	model_momentum_resnet50 = model_momentum_resnet50.cuda(gpu_indexes[0])
	model_momentum_resnet50 = model_momentum_resnet50.eval()

	# loading OSNet	
	model_source_osnet = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
	model_source_osnet = OSNETReID(model_source_osnet)

	model_momentum_osnet = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
	model_momentum_osnet = OSNETReID(model_momentum_osnet)

	model_source_osnet = nn.DataParallel(model_source_osnet, device_ids=gpu_indexes)
	model_momentum_osnet = nn.DataParallel(model_momentum_osnet, device_ids=gpu_indexes)

	model_momentum_osnet.load_state_dict(model_source_osnet.state_dict())

	model_source_osnet = model_source_osnet.cuda(gpu_indexes[0])
	model_source_osnet = model_source_osnet.eval()

	model_momentum_osnet = model_momentum_osnet.cuda(gpu_indexes[0])
	model_momentum_osnet = model_momentum_osnet.eval()

	# loading DenseNet121
	model_source_densenet121 = densenet121(pretrained=True)
	model_source_densenet121 = DenseNet121ReID(model_source_densenet121)

	model_momentum_densenet121 = densenet121(pretrained=True)
	model_momentum_densenet121 = DenseNet121ReID(model_momentum_densenet121)

	model_source_densenet121 = nn.DataParallel(model_source_densenet121, device_ids=gpu_indexes)
	model_momentum_densenet121 = nn.DataParallel(model_momentum_densenet121, device_ids=gpu_indexes)

	model_momentum_densenet121.load_state_dict(model_source_densenet121.state_dict())

	model_source_densenet121 = model_source_densenet121.cuda(gpu_indexes[0])
	model_source_densenet121 = model_source_densenet121.eval()

	model_momentum_densenet121 = model_momentum_densenet121.cuda(gpu_indexes[0])
	model_momentum_densenet121 = model_momentum_densenet121.eval()

	return model_source_resnet50, model_momentum_resnet50, model_source_osnet, model_momentum_osnet, model_source_densenet121, model_momentum_densenet121
	


## New Model Definition for ResNet50
class ResNet50ReID(Module):
    
	def __init__(self, model_base):
		super(ResNet50ReID, self).__init__()


		self.conv1 = model_base.conv1
		self.bn1 = model_base.bn1
		self.relu = model_base.relu
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
		#x = self.relu(x)
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
		v = v_avg + v_max
		v = v.view(v.size(0), -1)
		output = self.last_bn(v)

		#output = self.fc(v)
		return output



def validate(queries, gallery, model, rerank=False, gpu_index=0):
    model.eval()
    queries_fvs = extractFeatures(queries, model, 500, gpu_index)
    gallery_fvs = extractFeatures(gallery, model, 500, gpu_index)

    queries_fvs = queries_fvs/torch.norm(queries_fvs, dim=1, keepdim=True)
    gallery_fvs = gallery_fvs/torch.norm(gallery_fvs, dim=1, keepdim=True)

    distmat = torchreid.metrics.compute_distance_matrix(queries_fvs, gallery_fvs, metric="euclidean")
    distmat = distmat.numpy()

    if rerank:
        print('Applying person re-ranking ...')
        distmat_qq = torchreid.metrics.compute_distance_matrix(queries_fvs, queries_fvs, metric="euclidean")
        distmat_gg = torchreid.metrics.compute_distance_matrix(gallery_fvs, gallery_fvs, metric="euclidean")
        distmat = torchreid.utils.re_ranking(distmat, distmat_qq, distmat_gg)


    # Compute Ranks
    ranks=[1, 5, 10, 20]
    print('Computing CMC and mAP ...')
    cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[:,1], gallery[:,1], 
                                                queries[:,2], gallery[:,2], use_metric_cuhk03=False)
    print('** Results **')
    print('mAP: {:.2%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))
        
    return cmc[:20], mAP, distmat

def validate_text(queries, gallery, model, model_name, rerank=False, gpu_index=0):

    model.eval()
    queries_fvs = extractTextFeatures(queries, model, 500, model_name, gpu_index)
    gallery_fvs = extractTextFeatures(gallery, model, 500, model_name, gpu_index)

    queries_fvs = queries_fvs/torch.norm(queries_fvs, dim=1, keepdim=True)
    gallery_fvs = gallery_fvs/torch.norm(gallery_fvs, dim=1, keepdim=True)

    distmat = torchreid.metrics.compute_distance_matrix(queries_fvs, gallery_fvs, metric="euclidean")
    distmat = distmat.numpy()

    if rerank:
        print('Applying person re-ranking ...')
        distmat_qq = torchreid.metrics.compute_distance_matrix(queries_fvs, queries_fvs, metric="euclidean")
        distmat_gg = torchreid.metrics.compute_distance_matrix(gallery_fvs, gallery_fvs, metric="euclidean")
        distmat = torchreid.utils.re_ranking(distmat, distmat_qq, distmat_gg)


    # Compute Ranks
    ranks=[1, 5, 10, 20]
    print('Computing CMC and mAP ...')
    cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[:,1], gallery[:,1], 
                                                queries[:,2], gallery[:,2], use_metric_cuhk03=False)
    print('** Results **')
    print('mAP: {:.2%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))
        
    return cmc[:20], mAP, distmat


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

	return cmc, mAP

