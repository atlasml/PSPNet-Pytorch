dependencies = ['scipy', 'torch', 'torchvision']

from pspnet import pspnet as PSPNet 
from sotabench.semantic_segmentation import cityscapes

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.transforms import FlipChannels, MaskToTensor

def benchmark():

    mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])
    val_input_transform = transforms.Compose([
        FlipChannels(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul_(255)),
        transforms.Normalize(*mean_std)
    ])
     
    target_transform = transforms.Compose([MaskToTensor()])

    pspnet = PSPNet(n_classes = 19)
    pspnet.load_pretrained_model(model_path = '')

    cityscapes.benchmark(
        model=pspnet,  
        input_transform = val_input_transform,
        target_transform = target_transform,
        paper_model_name='EfficientNet',
        paper_arxiv_id='1802.02611',
        paper_pwc_id='encoder-decoder-with-atrous-separable',
        batch_size=32,
        num_gpu=1
        )

benchmark()
