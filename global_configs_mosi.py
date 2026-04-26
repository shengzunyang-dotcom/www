import os
import torch

# system
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["WANDB_PROGRAM"] = "multimodal_driver.py"
#os.environ["WANDB_API_KEY"] = "local-084aec653f0d0874c7aa2c8d8b40f5448c28ec12"
os.environ["WANDB_MODE"] = "online"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = torch.device("cuda:0")
# parameters
EPOCHS = 20
BATCH_SIZE = 10
LEARNING_RATE = 4e-5#1e-5
TEXT_DIM = 768
VISUAL_DIM = 512
AUDIO_DIM = 768
LABEL_DIM = 3
EMO_LABEL_DIM = 7
TEMPERATURE = 0.07
PADDING_LENGTH = 77

ALPHA = 100
BETA = 0.1

K = 1
# dataset
PREFIX = 'mosi'
DATASETS = 'mosi_clip.pkl'
# DATASETS = 'mosi_clip_large.pkl'
# missing modality



TRAIN_MISSING_MODALITY = 'visual'
TEST_MISSING_MODALITY = 'visual'

#path

# PRETRAIN_PATH = '/home/zhaoxianbing/CMU/pretrain/clip-vit-base-patch32'
# PRETRAIN_PATH = '/home/zhaoxianbing/CMU/pretrain/bert-base-uncased'
# PRETRAIN_PATH = '/home/zhaoxianbing/CMU/pretrain/clip-vit-large-patch14'
# PRETRAIN_PATH = '/home/zhaoxianbing/CMU/pretrain/clip-vit-large-patch14-336'
BERT_PRETRAIN_PATH = '/data/zhaoxianbing/CMU/pretrain/bert-base-uncased'
ELECTRA_PRETRAIN_PATH = '/data/zhaoxianbing/CMU/pretrain/google/electra-base-discriminator'
# T5_PRETRAIN_PATH = '/data/zhaoxianbing/CMU/pretrain/flan-t5-xxl/flan_t5_xxl'
T5_PRETRAIN_PATH = '/data/zhaoxianbing/CMU/pretrain/flan-t5-xxl'
# PATH = '/disk3/CMU/llm-features'
PATH = '/data/zhaoxianbing/cmu-features'

#meld
# TRAIN_DOMAIN = 'meld/meld_clip_senti_emo.pkl'
# TEST_DOMAIN1 = 'mosei/mosei_clip_large.pkl'
# TEST_DOMAIN2 = 'mosi/mosi_clip_large.pkl'

#mosei
# TRAIN_DOMAIN = 'mosei/mosei_clip_large.pkl'
# TEST_DOMAIN1 = 'meld/meld_clip_senti_emo.pkl'
# TEST_DOMAIN2 = 'mosi/mosi_clip_large.pkl'

# mosi
# TRAIN_DOMAIN = 'mosi/mosi_clip_large.pkl' 
# TEST_DOMAIN1 = 'meld/meld_clip_senti_emo.pkl'
# TEST_DOMAIN2 = 'mosei/mosei_clip_large.pkl'
# PRETRAIN_PATH = '/home/zhaoxianbing/CMU/pretrain/clip-vit-large-patch14'


# 1 
# TRAIN_DOMAIN = 'mosi/mosi_vgg.pkl' 
# TEST_DOMAIN1 = 'meld/meld_vgg.pkl'
# TEST_DOMAIN2 = 'mosei/mosei_vgg.pkl'


# 2
# TRAIN_DOMAIN = 'meld/meld_vgg.pkl'
# TEST_DOMAIN1 = 'mosi/mosi_vgg.pkl' 
# TEST_DOMAIN2 = 'mosei/mosei_vgg.pkl'

# 3
# TRAIN_DOMAIN = 'mosei/mosei_vgg.pkl'
# TEST_DOMAIN1 = 'mosi/mosi_vgg.pkl' 
# TEST_DOMAIN2 = 'meld/meld_vgg.pkl'


# mosei mosi meld
# mosi mosei meld
# meld mosei mosi