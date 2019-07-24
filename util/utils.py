import os
import torch
from PIL import Image
import numpy as np
import random

# Mean and Standard Deiation of the Dataset
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t
def un_normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] * std[0]) + mean[0]
    t[:, 1, :, :] = (t[:, 1, :, :] * std[1]) + mean[1]
    t[:, 2, :, :] = (t[:, 2, :, :] * std[2]) + mean[2]

    return t


def read_identities(root):
    names = os.listdir(root)
    if '.DS_Store' in names:
        names.remove('.DS_Store')
    name_to_index = dict()
    index_to_name = dict()
    N = len(names)
    carray = np.zeros((N,3,112,112),dtype=np.float32)
    for i,name in enumerate(names):
        name_dir = os.path.join(root,name)
        file = os.listdir(name_dir)[0]
        with open(os.path.join(name_dir,file),'rb') as f:
            carray[i,:,:,:]= np.array(Image.open(f).convert('RGB')).transpose(2,0,1)/255.0
        name_to_index[name]=i
        index_to_name[i]=name
    return carray,name_to_index,index_to_name

def generate_target_index(test_class_to_idx,test_samples,id_class_to_idx):
    ids = list(id_class_to_idx.keys())
    test_idx_to_class = {v: k for k, v in test_class_to_idx.items()}
    target_index = []
    for file,label in test_samples:
        name = test_idx_to_class[label]
        ids.remove(name)
        target_index.append(id_class_to_idx[random.choice(ids)])
        ids.append(name)
    return target_index


def distance_loss(id_embeddings,batch_embeddings,batch_targets,device):
    
    target_embeddings = torch.from_numpy(id_embeddings[batch_targets]).float().to(device)
    #print(target_embeddings.numpy().shape,batch_embeddings.detach().numpy().shape)
    loss = -torch.sum(torch.mul(target_embeddings,batch_embeddings))
    return loss