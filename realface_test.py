from backbone.model_irse import IR_50
import numpy as np
from tqdm import tqdm
from util.generate_attacks import pgd_attack
import torch.nn as nn
from config import configurations
from PIL import Image
import torch

gallery = "/home/zengyuyuan/Exp/FaceReco/data/deepcamp_data/gallery.txt"
query = "/home/zengyuyuan/Exp/FaceReco/data/deepcamp_data/query.txt"

# gallery = "/mnt/nfs/team38/deecamp_face/gallery.txt"
# query = "/mnt/nfs/team38/deecamp_face/query.txt"
# advtrain_model_root = '/home/team38/yxy/deecamp/face.evoLVe.PyTorch/checkpoint/PGD_attck_advtrain/Backbone_IR_50_Epoch_62_Batch_6386_Time_2019-07-26-08-17_checkpoint.pth'
model_root = '/home/zengyuyuan/Exp/FaceReco/checkpoint/test/Backbone_IR_50_Epoch_19_Batch_969_Time_2019-07-26-18-09_checkpoint.pth'
# model_root = 'backbone_ir50_ms1m_epoch120.pth'

def reader(file):
    with open(file) as f:
        lines = f.readlines()
    data = [i.strip().split('\t')[0] for i in lines]
    labels = [i.strip().split('\t')[1] for i in lines]
    return data, labels

def name2id(gallery_name,query_name):
    id = range(len(gallery_name))
    mapping = dict(zip(gallery_name,id))
    query_labels = [mapping[name] for name in query_name]
    return query_labels


def img2tensor(img):
    img = np.array(img)
    img = img.swapaxes(1, 2).swapaxes(0, 1)
    img = np.reshape(img, [1, 3, 112, 112])
    img = np.array(img, dtype = np.float32)
    img = (img - 127.5) / 128.0
    img = torch.from_numpy(img)
    return img



def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def preprocess_image(filenames):
    img_list = []
    for filename in tqdm(filenames):
        img_align = Image.open(filename)
        img_tensor = img2tensor(img_align)
        img_list.append(img_tensor)
    return img_list

def get_feature(model, filename):
    img_align = Image.open(filename)
    feat = l2_norm(model(img2tensor(img_align).to(DEVICES)))
    return feat.detach().cpu().numpy(), True


def extract_raw_feature(model,query_imgs,batch_size):
    eval_size = len(query_imgs)
    features = []
    idx = 0
    for itr in tqdm(range(int(eval_size / batch_size))):
        # print(query_imgs[idx:idx + batch_size])
        batch = torch.cat(query_imgs[idx:idx + batch_size],0)
        # print(batch.shape)
        imgs = batch.to(DEVICES)
        feat = l2_norm(model(imgs).cpu().data)
        features.append(feat)
        idx = idx+batch_size
    if idx < eval_size:
        batch = torch.cat(query_imgs[idx:eval_size],dim=0)
        imgs = batch.to(DEVICES)
        feat = l2_norm(model(imgs).cpu().data)
        features.append(feat)
    features = torch.cat(features,dim=0)
    return features


def extract_attack_feature(model,gallery_features,query_imgs,adversary,batch_size):
    eval_size = len(query_imgs)
    features = []
    idx = 0
    for itr in tqdm(range(int(eval_size / batch_size))):
        # print(query_imgs[idx:idx + batch_size])
        batch = torch.cat(query_imgs[idx:idx + batch_size],0)
        # print(batch.shape)
        raw_feat = l2_norm(model(batch.to(DEVICES)).cpu().data)
        pred_labels = evaluation(gallery_features,raw_feat)
        imgs_attack = adversary.perturb(batch.to(DEVICES),pred_labels)
        feat = l2_norm(model(imgs_attack).cpu().data)
        features.append(feat)
        idx = idx+batch_size
    if idx < eval_size:
        batch = torch.cat(query_imgs[idx:eval_size],dim=0)
        raw_feat = l2_norm(model(batch.to(DEVICES)).cpu().data)
        pred_labels = evaluation(gallery_features, raw_feat)
        imgs_attack = adversary.perturb(batch.to(DEVICES), pred_labels)
        feat = l2_norm(model(imgs_attack).cpu().data)
        features.append(feat)
    features = torch.cat(features,dim=0)
    return features


def evaluation(gallery_features,query_features):
    gallery_features = torch.Tensor(gallery_features)
    all_dists = torch.matmul(query_features,torch.transpose(gallery_features,0,1))
    max_indices = all_dists.argmax(1)
    return max_indices

def cal_acc(pred_labels,query_ids):
    # right = 0
    # wrong = 0
    # for j in range(len(query_filenames)):
    #     print("query #{}, dist: {:.4}, true label: {}, predicted label: {}".format(j, all_dists[j][max_indices[j]],
    #                                                                                query_labels[j],
    #                                                                                gallery_labels[max_indices[j]]))
    #     if query_labels[j] == gallery_labels[max_indices[j]]:
    #         right += 1
    #     else:
    #         wrong += 1
    query_ids = torch.Tensor(query_ids).long()
    right = torch.sum((pred_labels == query_ids)).float()
    print('right samples:',right.item())
    accu = right / float(len(query_ids))
    return accu


def distance_loss(inputs,y):
    global gallery_features
    target_embeddings = gallery_features[y].float().to(DEVICES)
    # print(target_embeddings.numpy().shape,batch_embeddings.detach().numpy().shape)
    loss = -torch.sum(torch.mul(inputs,target_embeddings))
    return loss



if __name__ == "__main__":
    BATCH_SIZE = 64
    DEVICES = 4

    model = IR_50([112,112])
    model.load_state_dict(torch.load(model_root,map_location='cpu'))
    model.to(DEVICES)
    model.eval()

    query_filenames, query_labels = reader(query)
    gallery_filenames, gallery_labels = reader(gallery)
    query_labels_ids = name2id(gallery_labels,query_labels)

    print('Extracting gallery feature...')
    # gallery_features = extract(model, gallery_filenames)
    gallery_imgs = preprocess_image(gallery_filenames)
    gallery_features = extract_raw_feature(model, gallery_imgs,BATCH_SIZE)
    print(gallery_features.shape)

    print('Preprocessing query feature...')
    query_imgs = preprocess_image(query_filenames)
    # define attack
    print('Extracting query feature...')
    adversary = pgd_attack(model, loss_fn=distance_loss,
                           eps=0.03,nb_iter=20,eps_iter=0.01,rand_init=True,targeted=False)
    query_features = extract_raw_feature(model, query_imgs,BATCH_SIZE)
    # query_features = extract_attack_feature(model,gallery_features,query_imgs,adversary,BATCH_SIZE)
    # print(query_features.shape)

    pred_labels = evaluation(gallery_features,query_features)
    accu = cal_acc(pred_labels,query_labels_ids)
    print("Accuracy: {:.4}".format(accu))
