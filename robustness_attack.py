
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import pickle
from tqdm import tqdm
#from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50#, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152

from util import extract_features,utils

CHECKPOINT = "backbone_ir50_ms1m_epoch120.pth"
IMG_DIR="/home/zhangao/datasets/lfwtestAligned/"
ID_DIR="/home/zhangao/datasets/lfwidAligned/"
MULTIGPU=False

LOAD_EMBEDDINGS="id_embeddings.pkl"
PIN_MEMORY=False
NUM_WORKERS=0
BATCH_SIZE=8
EMBEDDING_SIZE = 512

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


model = IR_50([112,112])
model.to(device)
model.load_state_dict(torch.load(CHECKPOINT,map_location=device))
model.eval()

# Loading Test Data (Un-normalized)
transform_test = transforms.Compose([transforms.ToTensor(),])
dataset = datasets.ImageFolder(IMG_DIR,transform=transform_test)

loader = torch.utils.data.DataLoader(
    dataset, batch_size = BATCH_SIZE,  pin_memory = PIN_MEMORY,
    num_workers = NUM_WORKERS
)

# Attacking Images batch-wise
def attack(model,  img, label, id_embeddings, eps, attack_type, iters,device,target_index=None):
    adv = img.detach()
    adv.requires_grad = True

    if not target_index == None:
        flag = -1
    else:
        flag = 1
    
    if attack_type == 'fgsm':
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations  
        noise = 0
        
    for j in range(iterations):
        if not target_index == None:
            features = extract_features.l2_normlize(model(utils.normalize(adv.clone())))
            loss = utils.distance_loss(id_embeddings,features,target_index,device)
        loss.backward()
        if attack_type == 'mim':
            adv_mean= torch.mean(torch.abs(adv.grad), dim=1,  keepdim=True)
            adv_mean= torch.mean(torch.abs(adv_mean), dim=2,  keepdim=True)
            adv_mean= torch.mean(torch.abs(adv_mean), dim=3,  keepdim=True)
            adv.grad = adv.grad / adv_mean
            noise = noise + adv.grad
        else:
            noise = adv.grad
        # Optimization step
        adv.data = adv.data + flag * step * noise.sign()
        if attack_type == 'pgd':
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    return adv.detach()

if  not LOAD_EMBEDDINGS==None:
    fr = open(LOAD_EMBEDDINGS,'rb')
    id_embeddings,id_name_to_index,id_index_to_name = pickle.load(fr)

    #carray,id_name_to_index,id_index_to_name = np.load(LOAD_EMBEDDINGS)
else:
    carray,id_name_to_index,id_index_to_name = utils.read_identities(ID_DIR)
    
    id_embeddings = extract_features.generate_id_embeddings(MULTIGPU,device,EMBEDDING_SIZE,BATCH_SIZE,model,carray)
    fw = open('id_embeddings.pkl','wb')
    pickle.dump([id_embeddings,id_name_to_index,id_index_to_name],fw)
    fw.close()

target_index = utils.generate_target_index(dataset.class_to_idx,dataset.samples,id_name_to_index) 


adv_acc = 0
clean_acc = 0
adv_success = 0
eps =16/255 # Epsilon for Adversarial Attack
id_embeddings_t = torch.from_numpy(id_embeddings.T).float().to(device)


for i, (img, label) in tqdm(enumerate(loader)):
    batch_target_index = target_index[i*BATCH_SIZE:i*BATCH_SIZE+len(label.numpy())]
    img = img.to(device)
    
    #print(img.numpy().shape)
    features =extract_features.l2_normlize( model(utils.normalize(img.clone().detach())))
    #print(id_embeddings_t.transpose(1,0).detach().numpy())
    _pred = features.mm(id_embeddings_t).argmax(dim=-1).detach().cpu().numpy()
    pred= np.zeros(BATCH_SIZE,dtype=np.int)
    for j,index in enumerate(_pred):
        pred[j] = dataset.class_to_idx[id_index_to_name[index]]
    clean_acc += np.sum(pred==label.numpy())

    adv= attack(model, img, label,id_embeddings, eps=eps, attack_type= 'pgd', iters= 10,device=device, target_index=batch_target_index)
    features_adv = extract_features.l2_normlize( model(utils.normalize(adv.clone().detach())))
    _pred_adv = features_adv.mm(id_embeddings_t).argmax(dim=-1).detach().cpu().numpy()
    pred_adv= np.zeros(BATCH_SIZE,dtype=np.int)
    for j,index in enumerate(_pred_adv):
        pred_adv[j] = dataset.class_to_idx[id_index_to_name[index]]
    adv_acc += np.sum(pred_adv==label.numpy())
    adv_success += np.sum(_pred_adv==np.array(batch_target_index))

adv_acc = adv_acc/len(dataset)
clean_acc = clean_acc/len(dataset)
adv_success = adv_success/len(dataset)
print('Clean accuracy:{0:.3%}\t Adversarial accuracy:{1:.3%}\t Attack success rate:{2:.3%}'.format(clean_acc , adv_acc, adv_success))



