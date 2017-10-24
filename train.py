from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np

from dataset import ImageDataset
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere


parser = argparse.ArgumentParser(description='PyTorch sphereface')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--dataset', default='../../dataset/face/casia/casia.zip', type=str)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=256, type=int, help='')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()


def alignment(src_img,src_pts):
    of = 2
    ref_pts = [ [30.2946+of, 51.6963+of],[65.5318+of, 51.5014+of],
        [48.0252+of, 71.7366+of],[33.5493+of, 92.3655+of],[62.7299+of, 92.2041+of] ]
    crop_size = (96+of*2, 112+of*2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def dataset_load(name,filename,pindex,cacheobj,zfile):
    position = filename.rfind('.zip:')
    zipfilename = filename[0:position+4]
    nameinzip = filename[position+5:]

    split = nameinzip.split('\t')
    nameinzip = split[0]
    classid = int(split[1])
    src_pts = []
    for i in range(5):
        src_pts.append([int(split[2*i+2]),int(split[2*i+3])])

    data = np.frombuffer(zfile.read(nameinzip),np.uint8)
    img = cv2.imdecode(data,1)
    img = alignment(img,src_pts)

    if ':train' in name:
        if random.random()>0.5: img = cv2.flip(img,1)
        if random.random()>0.5:
            rx = random.randint(0,2*2)
            ry = random.randint(0,2*2)
            img = img[ry:ry+112,rx:rx+96,:]
        else:
            img = img[2:2+112,2:2+96,:]
    else:
        img = img[2:2+112,2:2+96,:]


    img = img.transpose(2, 0, 1).reshape((1,3,112,96))
    img = ( img - 127.5 ) / 128.0
    label = np.zeros((1,1),np.float32)
    label[0,0] = classid
    return (img,label)


def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()

def save_model(model,filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')



def train(epoch,args):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    ds = ImageDataset(args.dataset,dataset_load,'data/casia_landmark.txt',name=args.net+':train',
        bs=args.bs,shuffle=True,nthread=6,imagesize=128)
    while True:
        img,label = ds.get()
        if img is None: break
        inputs = torch.from_numpy(img).float()
        targets = torch.from_numpy(label[:,0]).long()
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        lossd = loss.data[0]
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        outputs = outputs[0] # 0=cos_theta 1=phi_theta
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        printoneline(dt(),'Te=%d Loss=%.4f | AccT=%.4f%% (%d/%d) %.4f %.2f %d'
            % (epoch,train_loss/(batch_idx+1), 100.0*correct/total, correct, total, 
            lossd, criterion.lamb, criterion.it))
        batch_idx += 1
    print('')


net = getattr(net_sphere,args.net)()
# net.load_state_dict(torch.load('sphere20a_0.pth'))
net.cuda()
criterion = net_sphere.AngleLoss()


print('start: time={}'.format(dt()))
for epoch in range(0, 20):
    if epoch in [0,10,15,18]:
        if epoch!=0: args.lr *= 0.1
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    train(epoch,args)
    save_model(net, '{}_{}.pth'.format(args.net,epoch))

print('finish: time={}\n'.format(dt()))

