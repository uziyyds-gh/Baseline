import os
import argparse
import csv
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchattacks
from torch.utils import data
from torchvision import datasets
import torchvision
import numpy as np
import torch
import torch.nn.functional as F #使用functional中的ReLu激活函数
import torch.nn as nn
import time
import copy

def parse_args():

  parser = argparse.ArgumentParser()
  temp = os.path.abspath(os.curdir)
  temp =  os.path.join(temp,'dataset\\images')
  parser.add_argument("--dataset_dir", default=temp)

  temp = os.path.abspath(os.curdir)
  temp = os.path.join(temp, 'dataset\\dev_dataset.csv')
  parser.add_argument("--dataset_csv_dir", default=temp)



  temp = os.path.abspath(os.curdir)
  temp = os.path.join(temp, 'outputs','fgsm')
  parser.add_argument("--outputs_fgsm_dir", default=temp)
  return parser.parse_args()

def load_model(model_ft,filename):
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if train_on_gpu else 'cpu')
    # 先加载好模型   模型=结构+参数

    checkpoint = torch.load(filename)
    model_ft.load_state_dict(checkpoint['state_dict'])

    model = model_ft.to(device)

    return model
class DatasetMetadata(object):

  """Helper class which loads and stores dataset metadata."""

  def __init__(self, dataset_csv_dir, dataset_dir):
    """Initializes instance of DatasetMetadata."""
    self.id_label = {}
    self.id_image = {}

    with open(dataset_csv_dir) as f:
      reader = csv.reader(f)
      header_row = next(reader)
      try:
        row_idx_image_id = header_row.index('ImageId')
        row_idx_true_label = header_row.index('TrueLabel')

      except ValueError:
        raise IOError('Invalid format of dataset metadata.')
      for row in reader:
        if len(row) < len(header_row):
          # skip partial or empty lines
          continue
        try:
          image_id = row[row_idx_image_id]
          self.id_label[image_id] = int(row[row_idx_true_label])

        except (IndexError, ValueError):
          raise IOError('Invalid format of dataset metadata')

    # 遍历当前路径下所有文件
    for fname in os.listdir(dataset_dir):
      if not (fname.endswith('.png') or fname.endswith('.jpg')):
        continue
      image_id = fname[:-4] # 把后缀去掉
      real_url = os.path.join(dataset_dir, fname)
      img = plt.imread(real_url)  # 读取数据
      self.id_image[image_id] = img


  def get_label(self, image_id):
    return self.id_label[image_id]


  def get_image(self, image_id):
    return self.id_image[image_id]



train_on_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if train_on_gpu else 'cpu')
args = parse_args()

# 加载替身模型
vgg11 = torchvision.models.vgg11(pretrained=True)


# 加载数据
DataSet = DatasetMetadata(args.dataset_csv_dir, args.dataset_dir)

# 进行攻击
fgsm = torchattacks.FGSM(vgg11,eps=0.1)



for fname in os.listdir(args.dataset_dir):
    if not (fname.endswith('.png') or fname.endswith('.jpg')):
        continue
    image_id = fname[:-4]  # 把后缀去掉
    label = DataSet.get_label(image_id)
    mylabel = torch.tensor(label)
    mylabel = torch.reshape(mylabel,[1])





    image = DataSet.get_image(image_id)
    # 处理成tensor
    mytransforms = transforms.Compose(
        [
         # 转化成张量,#归一化[0,1]（是将数据除以255），
         # transforms.ToTensor（）会把HWC会变成C *H *W（拓展：格式为(h,w,c)，像素顺序为RGB）
         transforms.ToTensor()

         ])

    image = mytransforms(image)
    myimage = torch.reshape(image, [1, 3, 299, 299])




    myimage = fgsm(myimage, mylabel)

    adverarial = myimage
    adverarial = adverarial[0,:,:,:]
    adverarial = torch.permute(adverarial,[1,2,0])
    adverarial = adverarial.numpy()
    path =args.outputs_fgsm_dir
    path = os.path.join(path,str(image_id)+".png")
    plt.imsave(path,adverarial)
