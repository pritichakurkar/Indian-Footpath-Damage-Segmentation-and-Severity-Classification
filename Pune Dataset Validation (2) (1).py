#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[34]:


import os
import math
import json
from PIL import Image, ImageOps
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import *
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[15]:


get_ipython().system('pip install Pillow')


# In[35]:


import os
import math
import json
from PIL import Image, ImageOps
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import *
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[8]:


source myenv/bin/activate


# In[9]:


pip install Pillow


# In[36]:


import os
import math
import json
from PIL import Image, ImageOps
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import *
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[18]:


pip install torch torchvision torchaudio


# In[37]:


import os
import math
import json
from PIL import Image, ImageOps
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import *
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[38]:


pip install pandas


# In[39]:


pip install matplotlib


# In[40]:


pip install tqdm


# In[41]:


pip install seaborn


# In[21]:


pip install scikit-learn


# In[42]:


import os
import math
import json
from PIL import Image, ImageOps
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import *
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[43]:


import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import cv2


# In[44]:


pip install albumentations


# In[10]:


import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import cv2


# In[11]:


torch.cuda.is_available()


# In[12]:


seed = 217483647
random.seed(seed)
torch.manual_seed(seed)
cudnn.benchmark = True


# In[47]:


path_train = "PRITIVALIDATED/Datasetnew_Pune/FinalFootpathdamagedataset/train"
path_val = "PRITIVALIDATED/Datasetnew_Pune/FinalFootpathdamagedataset/val"
path_test = "PRITIVALIDATED/Datasetnew_Pune/FinalFootpathdamagedataset/test"
train_data = {
    'images': sorted(glob.glob(path_train + "/*.jpg")),
    'masks': sorted(glob.glob(path_train + "/*.png"))
}
val_data = {
    'images': sorted(glob.glob(path_val + "/*.jpg")),
    'masks': sorted(glob.glob(path_val + "/*.png"))
}
test_data = {
    'images': sorted(glob.glob(path_test + "/*.jpg")),
    'masks': sorted(glob.glob(path_test + "/*.png"))
}


# In[31]:


get_ipython().system('unzip ML_DATASET.zip -d ML_DATASET.zip')


# In[169]:


import zipfile
import os

zip_path = 'ML_DATASET - Copy-20250510T083053Z-1-003.zip'       # Path to the zip file
extract_to = 'priti'    # Folder to extract contents

# Create the extract folder if it doesn't exist
os.makedirs(extract_to, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Extraction complete.")


# In[45]:


import glob


# In[48]:


len(train_data["images"]),len(train_data["masks"])


# In[49]:


len(val_data["images"]),len(val_data["masks"])


# In[50]:


len(test_data["images"]),len(test_data["masks"])


# In[51]:


x = [img[:-4] for img in val_data["masks"]]
l = len(val_data["images"])
absent = []
for i in range(l):
    image = val_data["images"][i]
    if image[:-4] not in x:
        absent.append(i) 

for i in absent[::-1]:
    del val_data["images"][i]


# In[52]:


x = [img[:-4] for img in val_data["images"]]
l = len(val_data["masks"])
absent = []
for i in range(l):
    image = val_data["masks"][i]
    if image[:-4] not in x:
        absent.append(i) 

for i in absent[::-1]:
    del val_data["masks"][i]


# In[53]:


len(val_data["images"]),len(val_data["masks"])


# In[54]:


for img_path, mask_path in zip(train_data['images'], train_data['masks']):
  assert img_path[:-4] == mask_path[:-4]

for img_path, mask_path in zip(val_data['images'], val_data['masks']):
  assert img_path[:-4] == mask_path[:-4]

for img_path, mask_path in zip(test_data['images'], test_data['masks']):
  assert img_path[:-4] == mask_path[:-4]


# In[55]:


df_train = pd.DataFrame(train_data)
df_val = pd.DataFrame(val_data)
df_test = pd.DataFrame(test_data)


# In[56]:


class CrackData(Dataset):
  def __init__(self, df, img_transforms=None, mask_transform=None, aux_transforms=None):
    self.data = df
    self.img_transform = img_transforms
    self.mask_transform = mask_transform
    self.aux_transforms = aux_transforms
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    img = Image.open(self.data['images'].iloc[idx]).convert('RGB')
    mask = Image.open(self.data['masks'].iloc[idx]).convert('L')

    if self.aux_transforms is not None:
      img = self.aux_transforms(img)
    
    seed = np.random.randint(420)
    
    random.seed(seed)
    torch.manual_seed(seed)

    img = transforms.functional.equalize(img)
    image = self.img_transform(img)
    random.seed(seed)
    torch.manual_seed(seed)
    
    mask = self.mask_transform(mask)


    return image, mask


# In[57]:


def visualize(**images):
  """PLot images in one row."""
  n = len(images)
  plt.figure(figsize=(16, 5))
  for i, (name, image) in enumerate(images.items()):
      plt.subplot(1, n, i + 1)
      plt.xticks([])
      plt.yticks([])
      plt.title(' '.join(name.split('_')).title())
      plt.imshow(image)
  plt.show()


# In[58]:


train_tfms = transforms.Compose([
                                 transforms.Resize((320, 320)),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomVerticalFlip(p=0.25),
                                 transforms.ToTensor(),
                                ])

val_tfms = transforms.Compose([transforms.Resize((320, 320)),
                               transforms.ToTensor(),
                              ])

dataset_train = CrackData(df_train, train_tfms, train_tfms, None)
dataset_val = CrackData(df_val, val_tfms, val_tfms)
dataset_test = CrackData(df_test, val_tfms, val_tfms)


# In[59]:


from tqdm import tqdm
train_loader = DataLoader(dataset_train, 16, shuffle=True, pin_memory=torch.cuda.is_available())
valid_loader = DataLoader(dataset_val, 16, shuffle=False, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(dataset_test, 16, shuffle=False, pin_memory=torch.cuda.is_available())
# train_loader = tqdm(train_loader,desc="Training")
# valid_loader = tqdm(valid_loader,desc="Training")


# In[60]:


for i in range(20):
    image, mask = dataset_train[i]
    image = image.permute(1, 2, 0)
    mask.squeeze_(0)
    # visualize(image=image, mask=mask)


# In[61]:


data = next(iter(test_loader))
print(len(data))


# In[62]:


len(test_loader)


# In[63]:


len(train_loader)


# In[64]:


imgs, masks = data
print(imgs.shape, masks.shape)


# In[65]:


for img, mask in zip(imgs, masks):
  image, mask = img, mask
  image = image.permute(1, 2, 0)
  mask.squeeze_(0)
  #visualize(image=image, mask=mask)


# In[66]:


def compute_dice2(pred, gt):
  pred = ((pred) >= .5).float()
  dice_score = (2 * (pred * gt).sum()) / ((pred + gt).sum() + 1e-8)
  
  return dice_score 


# In[67]:


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = (BCE + dice_loss)
        
        return Dice_BCE


# In[68]:


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice_loss


# In[69]:


class log_cosh_dice_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(log_cosh_dice_loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        dice = compute_dice2(inputs, targets).item()
        loss = 1 - dice 
        # log_cosh = torch.log((torch.exp(loss) + torch.exp(-loss)))
        return loss 


# In[70]:


ALPHA = 0.5
BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky


# In[71]:


def get_IoU(outputs, labels):
  EPS = 1e-6
  outputs = (outputs> 0.5).int()
  labels = (labels > 0.5).int()   
  intersection = (outputs & labels).float().sum((1, 2))
  union = (outputs | labels).float().sum((1, 2))

  iou = (intersection + EPS) / (union + EPS)  # We smooth our devision to avoid 0/0
  return iou.mean()


# In[72]:


def accuracy(preds, label):
    preds = (preds > 0.5).int()
    label = (label > 0.5).int()   
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc


# In[73]:


def precision_recall_f1(preds, label):
  epsilon = 1e-7
  y_true = (label > 0.5).int()
  y_pred = (preds > 0.5).int()
  tol_pix = (label >= 0).int()
  tp = (y_true * y_pred).sum().to(torch.float32)
  tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
  fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
  fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
  precision = tp / (tp + fp + epsilon)
  recall = tp / (tp + fn + epsilon)
  f1 = 2* (precision*recall) / (precision + recall + epsilon)
  return precision, recall, f1


# In[74]:


def confusion_mat(preds, label):
  epsilon = 1e-7
  y_true = (label > 0.5).int()
  y_pred = (preds > 0.5).int()
  tol_pix = (label >= 0).int()
  tp = (y_true * y_pred).sum().to(torch.float32)
  tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
  fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
  fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
  return tp, tn, fp, fn


# In[75]:


print(compute_dice2(masks, masks))
print(compute_dice2(masks, torch.zeros(masks.shape)))


# In[76]:


loss = log_cosh_dice_loss()
print(loss(masks, masks), loss(masks, torch.zeros(masks.shape)))


# In[77]:


loss = TverskyLoss()
print(loss(masks, masks), loss(masks, torch.zeros(masks.shape)))


# In[78]:


print(get_IoU(masks, masks), get_IoU(masks, torch.zeros(masks.shape)))


# In[79]:


print(accuracy(masks, masks))
print(accuracy(masks[3], torch.zeros(masks[3].shape)))


# In[80]:


pip install torchcontrib


# In[81]:


from torchcontrib.optim import SWA


# In[82]:


pip install efficientunet


# In[83]:


pip install torchsummary


# In[84]:


from torchvision import models
from torch.cuda.amp import GradScaler, autocast
from time import time
from torchsummary import summary
from efficientunet import *


# In[85]:


pip install tensorflow


# In[75]:


pip install efficientunet


# In[86]:


from torchvision import models
from torch.cuda.amp import GradScaler, autocast
from time import time
from torchsummary import summary
from efficientunet import *


# In[87]:


pip install segmentation_models_pytorch==0.2.0


# In[88]:


import segmentation_models_pytorch as smp
import torch.nn.functional as F
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.encoders._base import EncoderMixin
from typing import Optional, Union, List
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (

    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)


# In[89]:


pip install efficientnet_pytorch


# In[90]:


from efficientnet_pytorch import EfficientNet
class EfficientNetEncoder(EfficientNet, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, model_name, depth=5):

        blocks_args, global_params = get_model_params(model_name, override_params=None)
        super().__init__(blocks_args, global_params)

        self._stage_idxs = stage_idxs
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self._fc

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self._conv_stem, self._bn0, self._swish),
            self._blocks[: self._stage_idxs[0]],
            self._blocks[self._stage_idxs[0] : self._stage_idxs[1]],
            self._blocks[self._stage_idxs[1] : self._stage_idxs[2]],
            self._blocks[self._stage_idxs[2] :],
        ]

    def forward(self, x):
        stages = self.get_stages()

        block_number = 0.0
        drop_connect_rate = self._global_params.drop_connect_rate

        features = []
        for i in range(self._depth + 1):

            # Identity and Sequential stages
            if i < 2:
                x = stages[i](x)

            # Block stages need drop_connect rate
            else:
                for module in stages[i]:
                    drop_connect = drop_connect_rate * block_number / len(self._blocks)
                    block_number += 1.0
                    x = module(x, drop_connect)

            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("_fc.bias", None)
        state_dict.pop("_fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


def _get_pretrained_settings(encoder):
    pretrained_settings = {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": url_map[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
        "advprop": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "url": url_map_advprop[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    }
    return pretrained_settings


# In[91]:


class Convolution(nn.Module):
  def __init__(self, in_channels, out_channels, k_size=3, rate=1):
    super().__init__()
    self.convlayer = nn.Sequential(
        nn.Conv2d(in_channels=int(in_channels), out_channels=int(out_channels), kernel_size=k_size, dilation=rate, padding='same'),
        nn.BatchNorm2d(int(out_channels)),
        nn.ReLU(inplace=True))
  
  def forward(self, x):
    return self.convlayer(x)

class MultiScaleAttention(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.in_channels = in_channels
    self.conv = Convolution(in_channels, in_channels, 3)
    self.conv2 = nn.Sequential(
        Convolution(in_channels, in_channels/4, 3),
        Convolution(in_channels/4, in_channels/4, 1)
        )
    self.conv3 = Convolution(in_channels, in_channels/4, 1)
    self.conv1 = nn.Sequential(
        Convolution(in_channels, in_channels/2, 3),
        Convolution(in_channels/2, in_channels/2, 3, rate=2),
        Convolution(in_channels/2, in_channels/2, 1)
        )
    self.comb_conv = Convolution(in_channels, in_channels, 1)
    self.final = Convolution(2*in_channels, in_channels, 3, 2)
  
  def forward(self, x):
    x = self.conv(x)
    x1 = self.conv1(x)
    x2 = self.conv2(x)
    x3 = self.conv3(x)
    x_comb = torch.cat((x1, x2, x3), dim=1)
    x_n = self.comb_conv(x_comb)
    x_new = torch.cat((x, x_n), dim=1)
    out = self.final(x_new)
    return out


# In[92]:


class DecoderBlock(nn.Module):
   def __init__(
       self,
       in_channels,
       skip_channels,
       out_channels,
       use_batchnorm=True,
       attention_type=None,
   ):
       super().__init__()
       self.conv1 = md.Conv2dReLU(
           in_channels + skip_channels,
           out_channels,
           kernel_size=3,
           #activation=nn.ReLU(inplace=True),
           padding=1,
           #norm_layer=nn.BatchNorm2d if use_batchnorm else None,
           #use_batchnorm=use_batchnorm,
       )
       self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
       self.conv2 = md.Conv2dReLU(
           out_channels,
           out_channels,
           kernel_size=3,
           padding=1,
           #activation=nn.ReLU(inplace=True),
           #norm_layer=nn.BatchNorm2d if use_batchnorm else None,
           #use_batchnorm=use_batchnorm,
       )
       self.attention2 = md.Attention(attention_type, in_channels=out_channels)
       if skip_channels > 0:
         self.multiAttention = MultiScaleAttention(skip_channels)
       else:
         self.multiAttention = nn.Identity()

   def forward(self, x, skip=None, i=None):
       x = F.interpolate(x, scale_factor=2, mode="nearest")
       if skip is not None:
           if i is not None and i==2:
             skip = self.multiAttention(skip)
           x = torch.cat([x, skip], dim=1)
           x = self.attention1(x)
       x = self.conv1(x)
       x = self.conv2(x)
       x = self.attention2(x)
       return x


class SelfAttention(nn.Module):
   def __init__(self, n_channels):
       super().__init__()
       self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
       self.gamma = nn.Parameter(torch.tensor([0.]))

   def _conv(self,n_in,n_out):
       return nn.Conv1d(n_in, n_out, kernel_size=1, stride=1, bias=False)

   def forward(self, x):
       size = x.size()
       x = x.view(*size[:2],-1)
       f,g,h = self.query(x),self.key(x),self.value(x)
       beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
       o = self.gamma * torch.bmm(h, beta) + x
       return o.view(*size).contiguous()


class UnetDecoder(nn.Module):
   def __init__(
       self,
       encoder_channels,
       decoder_channels,
       n_blocks=5,
       use_batchnorm=True,
       attention_type=None,
       center=False,
   ):
       super().__init__()

       if n_blocks != len(decoder_channels):
           raise ValueError(
               "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                   n_blocks, len(decoder_channels)
               )
           )

       # remove first skip with same spatial resolution
       encoder_channels = encoder_channels[1:]
       # reverse channels to start from head of encoder
       encoder_channels = encoder_channels[::-1]

       # computing blocks input and output channels
       head_channels = encoder_channels[0]
       in_channels = [head_channels] + list(decoder_channels[:-1])
       skip_channels = list(encoder_channels[1:]) + [0]
       out_channels = decoder_channels

       self.center = nn.Identity()

       # combine decoder keyword arguments
       kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
       blocks = [
           DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
           for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
       ]
       self.blocks = nn.ModuleList(blocks)

   def forward(self, *features):

       features = features[1:]  # remove first skip with same spatial resolution
       features = features[::-1]  # reverse channels to start from head of encoder

       head = features[0]
       skips = features[1:]

       x = self.center(head)
       for i, decoder_block in enumerate(self.blocks):
           n = len(skips)
           skip = skips[i] if i < len(skips) else None
           x = decoder_block(x, skip, i)

       return x


# In[93]:


class Unet(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnext50_32x4d",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()


# In[94]:


class Ensemble(nn.Module):
  def __init__(self, model1, model2, inChannel=3, numClass=1):
    super(Ensemble, self).__init__()
    self.inChannel = inChannel
    self.classes = numClass
    self.model1 = model1
    self.model2 = model2
    self.convOut = nn.Sequential(
        nn.Conv2d(2, 8, kernel_size=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 1, kernel_size=1))
  
  def forward(self, x):
    x1 = self.model1(x)
    x2 = self.model2(x)
    x3 = torch.cat((x1, x2), axis=1)
    out = self.convOut(x3)
    return out


# In[95]:


class Ensemble2(nn.Module):
  def __init__(self, model1, model2, model3, inChannel=3, numClass=1):
    super(Ensemble, self).__init__()
    self.inChannel = inChannel
    self.classes = numClass
    self.model1 = model1
    self.model2 = model2
    self.convOut = model3
  
  def forward(self, x):
    x1 = self.model1(x)
    x2 = self.model2(x)
    x1 = x*x1
    x2 = x*x2
    x3 = torch.cat((x1, x2), axis=1)
    out = self.convOut(x3)
    return out


# In[98]:


import torch
torch.cuda.empty_cache()


# In[99]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = Unet(encoder_name='efficientnet-b2', classes=1)
model2 = Unet(encoder_name='resnext50_32x4d', classes=1)
model = Ensemble(model1, model2)
#model = Unet(encoder_name='efficientnet-b2')
model = model.to(device)


# In[102]:


import torch
from segmentation_models_pytorch import Unet

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model1 = Unet(encoder_name='efficientnet-b2', classes=1)
model2 = Unet(encoder_name='resnext50_32x4d', classes=1)

model = Ensemble(model1, model2)
model = model.to(device)


# In[100]:


import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# In[101]:


inputs = inputs.to(device)
labels = labels.to(device)


# In[103]:


rand_t = torch.rand((1,3,128,128)).to(device)
out = model(rand_t)
print(out.size())


# In[104]:


class OneCycleLR:
  def __init__(self,
                optimizer,
                num_steps,
                lr_range = (0.1, 1.),
                momentum_range = (0.85, 0.95),
                annihilation_frac = 0.1,
                reduce_factor = 0.01,
                last_step = -1):

    self.optimizer = optimizer

    self.num_steps = num_steps

    self.min_lr, self.max_lr = lr_range[0], lr_range[1]
    assert self.min_lr < self.max_lr

    self.min_momentum, self.max_momentum = momentum_range[0], momentum_range[1]
    assert self.min_momentum < self.max_momentum

    self.num_cycle_steps = int(num_steps * (1. - annihilation_frac))
    self.final_lr = self.min_lr * reduce_factor

    self.last_step = last_step

    if self.last_step == -1:
      self.step()

  def state_dict(self):
    return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)

  def get_lr(self):
    return self.optimizer.param_groups[0]['lr']

  def get_momentum(self):
    return self.optimizer.param_groups[0]['momentum']

  def step(self):
    current_step = self.last_step + 1
    self.last_step = current_step

    if current_step <= self.num_cycle_steps // 2:
      scale = current_step / (self.num_cycle_steps // 2)
      lr = self.min_lr + (self.max_lr - self.min_lr) * scale
      momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * scale
    elif current_step <= self.num_cycle_steps:
      scale = (current_step - self.num_cycle_steps // 2) / (self.num_cycle_steps - self.num_cycle_steps // 2)
      lr = self.max_lr - (self.max_lr - self.min_lr) * scale
      momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * scale
    elif current_step <= self.num_steps:
      scale = (current_step - self.num_cycle_steps) / (self.num_steps - self.num_cycle_steps)
      lr = self.min_lr - (self.min_lr - self.final_lr) * scale
      momentum = None
    else:
      return

    self.optimizer.param_groups[0]['lr'] = lr
    if momentum:
      self.optimizer.param_groups[0]['momentum'] = momentum


# In[105]:


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


# In[106]:


def init_log():
  log = {
    'loss' : AverageMeter(),
    'time' : AverageMeter(),
    'iou' : AverageMeter(),
    'dice' : AverageMeter(),
    'acc' : AverageMeter(),
    'precision' : AverageMeter(),
    'recall' : AverageMeter(),
    'f1' : AverageMeter()
  }
  return log


# In[107]:


def train_step(model, optim, criteria, loader, accumulation_steps, scaler, epoch, max_epochs):
  model.train()
  train_logs = init_log()
  bar = tqdm(loader, dynamic_ncols=True)
  torch.cuda.empty_cache()
  start = time()
  with torch.enable_grad():
    for idx, data in enumerate(bar):
      imgs, masks = data
      imgs, masks = imgs.to(device), masks.to(device)
      
      with autocast():
        output = model(imgs)
        output = output.squeeze(1)
        op_preds = torch.sigmoid(output)
        masks = masks.squeeze(1)
        loss = criteria(op_preds, masks)
        # loss = criteria(op_preds, masks) / accumulation_steps 
      
      batch_size = imgs.size(0) 
      
      scaler.scale(loss).backward()
      
      if ((idx + 1) % accumulation_steps == 0) or (idx + 1 == len(loader)):
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()
    
      train_logs['loss'].update(loss.item(), batch_size) 
      train_logs['time'].update(time() - start)
      train_logs['dice'].update(compute_dice2(op_preds, masks).item(), batch_size)
      train_logs['iou'].update(get_IoU(op_preds, masks).item(), batch_size)
      train_logs['acc'].update(accuracy(op_preds, masks).item(), batch_size)
      p, r, f = precision_recall_f1(op_preds, masks)
      train_logs['precision'].update(p.item(), batch_size)
      train_logs['recall'].update(r.item(), batch_size)
      train_logs['f1'].update(f.item(), batch_size)
      
      bar.set_description(f"Training Epoch: [{epoch}/{max_epochs}] Loss: {train_logs['loss'].avg}"
                          f" Dice: {train_logs['dice'].avg} IoU: {train_logs['iou'].avg}"
                          f" Accuracy: {train_logs['acc'].avg} Precision: {train_logs['precision'].avg}"
                          f" Recall: {train_logs['recall'].avg} F1: {train_logs['f1'].avg}")
  return train_logs


# In[108]:


def val(model, criteria, loader, epoch, epochs, split='Validation'):
  model.eval()
  val_logs = init_log()
  bar = tqdm(loader, dynamic_ncols=True)
  start = time()
  with torch.no_grad():
    for idx, data in enumerate(bar):
      imgs, masks = data
      imgs, masks = imgs.to(device), masks.to(device)
      
      output = model(imgs)
      output = output.squeeze(1)
      op_preds = torch.sigmoid(output)
      masks = masks.squeeze(1)
      loss = criteria(op_preds, masks)

      batch_size = imgs.size(0)
      val_logs['loss'].update(loss.item(), batch_size)
      val_logs['time'].update(time() - start)
      val_logs['dice'].update(compute_dice2(op_preds, masks).item(), batch_size)
      val_logs['iou'].update(get_IoU(op_preds, masks).item(), batch_size)
      val_logs['acc'].update(accuracy(op_preds, masks).item(), batch_size)
      p, r, f = precision_recall_f1(op_preds, masks)
      val_logs['precision'].update(p.item(), batch_size)
      val_logs['recall'].update(r.item(), batch_size)
      val_logs['f1'].update(f.item(), batch_size)

      bar.set_description(f"{split} Epoch: [{epoch}/{epochs}] Loss: {val_logs['loss'].avg}"
                          f" Dice: {val_logs['dice'].avg} IoU: {val_logs['iou'].avg}"
                          f" Accuracy: {val_logs['acc'].avg} Precision: {val_logs['precision'].avg}"
                          f" Recall: {val_logs['recall'].avg} F1: {val_logs['f1'].avg}")
      
  return val_logs


# In[109]:


class CallBacks:
  def __init__(self, best):
    self.best = best
    self.earlyStop = AverageMeter()
  
  def saveBestModel(self, cur, model):
    if cur > self.best:
      self.best = cur
      torch.save(model.state_dict(), './model_best_2.pth'.format(save_path))
      self.earlyStop.reset()
      print("\n Saving Best Model....\n")
    return
  
  def earlyStoping(self, cur, maxVal):
    if cur < self.best:
      self.earlyStop.update(1)

    return self.earlyStop.count > maxVal


# In[110]:



lr = 0.09
base_opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
import torch
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.06)
schedular = OneCycleLR(optimizer, num_steps=50, lr_range=(1e-5, 0.1), annihilation_frac=0.75)
criteria = DiceLoss()
epochs = 50 # change no of epoch 51 run for 24 hrs 
accumulation_steps = 4
best_dice = 0.55
scaler = GradScaler()

cb = CallBacks(best_dice)

results = {"train_loss": [], "train_dice": [], "train_iou": [], 'train_acc': [],
           "train_pre": [], "train_rec": [], "train_f1": [],
           "val_loss": [], "val_dice": [], "val_iou": [], "val_acc": [],
           "val_pre": [], "val_rec": [], "val_f1": []}
           
# save_path = f"/content/drive/MyDrive/BTech Minor/Crack500/{model1.__class__.__name__}_{model1.__class__.__name__}_eff_res_2"
save_path = f"priti"
if not os.path.exists(save_path):
  os.makedirs(save_path)
else:
  model_path = save_path + "/model_best.pth"
  if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))


earlyStopEpoch = 10

try:
  for epoch in range(1, epochs + 1):
    print(epoch)
    train_logs = train_step(model, optimizer, criteria, train_loader, accumulation_steps, scaler, epoch, epochs)
    print("\n")
    val_logs = val(model, criteria, valid_loader, epoch, epochs)
    print("\n")
    schedular.step()
    
    results['train_loss'].append(train_logs['loss'].avg)
    results['train_dice'].append(train_logs['dice'].avg)
    results['train_iou'].append(train_logs['iou'].avg)
    results['train_acc'].append(train_logs['acc'].avg)
    results['train_pre'].append(train_logs['precision'].avg)
    results['train_rec'].append(train_logs['recall'].avg)
    results['train_f1'].append(train_logs['f1'].avg)
    results['val_loss'].append(val_logs['loss'].avg)
    results['val_dice'].append(val_logs['dice'].avg)
    results['val_iou'].append(val_logs['iou'].avg)
    results['val_acc'].append(val_logs['acc'].avg)
    results['val_pre'].append(val_logs['precision'].avg)
    results['val_rec'].append(val_logs['recall'].avg)
    results['val_f1'].append(val_logs['f1'].avg)

    data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
    data_frame.to_csv(f'{save_path}/logs_2.csv', index_label='epoch')

    test_logs = val(model, criteria, test_loader, epoch, epochs, split="Testing")
    print("\n")

    cb.saveBestModel(test_logs['dice'].avg, model)
    cb.earlyStoping(val_logs['dice'].avg, earlyStopEpoch)
  
except KeyboardInterrupt:
  data_frame = pd.DataFrame(data=results, index=range(1, epoch+1))
  data_frame.to_csv(f'{save_path}/logs_2.csv', index_label='epoch')
  val_logs = val(model, criteria, valid_loader, 1, 1)
  test_logs = val(model, criteria, test_loader, 1, 1, split="Testing")
  cb.saveBestModel(test_logs['dice'].avg, model)


# In[113]:


def score(model, criteria, loader, epoch, epochs, split='Validation'):
  model.eval()
  val_logs = init_log()
  # Batch size should be 1
  bar = tqdm(loader, dynamic_ncols=True)
  start = time()
  with torch.no_grad():
    for idx, data in enumerate(bar):
      imgs, masks = data
      imgs, masks = imgs.to(device), masks.to(device)
      
      output = model(imgs)
      output = output.squeeze(1)
      op_preds = torch.sigmoid(output)
      masks = masks.squeeze(1)
      loss = criteria(op_preds, masks)


      
  return val_logs


# In[116]:


save_path = "priti/Priti Unet"
model_path = 'model_best_2.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_best = model
model_best = model_best.to(device)
print("loading best model...")
model_best.load_state_dict(torch.load(model_path, map_location=device))


# In[128]:


save_path = "priti/Priti Unet"
model_path = 'model_best_2.pth'

# Force use of GPU1
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load model to GPU1
model_best = model.to(device)

print("Loading best model on", device, "...")
model_best.load_state_dict(torch.load(model_path, map_location=device))


# In[115]:


import torch
from segmentation_models_pytorch import Unet

# Choose GPU1 if available
device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cpu")

# Define models
model1 = Unet(encoder_name='efficientnet-b2', classes=1)
model2 = Unet(encoder_name='resnext50_32x4d', classes=1)

# Build ensemble
model = Ensemble(model1, model2)

# 🔑 Load checkpoint on CPU first (avoids accidental cuda:0 loading)
state_dict = torch.load("model_best_2.pth", map_location="cpu")
model.load_state_dict(state_dict)

# Now move safely to GPU1 (or CPU if no GPU)
model = model.to(device)

print(f"Model loaded on {device}")


# In[122]:


criteria = TverskyLoss()
valid_logs = val(model_best, criteria, valid_loader, 1, 1)
tests_logs = val(model_best, criteria, test_loader, 1, 1)
save_plots = save_path 
if not os.path.exists(save_plots):
  os.makedirs(save_plots)

class CrackDataTest(Dataset):
  def __init__(self, df, img_transforms=None, mask_transform=None, aux_transforms=None):
    self.data = df
    self.img_transform = img_transforms
    self.mask_transform = mask_transform
    self.aux_transforms = aux_transforms
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    img = Image.open(self.data['images'].iloc[idx]).convert('RGB')
    mask = Image.open(self.data['masks'].iloc[idx]).convert('L')

    if self.aux_transforms is not None:
      img = self.aux_transforms(img)
    
    seed = np.random.randint(420)
    
    random.seed(seed)
    torch.manual_seed(seed)

    img = transforms.functional.equalize(img)
    image = self.img_transform(img)

    random.seed(seed)
    torch.manual_seed(seed)
    
    mask = self.mask_transform(mask)


    return image, mask, self.data['images'].iloc[idx]

dataset_test_2 = CrackDataTest(df_test, val_tfms, val_tfms)
dataset_val_2 = CrackDataTest(df_val, val_tfms, val_tfms)
print(dataset_test_2[0][1].shape)
print(dataset_test_2[0][0].shape)
print(dataset_test_2[0][-1])


# In[126]:


import torch
import torch.nn as nn
from segmentation_models_pytorch import Unet

# ----------------------------
# 1. Device setup
# ----------------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# 2. Define models
# ----------------------------
model1 = Unet(encoder_name='efficientnet-b2', classes=1, encoder_weights="imagenet")
model2 = Unet(encoder_name='resnext50_32x4d', classes=1, encoder_weights="imagenet")

# Example Ensemble wrapper
class Ensemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(Ensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        return (out1 + out2) / 2.0   # simple average

model = Ensemble(model1, model2)
model = model.to(device)

# ----------------------------
# 3. Loss function
# ----------------------------
criterion = nn.BCEWithLogitsLoss()

# ----------------------------
# 4. Validation function
# ----------------------------
def val(model, criterion, dataloader, epoch, fold, device):
    model.eval()
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in dataloader:   # Assuming dataloader returns (image, mask)
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            num_batches += 1

    avg_loss = val_loss / max(1, num_batches)
    print(f"[Validation] Fold {fold} Epoch {epoch} | Loss: {avg_loss:.4f}")
    return {"val_loss": avg_loss}

# ----------------------------
# 5. Example usage
# ----------------------------
# Suppose you already have valid_loader, test_loader
# model_best is your trained model

model_best = model  # just using the ensemble as example
model_best = model_best.to(device)

valid_logs = val(model_best, criterion, valid_loader, epoch=1, fold=1, device=device)
tests_logs = val(model_best, criterion, test_loader, epoch=1, fold=1, device=device)


# In[127]:


def plot_test(model, dataset, save_plots, save_path, split="test"):
  idx = 0
  if not os.path.exists(save_plots):
    os.makedirs(save_plots)

  results_test = {"img": [], "test_loss": [], "test_dice": [], "test_iou": [], "test_acc": [],
           "test_pre": [], "test_rec": [], "test_f1": []}
  model = model.cpu()
  with torch.no_grad():
    bar = tqdm(range(len(dataset)))
    for i in bar:
      images, masks, path = dataset[i]
      images, masks = images.cpu(), masks.cpu()
      images = images.unsqueeze(dim=0)

      mask_pred = model(images)
      prob_msk = mask_pred.cpu()
      masks_2 = (torch.sigmoid(mask_pred.cpu()) >= 0.5).int()

      loss = criteria(masks_2, masks)
      results_test['img'].append(path)
      results_test['test_loss'].append(loss.item())
      results_test['test_dice'].append(compute_dice2(masks_2, masks).item())
      results_test['test_iou'].append(get_IoU(masks_2, masks).item())
      results_test['test_acc'].append(accuracy(masks_2, masks).item())
      p, r, f = precision_recall_f1(masks_2, masks)
      results_test['test_pre'].append(p.item())
      results_test['test_rec'].append(r.item())
      results_test['test_f1'].append(f.item())

      #print(images.shape, masks.shape, masks_2.shape)
      masks *= 255.
      masks_2 = masks_2.squeeze(dim=0)
      masks_2 = masks_2.to(torch.float)
      masks_2 *= 255.
      image = transforms.ToPILImage()(images[0])
      gt = transforms.ToPILImage()(masks.byte().cpu())
      pred = transforms.ToPILImage()(masks_2.byte().cpu())
      
      image = ImageOps.expand(image,border=5,fill='white')
      gt = ImageOps.expand(gt,border=5,fill='white')
      pred = ImageOps.expand(pred,border=5,fill='white')
      
      (img_width, img_height) = image.size
      (gt_width, gt_height) = gt.size
      (pred_width, pred_height) = pred.size

      name = path.split('/')[-1][:-4]
      final_width, final_height = (img_width + gt_width + pred_width), max(img_height, max(gt_height, pred_height))
      result = Image.new('RGB', (final_width, final_height))
      result.paste(im=image, box=(0, 0))
      result.paste(im=gt, box=(img_width, 0))
      result.paste(im=pred, box=(img_width + gt_width, 0))
      result.save(f"{save_plots}/{name}_res.png")
      bar.set_description(f"Saving Test Results")
      idx += 1
  data_frame = pd.DataFrame(data=results_test)
  data_frame.to_csv(f'{save_path}/logs_finished_{split}.csv')


# In[125]:


plot_test(model_best, dataset_test_2, save_plots, save_path, "test")
plot_test(model_best, dataset_val_2, save_plots, save_path, "val")


# In[ ]:


df = pd.read_csv((f'{save_path}/logs_finished_test.csv'))
df_val = pd.read_csv((f'{save_path}/logs_finished_val.csv'))


# In[370]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from a CSV file
df = pd.read_csv((f'{save_path}/logs_finished_test.csv'))

# Optionally inspect first few rows
print(df.head())

# Drop rows with all-zero precision, recall, and F1 (optional, to remove invalid samples)
df_filtered = df[~((df['test_pre'] == 0) & (df['test_rec'] == 0) & (df['test_f1'] == 0))]

# Descriptive statistics
print("Descriptive statistics:\n", df_filtered.describe())

# Boxplot for each metric
metrics = ['test_iou', 'test_acc', 'test_pre', 'test_rec', 'test_f1']
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_filtered[metrics])
plt.title("Boxplot of Evaluation Metrics")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histogram for each metric
for metric in metrics:
    plt.figure()
    sns.histplot(df_filtered[metric], kde=True, bins=20)
    plt.title(f"Distribution of {metric}")
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[373]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv((f'{save_path}/logs_finished_test.csv'))

# Filter invalid rows (optional)
df = df[~((df['test_pre'] == 0) & (df['test_rec'] == 0) & (df['test_f1'] == 0))]

# Metrics to visualize
metrics = ['test_iou', 'test_acc', 'test_pre', 'test_rec', 'test_f1']
metric_labels = {
    'test_iou': 'IoU',
    'test_acc': 'Accuracy',
    'test_pre': 'Precision',
    'test_rec': 'Recall',
    'test_f1': 'F1-Score'
}

# Set the style and color palette
sns.set(style='whitegrid')
palette = sns.color_palette("Set2")

# High-Resolution Boxplot
plt.figure(figsize=(10, 6), dpi=300)
sns


# In[374]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv((f'{save_path}/logs_finished_test.csv'))

# Remove rows where all scores are zero
df = df[~((df['test_pre'] == 0) & (df['test_rec'] == 0) & (df['test_f1'] == 0))]

# Define metrics and labels
metrics = ['test_iou', 'test_acc', 'test_pre', 'test_rec', 'test_f1']
metric_labels = {
    'test_iou': 'IoU',
    'test_acc': 'Accuracy',
    'test_pre': 'Precision',
    'test_rec': 'Recall',
    'test_f1': 'F1-Score'
}

# Set seaborn style and color palette
sns.set(style='whitegrid')
palette = sns.color_palette("Set2")

# ---- Boxplot ----
plt.figure(figsize=(10, 6), dpi=300)
box = sns.boxplot(data=df[metrics], palette=palette)
box.set_xticklabels([metric_labels[m] for m in metrics], fontsize=10)
box.set_title('Model Evaluation Metrics (Boxplot)', fontsize=14)
box.set_ylabel('Score', fontsize=12)
plt.tight_layout()
plt.savefig('boxplot_metrics_highres.png', dpi=300)
plt.show()

# ---- Histograms ----
for i, metric in enumerate(metrics):
    plt.figure(figsize=(6, 4), dpi=300)
    hist = sns.histplot(df[metric], bins=20, kde=True, color=palette[i], edgecolor='black')
    hist.set_title(f'Distribution of {metric_labels[metric]}', fontsize=12)
    hist.set_xlabel(metric_labels[metric], fontsize=10)
    hist.set_ylabel('Frequency', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'hist_{metric}_highres.png', dpi=300)
    plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and melt the dataframe
df = pd.read_csv("model_metrics.csv")

# Drop rows where all values are zero
df = df.loc[:, (df != 0).any(axis=0)]

# Melt the DataFrame to long format
df_long = df.melt(var_name='MetricType', value_name='Score')

# Split MetricType into 'Split' and 'Metric'
df_long['Split'] = df_long['MetricType'].apply(lambda x: x.split('_')[0])
df_long['Metric'] = df_long['MetricType'].apply(lambda x: '_'.join(x.split('_')[1:]))

# Optional: Clean up display names
metric_labels = {
    'iou': 'IoU', 'acc': 'Accuracy', 'pre': 'Precision',
    'rec': 'Recall', 'f1': 'F1-Score'
}
df_long['Metric'] = df_long['Metric'].map(metric_labels)

# Set plot style
sns.set(style='whitegrid')
palette = sns.color_palette("Set2")

# Create boxplot grouped by Metric, hue by Split (train/val/test)
plt.figure(figsize=(12, 6), dpi=300)
ax = sns.boxplot(data=df_long, x='Metric', y='Score', hue='Split', palette=palette)

# Labeling
plt.title('Train vs Validation vs Test Metrics', fontsize=14)
plt.ylabel('Score', fontsize=12)
plt.xlabel('Metric', fontsize=12)
plt.legend(title='Dataset Split', fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.savefig('comparison_boxplot_highres.png', dpi=300)
plt.show()

