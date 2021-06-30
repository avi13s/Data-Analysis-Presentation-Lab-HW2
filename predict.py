import os
import pandas as pd
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import argparse

parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

DIR_Test = args.input_folder
DIR_pkl = 'checkpoint_model2_34000.pth'


def check_validity(xmin,xmax,ymin,ymax):
  if xmin == xmax == ymin == ymax:
    return False
  if np.min([xmin, xmax, ymin, ymax, ymax-ymin, xmax-xmin])<=0:
    return False
  return True


def path_to_annot_df(path):
  annotation_dict = {}
  idx = 0
  for filename in os.listdir(path):
    split_name = filename.split(',')
    xmin = np.int(split_name[0].split('[')[1])
    xmax = xmin + np.int(split_name[2])
    ymin = np.int(split_name[1])
    ymax = ymin + np.int(split_name[3].split(']')[0])
    label = split_name[3].split('__')[1][:-4]
    if check_validity(xmin,xmax,ymin,ymax):
      annotation_dict[str(idx)] = [filename,xmin,xmax,ymin,ymax,label]
      idx += 1

  df = pd.DataFrame.from_dict(annotation_dict, orient='index',
                       columns=['file_name','x1', 'x2', 'y1', 'y2','label'])
  return df


def get_transform():
  return T.Compose([T.ToTensor()])

class FaceMaskDetectionDataset(Dataset):
    
    def __init__(self, dataframe, image_dir, mode = 'train', transforms = None):
        
        super().__init__()
        
        self.image_names = dataframe["file_name"].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.mode = mode
        
    def __getitem__(self, index: int):
        
        #Retrive Image name and its records (x1, y1, x2, y2, classname) from df
        image_name = self.image_names[index]
        records = self.df[self.df["file_name"] == image_name]
        
        #Loading Image
        image = cv2.imread(self.image_dir + image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        if self.mode == 'train':
            
            #Get bounding box co-ordinates for each box
            boxes = records[['x1', 'y1', 'x2', 'y2']].values

            #Getting labels for each box
            labels = [int(records['label'][0] == 'True')]

            #Converting boxes & labels into torch tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            #Creating target
            target = {}
            target['boxes'] = boxes
            target['labels'] = labels

            #Transforms
            if self.transforms:
                image = self.transforms(image)


            return image, target, image_name
        
        elif self.mode == 'test':

            if self.transforms:
                image = self.transforms(image)

            return image, image_name
    
    def __len__(self):
        return len(self.image_names)

test_df = path_to_annot_df(DIR_Test)

def collate_fn(batch):
    return tuple(zip(*batch))

#Test Dataset
test_dataset = FaceMaskDetectionDataset(test_df, DIR_Test, mode = 'test', transforms = get_transform())

#Test data loader
test_data_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    drop_last=False,
    collate_fn=collate_fn
)

#Loading cuda device and initializing the model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()

#Faster - RCNN Model - not pretrained
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 2

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#Retriving all trainable parameters from model (for optimizer)
params = [p for p in model.parameters() if p.requires_grad]

#Defininig Optimizer

#optimizer = torch.optim.Adam(params, lr = 0.0001)
optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9)

#LR
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# If have pretrained (own pretrained)
checkpoint = torch.load(DIR_pkl)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.to(device)
model.eval()

submission = pd.DataFrame(columns = ["filename", "x", "y", "w", "h", "proper_mask"])

for images, image_names in test_data_loader:
    #Forward
    images = list(image.to(device) for image in images)
    output = model(images)
    
    #Converting tensors to array
    boxes = output[0]['boxes'].data.cpu().numpy()
    scores = output[0]['scores'].data.cpu().numpy()

    # If no box found - we'll give a default prediction of False and a box in the middle
    if len(boxes)<1:
      row = {"filename" : image_names[0], "x" : 0.4*images[0].size()[1],  "y" : 0.4*images[0].size()[2], "w" : 0.2*images[0].size()[1], "h" : 0.2*images[0].size()[2], "proper_mask" : 'False'}
      submission = submission.append(row, ignore_index = True)
      continue
  
    #Bboxes, classname & image name
    x1 = boxes[0][0]
    y1 = boxes[0][1]
    x2 = boxes[0][2]
    y2 = boxes[0][3]
    prediction = 'True' if scores[0]>0.5 else 'False'
    
    #Creating row for df
    row = {"filename" : image_names[0], "x" : x1, "y" : y1, "w" : x2-x1, "h" : y2-y1, "proper_mask" : prediction}
    
    #Appending to df
    submission = submission.append(row, ignore_index = True)

submission.to_csv('prediction.csv',index=False)
