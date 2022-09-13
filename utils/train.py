import sys
sys.path.append('/home/insign/Doc/insign/Mask_yolo')
import datetime, time, copy, yaml
import wandb

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD, lr_scheduler

from model.od.data.datasets import letterbox
from model.backbone_YOLO import *
from model.head_RCNN import *
from model.groundtrue_import import *
from dataset import SegmentDataset


def train_model(model, yolo, dataloader, criterion, optimizer, num_epochs=30):
    print("\nTraining:-\n")
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    loss_save = []

    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        
        #   Each epoch have train and validation phase
        for phase in ['train']: #TODO 0822 ['train', 'valid'] --> ['train']
            print(phase)
            if phase == 'train':
                model.train()
            else:
                model.eval()
        
            running_loss = 0.0
            running_correct = 0

            #   Iterate over data
            for image, mask in tqdm(dataloader[phase]):
                #   yolo_backbone return rois and features
                pred = yolo(image)
                rois = non_max_suppression(pred['rois'][0],cfg['nms']['conf_thres'], cfg['nms']['iou_thres'], classes= cfg['nms']['classes'],agnostic=cfg['nms']['agnostic_nms'])
                boxes = rois[0][:,:4]#  rois
                #   if do not have bbox, set bbox as full image size
                if len(boxes)==0:
                    boxes = torch.tensor([[0.0, 0.0, 497.0, 497.0]]).to(device)
                feature_map = featuremapPack(pred['feature_map']) #   extract feature map and boxes
                
                #   mask prediction
                mask_logits = model(feature_map, boxes)
                #   import ground true
                mask_roi =  maskRoiAlign(mask, boxes, mask_logits.shape[-1])
                
                #   zero the parameter gradient
                optimizer.zero_grad()
                #   back propagation
                loss = criterion(mask_logits, mask_roi)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
            wandb.log({'training_loss':running_loss})
                #   accuracy

            #   preserve best model
            if phase == 'train' and running_loss < best_loss:  #   TODO 0822 'valid' --> 'train'
                best_loss = running_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)
                torch.save(model, 'best.pt')

        #   Save loss log file
        with open('loss.txt', 'a') as f:
            f.write("Epoch {}     =     ".format(str(epoch+1)))
            f.write(str(running_loss)+'\n')
        loss_save.append(running_loss)        
        print('Loss = {:.4f}'.format(running_loss))
    print(' ')
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    today = datetime.date.today()
    wandb.init(project="YOLO_Mask Polyp {}_".format(today))

    yolo_cfg = 'config/config.yaml'
    
    with open(yolo_cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    
    root = cfg['data']['root']
    img_dir = cfg['data']['img_dir']
    mask_dir = cfg['data']['mask_dir']
    
    #   Load data from folder
    dataset = {
        'train': SegmentDataset(root=root, image_folder=img_dir, mask_folder=mask_dir),
        'valid': SegmentDataset(root=root, image_folder=img_dir, mask_folder=mask_dir)
    }
    
    #   Create iterators for data loading
    dataloader = {
        'train': DataLoader(dataset['train'], batch_size=1, num_workers=0),
        'valid': DataLoader(dataset['valid'], batch_size=1, num_workers=0)
    }

    #   Set default device as GPU, if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #   Loading yolo_backbone
    yolo_backbone = model_manipulate(cfg['model']['weight']).eval().to(device)

    #   Loading mask_head
    if cfg['maskrcnn']['weight']:
        mask_wts = cfg['maskrcnn']['weight']
        mask_head = torch.load(mask_wts).to(device)
    else:
        mask_head = ROIHeadsMask().to(device)
    
    #   Criterions
    # criterions = torch.nn.BCEWithLogitsLoss()
    criterions = torch.nn.MSELoss(reduction='mean')

    #   Number of epochs
    num_epochs = cfg['num_epochs']

    #   Optimizer
    if cfg['optimizer'] == 'SGD':
        optimizer_ft = SGD(mask_head.parameters(), lr = 0.001, momentum=0.9)
    else:
        optimizer_ft = Adam(mask_head.parameters(), lr = 1e-4, betas=(0.9, 0.999), eps = 1e-08, weight_decay=0, amsgrad=False)

    #   Learning rate decay
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
    
    best_model = train_model(
                    model= mask_head,
                    yolo= yolo_backbone,
                    dataloader= dataloader,
                    criterion= criterions,
                    optimizer= optimizer_ft,
                    num_epochs= num_epochs
                )