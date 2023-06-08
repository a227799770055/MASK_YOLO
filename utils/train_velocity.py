import sys, os
sys.path.append('.')
import datetime, time, copy, yaml
import wandb

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD, lr_scheduler

from model.od.data.datasets import letterbox
from model.backbone_YOLO import *
from model.head_RCNN import *
from model.head_velocity import *
from model.groundtrue_import import *
from dataset import SegmentDataset
from dataset_velocity import veclocityDataset

def train_segm(model, yolo, dataloader, criterion, optimizer):
    running_loss = 0.0
    #   Iterate over data
    for image, mask in tqdm(dataloader):
        #   yolo_backbone return rois and features
        pred = yolo(image)
        rois = non_max_suppression(pred['rois'][0],cfg['nms']['conf_thres'], cfg['nms']['iou_thres'], classes= cfg['nms']['classes'],agnostic=cfg['nms']['agnostic_nms'])
        boxes = rois[0][:,:4]#  rois
        #   if do not have bbox, set bbox as full image size
        if len(boxes)==0:
            boxes = torch.tensor([[0.0, 0.0, 497.0, 497.0]]).to(device)
        feature_map = featuremapPack(pred['feature_map']) #   extract feature map and boxes
        
        #   mask prediction
        #   0920 reconstruct feature_map --> feature1 feature2 feature3 
        f1,f2,f3 = pred['feature_map'][0], pred['feature_map'][1],pred['feature_map'][2]
        mask_logits = model(f1,f2,f3, boxes)
        #mask_logits = model(feature_map, boxes)

        #   import ground true
        mask_roi =  maskRoiAlign(mask, boxes, mask_logits.shape[-1])
        
        #   zero the parameter gradient
        optimizer.zero_grad()
        #   back propagation
        loss = criterion(mask_logits, mask_roi)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    wandb.log({'training_loss_segm':running_loss})
    return running_loss

def train_velo(model, yolo, dataloader, criterion, optimizer):
    running_loss = 0.0
    
    for image1, image2, _, _, position1, position2 in tqdm(dataloader):
        
        image1 = image1.cuda()
        image2 = image2.cuda()
        position1 = position1.cuda()
        position2 = position2.cuda()

        pred1 = yolo(image1)
        pred2 = yolo(image2)
        # feature_map1 = featuremapPack(pred1['feature_map'])
        # feature_map2 = featuremapPack(pred2['feature_map'])
        
        f11,f12,f13 = pred1['feature_map'][0], pred1['feature_map'][1],pred1['feature_map'][2]
        f21,f22,f23 = pred2['feature_map'][0], pred2['feature_map'][1],pred2['feature_map'][2]
        
        veloctiy_logits = model(f11, f12, f13, f21, f22, f23)
        
        loss = criterion(veloctiy_logits, position2 - position1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    wandb.log({'training_loss_velo':running_loss})
    
    return running_loss

def train_model(models, yolo, dataloader, criterion, optimizer, num_epochs=30):
    print("\nTraining:-\n")
    since = time.time()

    best_model_wts = {'segm':copy.deepcopy(models['segm'].state_dict()), 'velo':copy.deepcopy(models['velo'].state_dict())}
    best_loss = {'segm':float('inf'), 'velo':float('inf')}
    loss_save = {'segm':[], 'velo':[]}

    for epoch in range(num_epochs):
        print('-' * 10)
        #   Each epoch have train and validation phase
        running_loss = {'segm':0.0, 'velo':0.0}
        for phase in ['segm', 'velo']: 
            print(f'Epoch {epoch}/{num_epochs-1} {phase}')
            model = models[phase]
            model.train()
            
            if phase == 'segm':
                running_loss[phase] = train_segm(model, yolo, dataloader['train_segm'], criterion, optimizer['segm'])
            elif phase == 'velo':
                running_loss[phase] = train_velo(model, yolo, dataloader['train_velo'], criterion, optimizer['velo'])
                            
            #   preserve best model
            if  running_loss[phase] < best_loss[phase]:
                best_loss[phase] = running_loss[phase]
                best_model_wts[phase] = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts[phase])
                torch.save(model, f'best_{phase}.pt')

        #   Save loss log file
        with open('loss.txt', 'a') as f:
            f.write("Epoch {}     =     ".format(str(epoch+1)))
            f.write(f'segm loss : {running_loss["segm"]}, velo loss : {running_loss["velo"]}\n')
        loss_save['segm'].append(running_loss['segm'])
        loss_save['velo'].append(running_loss['velo'])
        print(f'Loss = segm : {running_loss["segm"]:.4f}, velo : {running_loss["velo"]:.4f}')
    print(' ')
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    models['segm'].load_state_dict(best_model_wts['segm'])
    models['velo'].load_state_dict(best_model_wts['velo'])
    return models


if __name__ == '__main__':
    today = datetime.date.today()

    wandb.init(project="YOLO_Mask Polyp", name=f"train_{today}")

    config = 'config/config.yaml'
    
    with open(config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    
    root = cfg['data']['root']
    img_dir = cfg['data']['img_dir']
    mask_dir = cfg['data']['mask_dir']
    
    #   Load data from folder
    dataset = {
        'train_segm': SegmentDataset(root=root, image_folder=img_dir, mask_folder=mask_dir),
        'valid_segm': SegmentDataset(root=root, image_folder=img_dir, mask_folder=mask_dir),
        'train_velo': veclocityDataset(root=root),
        'valid_velo': veclocityDataset(root=root)
    }
    
    #   Create iterators for data loading
    dataloader = {
        'train_segm': DataLoader(dataset['train_segm'], batch_size=1, num_workers=0),
        'valid_segm': DataLoader(dataset['valid_segm'], batch_size=1, num_workers=0),
        'train_velo': DataLoader(dataset['train_velo'], batch_size=32, num_workers=0),
        'valid_velo': DataLoader(dataset['valid_velo'], batch_size=32, num_workers=0)
    }

    #   Set default device as GPU, if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #   Loading yolo_backbone
    yolo_backbone = model_manipulate(cfg['model']['weight']).eval().to(device).float()

    #   Loading mask_head
    if os.path.isfile(cfg['maskrcnn']['weight']):
        mask_wts = cfg['maskrcnn']['weight']
        mask_head = torch.load(mask_wts).to(device)
    else:
        mask_head = ROIHeadsMask().to(device)
        
    if os.path.isfile(cfg['velocity']['weight']):
        velocity_wts = cfg['velocity']['weight']
        velocity_head = torch.load(velocity_wts).to(device)
    else:
        velocity_head = VelocityHeads().to(device)
    
    head = {
        'segm': mask_head,
        'velo': velocity_head
    }
    
    #   Criterions
    # criterions = torch.nn.BCEWithLogitsLoss()
    criterions = torch.nn.MSELoss(reduction='mean')

    #   Number of epochs
    num_epochs = cfg['num_epochs']

    #   Optimizer
    if cfg['optimizer'] == 'SGD':
        optimizer_ft = {'segm':SGD(mask_head.parameters(), lr = 0.001, momentum=0.9),
                        'velo':SGD(velocity_head.parameters(), lr = 0.001, momentum=0.9)}
    else:
        optimizer_ft = {'segm':Adam(mask_head.parameters(), lr = 1e-4, betas=(0.9, 0.999), eps = 1e-08, weight_decay=0, amsgrad=False),
                        'velo':Adam(velocity_head.parameters(), lr = 1e-4, betas=(0.9, 0.999), eps = 1e-08, weight_decay=0, amsgrad=False)}

    #   Learning rate decay
    exp_lr_scheduler = {
        'segm': lr_scheduler.StepLR(optimizer_ft['segm'], step_size=3, gamma=0.1),
        'velo': lr_scheduler.StepLR(optimizer_ft['velo'], step_size=3, gamma=0.1)
    }
    
    best_model = train_model(
                    models= head,
                    yolo= yolo_backbone,
                    dataloader= dataloader,
                    criterion= criterions,
                    optimizer= optimizer_ft,
                    num_epochs= num_epochs
                )