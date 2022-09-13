import torch

model = torch.load(r'/home/insign/Doc/insign/flexible-yolov5/runs/train/polyp_YOLO5/weights/best.pt')
weights = dict()
#extrate weight from model_a
for name, para in model['model'].named_parameters():
  #sign the specific layer like 'backbone'
  if 'backbone' in name:
    weights[name] = para
    print(name, para.shape)

#loading model_b
model_b = torch.load(r'/home/insign/Doc/insign/flexible-yolov5/runs/train/polyp_morphYOLO3/weights/best.pt')
#update model_b weight with model_a
model_b_weight = model_b['model'].state_dict()
model_b_weight.update(weights)
new_weight = model_b['model'].load_state_dict(model_b_weight)
# save model 
torch.save(model_b, 'morphYOLO_1220.pt')