import matplotlib.pyplot as plt

loss =[]
with open('/home/insign/Doc/insign/Mask_yolo/loss.txt', 'r') as f:
    lines = f.readlines()
    for l in lines:
        loss_ = float(l.split(' ')[-1].split('\n')[0])
        print(loss_)
        loss.append(loss_)
plt.plot(range(len(loss)),loss)
plt.title('Loss')
plt.savefig('loss.png')