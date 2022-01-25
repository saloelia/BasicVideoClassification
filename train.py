from torch.nn.modules.conv import Conv2d
from dataloader import num_of_frames,labels_dict,get_videos_and_labels,VideoDataset
from model import load_model,save_model,VideoClassifier
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch.nn
from tqdm import tqdm
import torch.optim as optim
import torchvision.models as models
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_validation(model,criterion,loader):
    model.eval()
    losses, accs = [], []
    for frames, target in tqdm(loader):
        frames,target = frames.to(device),target.to(device)
        
        with torch.no_grad():
            output = model(frames)
            loss = criterion(output, target)
        
        frames, target, output=frames.to('cpu'), target.to('cpu'),output.to('cpu')

        losses.append(loss.item())
        predictions = np.argmax(output, axis=1)    
        acc = accuracy_score(target, predictions)
        accs.append(acc)

        del frames
        del target
        del output
        torch.cuda.empty_cache()

    return np.mean(losses), np.mean(accs)    


def train(model, loader, optimizer, criterion):
    model.train()
    losses, accs = [], []
    for frames, target in tqdm(loader):
        frames,target = frames.to(device),target.to(device)
        
        optimizer.zero_grad()
        output = model(frames)
        loss = criterion(output, target)   
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

        frames,target,output=frames.to('cpu'), target.to('cpu'),output.to('cpu')
        with torch.no_grad():
            predictions = np.argmax(output, axis=1)
        acc = accuracy_score(target,predictions)
        accs.append(acc)

        del frames
        del target
        del output
        torch.cuda.empty_cache()

    return np.mean(losses), np.mean(accs)



def train_val(model, loader_train, loader_val, optimizer, criterion, epochs,path):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    try:
        train_loss, val_loss, train_acc, val_acc, last_e = load_model(path, model, optimizer)
        print(f"loaded the model from epoch:{last_e} \n")
    except:
        last_e = 0
        print("starting a new train \n")

    for e in tqdm(range(last_e, epochs)):
        loss_t, acc_t = train(model, loader_train, optimizer, criterion)
        
        print(f"**\nepoch {e+1}: train loss: {loss_t} accuracy: {acc_t}\n**")
        writer.add_scalar('Loss/train', loss_t, e)
        writer.add_scalar('accuracy/train', acc_t, e)

        train_loss.append(loss_t)
        train_acc.append(acc_t)

        loss_v , acc_v = test_validation(model, criterion, loader_val)
        print(f"**\nepoch {e+1}: val loss: {loss_v} accuracy: {acc_v}\n**")
        writer.add_scalar('Loss/validation', loss_v, e)
        writer.add_scalar('accuracy/validation', acc_v, e)

        val_loss.append(loss_v)
        val_acc.append(acc_v)

        save_model(model, optimizer, e, train_loss, val_loss, train_acc, val_acc, path)
    
    return model,train_loss,train_acc,val_loss,val_acc    


def  main():
    transform = transforms.Compose(
            [ 
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
     
    videos,labels = get_videos_and_labels()
    videos_train, videos_val , _ , _ = train_test_split(videos, labels, test_size=0.2, random_state=42)
    videos_val, videos_test = train_test_split(videos, test_size=0.5, random_state=42)
    
    train_dataset = VideoDataset(videos_train, transforms=transform,train=True)
    loader_train = DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=4)
    val_dataset = VideoDataset(videos_val,transforms=transform ,train=False) 
    loader_val = DataLoader(val_dataset,batch_size=4, shuffle=False,num_workers=4)
    test_dataset = VideoDataset(videos_test,transforms=transform ,train=False) 
    loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    
    # model.conv1.in_channels = 3*num_of_frames
    # model.fc.out_features = len(labels_dict)
    
    model = VideoClassifier()
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimi = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 50
    path_to_model = '/home/linuxgpu/Downloads/VideoClassification/models/resnet_50_Adam_1e-3.pt'
    model,train_loss,train_acc,val_loss,val_acc = train_val(model, loader_train, loader_val, optimi, criterion, epochs,path_to_model)
    test_loss, test_acc = test_validation(model, criterion, loader_test)
    print(f'Test Metrics\nLoss : {test_loss}\naccuracy : {test_acc}')
    
    # model.train()
    # for frames,target in loader_train:
    #      bs,n_f,c,h,w = frames.shape
    #      frames = frames.view(bs,n_f*c,h,w)
         
    #      frames = frames.to(device)
    #      output = model(frames)
        
    #      frames,target,output=frames.to('cpu'), target.to('cpu'),output.to('cpu')
    #      output = output.numpy()
    #      predictions = np.argmax(output, axis=1)
    #      acc = accuracy_score(target,predictions)
    #      print(output)
    #      print(predictions)
    #      print(target)
    #      print(acc)
    #      break
    
    # train_loss, val_loss, train_acc, val_acc, last_e = load_model(path_to_model,model,optimi)
    # print(train_loss)
    # print(val_loss)
    # print(train_acc)
    # print(val_acc)
    # print(last_e)


if __name__ == "__main__":
    main()