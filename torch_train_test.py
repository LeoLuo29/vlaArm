## !!!!!!!!!!!!! READ BEFORE THE FOLLOWING BEFORE PROCEEDE !!!!!!!!!!!!!
## This program defines unet convolutional neural network structure and define how input x is propogated to yield prediction y
## This program also defines training method and loss function 
## Training data is inported from stair_data_generator  in the form of numpy array
## imported data is then changed to pytorch tensor and stroed as Dataset and DataLoarder
## to access the trained model and test it, use model_test
## ensure that the two programs mentioned above are placed into the same local file folder that this one is placed




from fileinput import filename
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import decode_image
import numpy as np 
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
outfile = TemporaryFile()
fig = plt.figure(figsize=(50, 50))
ax = fig.add_subplot(111, projection='3d')
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.utils.prune as prune





## creating a dataloader as an object that loads data from .npy file

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

## input data from numpy array into pytorch tensor
class Dataset():
    def __init__(self,inpfilename,ansfilename):
        inp_numpy = np.load(inpfilename)
        ans_numpy = np.load(ansfilename)

        self.tensor_inp = torch.from_numpy(inp_numpy).to(device).float() 
        self.tensor_ans = torch.from_numpy(ans_numpy).to(device).float()
        ## self.tensor_inp/ans[][][]
        ## self.inp/ans_numpu [][][]

    
    def __len__(self):
        return len(self.tensor_inp) ## This only returns the length of 1st rank of the tensor
        ##                          ## e.g.: returns how many sets of datas stored; not the total elements; ASK THIS 

    def __getitem__(self,idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        return self.tensor_inp[idx,...], self.tensor_ans[idx,...] ### it returns the ith set of data by calling __getitem__()
        



## defining the unet structure 
class Stair_UNet(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M ## dimension of the input data; input is a M*M tensor

        self.start_block = nn.Sequential( ## convert the data map from 1 channel into 8 channels. 
            nn.Conv2d(1,8,1,1),
            nn.ReLU()
        )
        ## encoders
        ## input argument: self.enc_block (inchannels, outchannels, kernelsize)
        self.enc_block1 = self.enc_block(8,16,3) ## dimention of data map, input:64*64 ; output:32*32
        self.enc_block2 = self.enc_block(16,32,3) ## dimention of data map, input:32*32 ; output:16*16
        self.enc_block3 = self.enc_block(32,64,3) ## dimention of data map, input:16*16 ; output:8*8 
        self.enc_block4 = self.enc_block(64,128,3) ## dimention of data map, input:8*8 ; output:4*4

        ## bottleneck 
        self.bottle_neck_block = nn.Sequential(
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,3,3,2),
            nn.ReLU()

        )


        ## decoders
        ## the current size values are given to generate the equal output tensor size so that the output tensor matches with the ones of skip connection tensor size
        ## input argument, self.dec_block(inchannel, outchannel, conv kernel, deconv kernel, stride, in padding, outpadding)
        self.dec_block4 = self.dec_block(128,32,3,4,2,1,0)
        self.dec_block3 = self.dec_block(64,16,3,4,2,1,0)
        self.dec_block2 = self.dec_block(32,8,3,4,2,1,0)
        self.dec_block1 = self.dec_block(16,1,3,1,1,0,0)




    ## define each single encoder block with varing channel number and kernelsize
    ## only the last Conv2d layer contracts the data map
    def enc_block(self, inchannels, outchannels, kernelsize):
        return nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernelsize,1,1),
            nn.ReLU(),
            nn.Conv2d(inchannels, inchannels, kernelsize,1,1),
            nn.ReLU(),
            nn.Conv2d(inchannels, outchannels, kernelsize,2,1),
            nn.ReLU()

        )
    

    ## define each single decoder block with varing chennel number and kernelsize
    ## the convolution layer and the decovolution layer have different kernel size, stride size, and padding size. 
    ## input arguments for ConvTranspoze2d() are marked with _t ending
    def dec_block(self, inchannels, outchannels, kernelsize, kernelsize_t = 1, stridesize_t = 1, paddingsize_t = 1, output_padding_t = 0):
        return nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernelsize,1,1),
            nn.ReLU(),
            nn.Conv2d(inchannels,inchannels,kernelsize,1,1),
            nn.ReLU(),
            nn.ConvTranspose2d(inchannels, outchannels, kernel_size = kernelsize_t, stride = stridesize_t, padding = paddingsize_t, output_padding = output_padding_t),
            nn.ReLU()

        )
    

    ## define the foward function; define how input data x is processed through each block to yield prediction y for return
    ## the data map's shape for each decoder map needs to match the corresponding encoder map. 
    def forward(self, x):
        x = x.unsqueeze(1)
        ## run input x through encoders
        ## "enc1" is the output data map after it goes through the 1st encoder;
        x = self.start_block(x)
        enc1 = self.enc_block1(x)
        enc2 = self.enc_block2(enc1)
        enc3 = self.enc_block3(enc2)
        enc4 = self.enc_block4(enc3)
        ## bottle_neck is the data map after it goes through the bottle neck block
        bottle_neck = self.bottle_neck_block(enc4)
        ## run input x through decoders
        ## concatenation for skip connection
        ## "dec4" is the data map after it goes through the 4th decoder.                                                  
        dec4 = self.dec_block4(torch.cat((bottle_neck, enc3), 1))
        dec3 = self.dec_block3(torch.cat((dec4, enc2), 1))
        dec2 = self.dec_block2(torch.cat((dec3, enc1), 1))
        dec1 = self.dec_block1(torch.cat((dec2, x), 1))  ## dec1 is defined slightly different than other decoders, so that it does not change the shape of the data map
        y = dec1
        return y





if __name__ == "__main__":

    ##
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader)
        model.train()
        for batch, (x,y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            ## Compute prediction error
            pred = model(x) ## pred stands for prediction 
            loss = loss_fn(pred.flatten(), y.flatten())

            ## Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            ##
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            

    ##
    def test(dataloader, model, loss_fn):
        size = len(dataloader)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                test_loss += loss_fn(pred.flatten(), y.flatten()).item()
                ## breakpoint()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n  Avg loss: {test_loss:>8f} \n")


    ##
    model = Stair_UNet(20).to(device)
    data = Dataset("generated_inp_train2.npy","generated_ans_train2.npy")
    batch_size = 7
    data_loader = DataLoader(data,batch_size = batch_size)

    ## define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ## adjust epoch number for training HERE BELOW
    epochs = 2560
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(data_loader, model, loss_fn, optimizer)
        test(data_loader, model, loss_fn)


    ## saving the trained model into the corresponding local file folder
    PATH = "unet_nn_v2a_t.pt"
    torch.save(model, PATH)
    print("Done!")



    ## text output to check whether the program is succesfully compiled and executed or not
    ## always keep this line at the very end of the program
    print("ok it went through")