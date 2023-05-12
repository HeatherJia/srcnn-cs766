import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import DatasetFromFolder
from model import SRCNN

from torchvision.models import vgg19
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='SRCNN training parameters')
parser.add_argument('--zoom_factor', type=int, required=True)
parser.add_argument('--nb_epochs', type=int, default=200)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Parameters
BATCH_SIZE = 4
NUM_WORKERS = 0 # on Windows, set this variable to 0

trainset = DatasetFromFolder("data/train", zoom_factor=args.zoom_factor)
testset = DatasetFromFolder("data/test", zoom_factor=args.zoom_factor)

trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(  # we use Adam instead of SGD like in the paper, because it's faster
    [
        {"params": model.conv1.parameters(), "lr": 0.0001},  
        {"params": model.conv2.parameters(), "lr": 0.0001},
        {"params": model.conv3.parameters(), "lr": 0.00001},
    ], lr=0.00001,
)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)

feature_extractor = FeatureExtractor().cuda()
feature_extractor.eval() # Set feature extractor to inference mode
criterion_content = nn.L1Loss().cuda()
vgg19_model = vgg19(pretrained=True)
Tensor = torch.cuda.FloatTensor

for epoch in range(args.nb_epochs):

    # Train
    epoch_loss = 0
    for iteration, batch in enumerate(trainloader):
        input, target = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()

        out = model(input)
        new_out = out.repeat(1,3,1,1)
        new_target = target.repeat(1,3,1,1)
        # print("\nout shape=", new_out.shape)
        # print("\ntarget shape=", new_target.shape)
        
        gen_features = feature_extractor(Variable(new_out.type(Tensor)))
        real_features = feature_extractor(Variable(new_target.type(Tensor)))
        content_loss = criterion_content(gen_features, real_features.detach())
        loss = criterion(out, target) + content_loss
        loss.backward()
        
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch}. Training loss: {epoch_loss / len(trainloader)}")

    # Test
    avg_psnr = 0
    with torch.no_grad():
        for batch in testloader:
            input, target = batch[0].to(device), batch[1].to(device)

            out = model(input)
            loss = criterion(out, target)
            psnr = 10 * log10(1 / loss.item())
            avg_psnr += psnr
    print(f"Average PSNR: {avg_psnr / len(testloader)} dB.")

    # Save model
    torch.save(model, f"model_{epoch}.pth")
