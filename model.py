from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, width=2, fix_w1=False, scale=1.):
        super().__init__()
        self.fc1 = nn.Linear(1,width)
        self.fc2 = nn.Linear(width,1, bias=False)

        # init
        if fix_w1:
            nn.init.constant_(self.fc1.weight, scale)
            self.fc1.weight.requires_grad = False
        else:
            nn.init.uniform_(self.fc1.weight, -scale, scale)
        nn.init.uniform_(self.fc1.bias, -scale, scale)
        nn.init.uniform_(self.fc2.weight, -1/(scale*sqrt(width)), 1/(scale*sqrt(width)))

        with torch.no_grad():
            ## randomly initialize by +- 1.
            #w = torch.ones(width, 1).float() * scale
            #r = torch.rand(width)
            #w[r>=0.5] = -1.
            #self.fc1.weight = nn.Parameter(w)

            #b = torch.linspace(-scale,scale,width)
            #w2 = torch.linspace(-1/(scale*sqrt(width)),1/(scale*sqrt(width)),width).view(1,width)
            b = torch.Tensor(sorted(self.fc1.bias.data.tolist()))
            self.fc1.bias = nn.Parameter(b)
            #self.fc2.weight = nn.Parameter(w2)
            pass

    def forward(self, x):
        hidden = F.leaky_relu(self.fc1(x))
        out = self.fc2(hidden)
        return out
    
    def get_knots(self):
        with torch.no_grad():
            knots = - self.fc1.bias.flatten() / self.fc1.weight.flatten()# + 1e-16)
            return knots.clone().detach().cpu()
    
    def print_weights(self):
        with torch.no_grad():
            print('fc1.weight', self.fc1.weight.data.flatten().numpy())
            print('fc1.bias', self.fc1.bias.data.flatten().numpy())
            print('fc2.weight', self.fc2.weight.data.flatten().numpy())
            print('knots_x', self.get_knots().flatten().numpy())
            print('knots_y', self.forward(self.get_knots().view(-1,1)).flatten().numpy())
        #if self.fc2.weight.grad is not None:
        #    print('fc1.bias.grad', self.fc1.bias.grad.flatten().numpy())
        #    print('fc2.weight.grad', self.fc2.weight.grad.flatten().numpy())
            

