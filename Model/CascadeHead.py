import torch
import numpy as np
import tqdm
import pickle


class CascadeHead(torch.nn.Module):
    def __init__(self, backbone_out, layers=4, layer_dims=[2, 50, 404, 555], activation=torch.nn.ReLU, cumulative=True, device='cuda:0'):
        super(CascadeHead, self).__init__()
        self.layer_dims = layer_dims
        self.cumsum = np.cumsum(layer_dims)
        self.cat = self.cumulative_cat
        
        if not cumulative:
            self.cumsum = self.layer_dims
            self.cat = self.noncumulative_cat
        
        self.linear1 = torch.nn.Linear(backbone_out, self.layer_dims[0])
        self.linear2 = torch.nn.Linear(backbone_out + self.cumsum[0], self.layer_dims[1])
        self.linear3 = torch.nn.Linear(backbone_out + self.cumsum[1], self.layer_dims[2])
        self.linear4 = torch.nn.Linear(backbone_out + self.cumsum[2], self.layer_dims[3])

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
    
        #with open('./labels/hierarchical_logit_mappings.pkl', 'rb') as f:
        #    class_mappings = pickle.load(f)

        #self.c12 = torch.tensor(class_mappings[0]).to(torch.float32).to(device)
        #self.c23 = torch.tensor(class_mappings[1]).to(torch.float32).to(device)
        #self.c34 = torch.tensor(class_mappings[2]).to(torch.float32).to(device)

        #print('With Conditional Probs!')


    def forward(self, x):
        probs = []
        out1 = self.linear1(x)
        probs.append(out1)
        out1 = self.relu(out1)

        #in2 = torch.cat((x,out1), 1) 
        in2 = self.cat(x, x, out1)

        # Use this line for CondProb
        #out2 = torch.mul(self.linear2(in2), self.softmax(probs[-1])@self.c12)

        # Use this line for plain
        out2 = self.linear2(in2)
        
        probs.append(out2)
        out2 = self.relu(out2)

        #in3 = torch.cat((in2,out2), 1) 
        #in3 = torch.cat((x,out2), 1) 
        in3 = self.cat(x, in2, out2)

        # Use this line for CondProb
        #out3 = torch.mul(self.linear3(in3), torch.mul(self.softmax(probs[-1]),self.softmax(probs[-2])@self.c12)@self.c23)

        # Use this line for plain
        out3 = self.linear3(in3)
        
        probs.append(out3)
        out3 = self.relu(out3)
        
        #in4 = torch.cat((in3,out3), 1) 
        #in4 = torch.cat((x,out3), 1) 
        in4 = self.cat(x, in3, out3)
       
        # Use this line for CondProb
        #out4 = torch.mul(self.linear4(in4), torch.mul(self.softmax(probs[-1]), torch.mul(self.softmax(probs[-2]),self.softmax(probs[-3])@self.c12)@self.c23)@self.c34)
        
        # Use this line for plain
        out4 = self.linear4(in4)
        
        probs.append(out4)

        return probs

    def noncumulative_cat(self, x, last_in, last_out):
        return torch.cat((x, last_out), 1)

    def cumulative_cat(self, x, last_in, last_out):
        return torch.cat((last_in, last_out), 1)


if __name__ == '__main__':
    print('')
