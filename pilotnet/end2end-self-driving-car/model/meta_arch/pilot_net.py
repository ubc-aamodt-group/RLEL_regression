import torch
import torch.nn as nn
from model.layer.feed_forward import FeedForward
import pickle

class PilotNet(nn.Module):
    def __init__(self, cfg, nbits, init="u",visualizing=False):
        super(PilotNet, self).__init__()
        self.cfg = cfg
        self.visualizing = visualizing
        self.nbits = nbits

        # BUILD CNN BACKBONE
        cnn_layers = []
        input_channels = self.cfg.MODEL.CNN.INPUT_CHANNELS
        cnn_configs = self.cfg.MODEL.CNN.LAYERS
        for cnn_config in cnn_configs:
            cnn_layer = [nn.Conv2d(input_channels,
                                   cnn_config['out_channels'],
                                   cnn_config['kernel'],
                                   cnn_config['stride']),
                         nn.ELU(),
                         nn.Dropout2d(p=self.cfg.MODEL.CNN.DROPOUT)]
            input_channels = cnn_config['out_channels']
            cnn_layers.extend(cnn_layer)

        self.cnn_backbone = nn.Sequential(*cnn_layers)

        # BUILD FULLY CONNECTED
        self.embedding = FeedForward(self.cfg)
        last_embedding_size = self.cfg.MODEL.FC.LAYERS[-1]['to_size']
        self.to_out = nn.Linear(last_embedding_size, self.nbits)
        self.feed_forward = nn.Sequential(self.embedding, self.to_out)
        self.code = torch.nn.Parameter(torch.rand(size=(self.nbits, 700), dtype=torch.float, requires_grad=True).cuda())
        if init!="rand":
            di=torch.tensor(pickle.load(open("./encodings/bel"+init+"_"+str(700)+"_tensor.pkl","rb")))
            di=torch.transpose(di,0,1).cuda()
            with torch.no_grad():
        	    self.code.copy_(di)
        
        # self.feed_forward = nn.Linear(self.cfg.MODEL.FC.INPUT, self.nbits)

        # BUILD LOSS CRITERION
        # self.loss_criterion = nn.MSELoss()

    def forward(self, input):
        batch_size = input.size(0)
        normalized_input = input / 127.5 - 1
        normalized_input = normalized_input.permute(0, 3, 1, 2)   # dislocate depth axis

        cnn_features = self.cnn_backbone(normalized_input)
        flattened_features = cnn_features.reshape((batch_size, -1))
        predictions = self.feed_forward(flattened_features)
        x = torch.matmul(predictions,self.code)
        #print(predictions.size(),x.size())
        if self.training:
            # loss = self.loss_criterion(targets, predictions)
            return x,predictions

        if self.visualizing:
            activations = []
            layers_activation = normalized_input
            for i, module in enumerate(self.cnn_backbone.children()):
                layers_activation = module(layers_activation)
                if type(module) == nn.ELU:
                    layers_activation_temp = layers_activation.clone()
                    layers_activation_temp = layers_activation_temp.detach()
                    layers_activation_temp = layers_activation_temp.mean(1, keepdim=True)
                    activations.append(layers_activation_temp)
            return predictions, activations

        return x,predictions
