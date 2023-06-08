import torch
import numpy as np
import torch.nn as nn
import dgl

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(2,100)

        num_blocks = 1

        self.blocks = nn.ModuleList([
            self.create_block_layers() for _ in range(num_blocks)
        ])

        self.output_layers = nn.Sequential(
            nn.Linear(100,50),
            nn.ReLU(inplace=True),
            nn.Linear(50,10)
        )



    def create_block_layers(self):
        block_layers = nn.Sequential(
            nn.Linear(100 * 2, 100),
            nn.ReLU(),
            nn.ModuleList([
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(100, 100),
                )
                for _ in range(5)
            ]),
            nn.BatchNorm1d(100)
        )
        return block_layers



        
        # ...
       
    def forward(self, g):

        g.ndata['hidden rep'] = self.input_layer(g.ndata['xy'])
        for block_layer in self.blocks:            
            mean_of_node_rep = dgl.mean_nodes(g,'hidden rep')
            broadcasted_sum = dgl.broadcast_nodes(g,mean_of_node_rep)
            g.ndata['global rep'] = broadcasted_sum
            input_to_block_layer = torch.cat([
                                g.ndata['global rep'], 
                                g.ndata['hidden rep']],dim=1)

            input_to_block_layer = block_layer[0](input_to_block_layer)
            input_to_block_layer = block_layer[1](input_to_block_layer)
            for layer in block_layer[2]:
                input_to_block_layer = layer(input_to_block_layer)
            g.ndata['hidden rep'] = block_layer[3](input_to_block_layer)

        mean_of_node_rep = dgl.mean_nodes(g,'hidden rep')


        output = self.output_layers(mean_of_node_rep)


        return output