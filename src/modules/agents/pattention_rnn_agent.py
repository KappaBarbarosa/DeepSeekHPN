import torch.nn as nn
import torch.nn.functional as F
import torch
from modules.layer.tokenformer.pattention import Pattention
from modules.layer.tokenformer.init_function import get_init_methods

class Pattention_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(Pattention_RNNAgent, self).__init__()
        self.args = args
        # print ("Pattention_RNNAgent start_init")
        self.init_method, self.output_layer_init_method = get_init_methods(
            self.args
        )
        # self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc1 = Pattention(args, input_shape, args.rnn_hidden_dim, args.qkv_slot_num, self.init_method, self.output_layer_init_method)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.fc2 = Pattention(args, args.rnn_hidden_dim, args.n_actions, args.qkv_slot_num, self.init_method, self.output_layer_init_method)
        # print ("Pattention_RNNAgent: ", self.fc1, self.fc2)
    # def init_hidden(self):
    #     # make hidden states on same device as model
    #     return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.rnn_hidden_dim, device=self.fc1.key_param_tokens.device)

    def forward(self, inputs, hidden_state=None):
        b, a, e = inputs.size()
        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            # print ("hidden_state shape", hidden_state.shape)
            
        # print ("x shape", x.shape)
        h = self.rnn(x, hidden_state)
        # print ("h shape", h.shape)
        q = self.fc2(h)
        # print ("q shape", q.shape)
        return q.view(b, a, -1), h.view(b, a, -1)