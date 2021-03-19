import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        self.lstm = nn.LSTM(input_size = cfg['NUM_CS'] + cfg['NUM_US'] + cfg['NUM_DIST'],
                            hidden_size = cfg['HIDDEN_L_SIZE'],
                            num_layers = 1,
                            dropout = 0)
        self.linear = nn.Linear(cfg['HIDDEN_L_SIZE'], 1)
        self.hidden = (torch.randn(1, 1, cfg['HIDDEN_L_SIZE']).to(device),
                       torch.randn(1, 1, cfg['HIDDEN_L_SIZE']).to(device))
        self.hidden_0 = self.hidden # used to save (h_0, c_0) of last batch

    def reset_hidden(self):
        self.hidden = (torch.randn(1, 1, cfg['HIDDEN_L_SIZE']).to(device),
                       torch.randn(1, 1, cfg['HIDDEN_L_SIZE']).to(device))

    def forward(self, obs):
        obs = obs.view(len(obs), 1, -1)
        lstm_out_0, self.hidden = self.lstm(obs[0].view(1,1,-1), self.hidden_0)

        # save the hidden state (h_0 and c_0) of first element for when the LSTM is called
        # so that it can be used when a new batch is passed in
        self.hidden_0 = (self.hidden[0].detach(), self.hidden[1].detach())
        lstm_out, self.hidden = self.lstm(obs[1:], self.hidden)
        lstm_out = torch.cat((lstm_out_0, lstm_out))

        # hidden_size -> 1 for each obs
        values = self.linear(lstm_out.view(len(obs), -1))
        return values
