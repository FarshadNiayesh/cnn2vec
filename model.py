import torch
import torch.nn.functional as F
import torch.nn as nn


class Word2CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_dims, kernel_sizes,
                 highway_layers=2, batch_norm=False):
        super(Word2CNN, self).__init__()
        self._batch_norm = batch_norm
        self._embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.init_weight(self.embeddings)
        self._convs = nn.ModuleList([
            nn.Conv2d(1, dim, (size, embedding_dim), padding=(size-1, 0))
            for dim, size in zip(kernel_dims, kernel_sizes)])
        self._convs.apply(self._init_weight)
        self._internal_dim = sum(kernel_dims)
        self._batch_norm_conv = nn.BatchNorm1d(self._internal_dim)
        self._hw_num_layers = highway_layers
        self._hw_nonlinear = nn.ModuleList([
            nn.Linear(self._internal_dim, self._internal_dim)
            for _ in range(highway_layers)])
        self._hw_nonlinear.apply(self._init_weight)
        self._hw_linear = nn.ModuleList([
            nn.Linear(self._internal_dim, self._internal_dim)
            for _ in range(highway_layers)])
        self._hw_linear.apply(self._init_weight)
        self._hw_gate = nn.ModuleList([
            nn.Linear(self._internal_dim, self._internal_dim)
            for _ in range(highway_layers)])
        self._hw_gate.apply(self._init_weight)
        self._batch_norm_hw = nn.BatchNorm1d(self._internal_dim)
        self._final_layer = nn.Linear(self._internal_dim * 2, 2)
        self._init_weight(self._final_layer)
        self._logsigmoid = nn.LogSigmoid()

    def _init_weight(self, layer):
        if type(layer) != nn.ModuleList:
            torch.nn.init.uniform(layer.weight, a=-0.05, b=0.05)

    def _highway(self, inputs):
        for layer in range(self._hw_num_layers):
            gate = F.sigmoid(self._hw_gate[layer](inputs))
            nonlinear = F.relu(self._hw_nonlinear[layer](inputs))
            linear = self._hw_linear[layer](inputs)
            inputs = gate * nonlinear + (1 - gate) * linear
        return inputs

    def char_cnn(self, inputs):
        # [BATCH, 1, MAX_LENGTH, EM_SIZE]
        inputs = self._embeddings(inputs).unsqueeze(1)
        # [BATCH, K_DIM, MAX_LENGTH]*len(Ks)
        inputs = [F.tanh(conv(inputs)).squeeze(3) for conv in self._convs]
        # [BATCH, K_DIM]*len(Ks)
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]
        # [BATCH, K_DIM*len(Ks)]
        inputs = torch.cat(inputs, 1)
        if self._batch_norm:
            inputs = self._batch_norm_conv(inputs)
        inputs = self._highway(inputs)
        if self._batch_norm:
            inputs = self._batch_norm_hw(inputs)
        return inputs

    def forward(self, sequences, num_negs):
        embeddings = self.char_cnn(sequences)
        embeddings = embeddings.view(-1, num_negs + 2, embeddings.size()[-1])
        center_embeds = embeddings[:, 0, :].unsqueeze(1)
        target_embeds = embeddings[:, 1, :].unsqueeze(1)
        negati_embeds = -1*embeddings[:, 2:, :]
        p_scre = target_embeds.bmm(center_embeds.transpose(1, 2))
        n_scre = torch.sum(negati_embeds.bmm(center_embeds.transpose(1, 2)), 1)
        loss = self._logsigmoid(p_scre) + self._logsigmoid(n_scre)
        return -torch.mean(loss)

    def prediction(self, inputs):
        return self.char_cnn(inputs)
