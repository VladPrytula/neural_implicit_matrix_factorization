import torch
import numpy as np
from collections import defaultdict
from typing import DefaultDict
from base_deep_recommender import BaseDeepRecommender
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeuIMF(torch.nn.Module):
    def __init__(self, config: DefaultDict, num_users: int, num_items: int):
        super().__init__()
        self._config = config
        self.num_users = num_users
        self.num_items = num_items
        print(config)
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']

        # Initial Embeddings
        self.embedding_user_mlp = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(
            in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.logistic = torch.nn.Sigmoid()

        self.embedding_user_mlp.weight.data.normal_(0., 0.01)
        self.embedding_item_mlp.weight.data.normal_(0., 0.01)
        self.embedding_user_mf.weight.data.normal_(0., 0.01)
        self.embedding_item_mf.weight.data.normal_(0., 0.01)

        def glorot_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(6. / (fan_in + fan_out))
            layer.weight.data.uniform_(-limit, limit)

        def lecunn_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features  # noqa: F841, E501
            limit = np.sqrt(3. / fan_in)
            layer.weight.data.uniform_(-limit, limit)

        for layer in self.fc_layers:
            if type(layer) != torch.nn.Linear:
                continue
            glorot_uniform(layer)
        lecunn_uniform(self.affine_output)

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        # the concat latent vector
        mlp_vector = torch.cat(
            [user_embedding_mlp, item_embedding_mlp], dim=-1)
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def load_pretrain_weights(self):
        raise NotImplementedError(
            r'NeuIMF says: \"I do not know how to load weights\"')


class NeuIMFDeepRecommender(BaseDeepRecommender):
    def __init__(self, config, num_users, num_items):
        self.model = NeuIMF(config, num_users, num_items)
        if config['use_cuda']:
            BaseDeepRecommender.use_cuda(True)
            # TypeError: only floating-point types are supported as the default type
            # torch.set_default_dtype(torch.LongTensor)
            self.model.cuda()
            assert next(self.model.parameters()).is_cuda
        super().__init__(config)
