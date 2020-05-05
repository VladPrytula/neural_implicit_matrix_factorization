import torch
from collections import defaultdict
from typing import DefaultDict
from base_deep_recommender import BaseDeepRecommender
from gimf import GIMF
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MLP(torch.nn.Module):
    def __init__(self, config: DefaultDict, num_users: int, num_items: int):
        super().__init__()
        self._config = config
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = int(self._config['latent_dim'])

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self._config['layers'][:-1], self._config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(
            in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

        # here i do very simple init
        self.embedding_user.weight.data.normal_(0., 0.01)
        self.embedding_item.weight.data.normal_(0., 0.01)
        # TODO: initialize the other weights in the network

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        # now concatenation layer
        layer_out = torch.cat([user_embedding, item_embedding], dim=-1)
        # now propagate over all the layers:
        for idx, _ in enumerate(range(len(self.fc_layers))):
            layer_out = self.fc_layers[idx](layer_out)
            layer_out = torch.nn.ReLU()(layer_out)
            # if self._config['dropout'] > 0:
                # layer_out = torch.nn.Dropout(p=self._config['dropout'])(layer_out)

        logits = self.affine_output(layer_out)
        rating = self.logistic(logits)
        return rating

    def load_pretrain_weights(self):
        # TODO: i am not sure that this is working correctly
        """Loading weights from trained GMF model"""
        gimf_model = GIMF(self._config)
        if self._config['use_cuda'] is True:
            gimf_model.use_cuda(True)
        resume_checkpoint(
            gimf_model, model_dir=self._config['pretrain_mf'])
        self.embedding_user.weight.data = gimf_model.embedding_user.weight.data
        self.embedding_item.weight.data = gimf_model.embedding_item.weight.data


class MLPDeepRecommender(BaseDeepRecommender):
    def __init__(self, config, num_users, num_items):
        self.model = MLP(config, num_users, num_items)
        if config['use_cuda']:
            BaseDeepRecommender.use_cuda(True)
            # TypeError: only floating-point types are supported as the default type
            # torch.set_default_dtype(torch.LongTensor)
            self.model.cuda()
        super().__init__(config)

        if config['pretrain']:
            raise NotImplementedError(r"i can't load pretrained weights yet")
