import torch
from collections import defaultdict
from typing import DefaultDict
from base_deep_recommender import BaseDeepRecommender
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GIMF(torch.nn.Module):
    def __init__(self, config: DefaultDict, num_users:int, num_items:int):
        # TODO : should move to new style super().__init__()
        super(GIMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items 
        self.latent_dim = int(config['latent_dim'])

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(
            in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()
        # here i do very simple init
        self.embedding_user.weight.data.normal_(0., 0.01)
        self.embedding_item.weight.data.normal_(0., 0.01)

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating

    """
        TODO: perhaps I miss here some _smart_ weight initialization, since I do only data.normal
    """
    def init_weight(self):
        return NotImplementedError("some advanced weight init is not implemented")


class GIMFDeepRecommender(BaseDeepRecommender):
    def __init__(self, config, num_users, num_items):
        self.model = GIMF(config, num_users, num_items)
        if config['use_cuda']:
            BaseDeepRecommender.use_cuda(True)
            # TypeError: only floating-point types are supported as the default type
            # torch.set_default_dtype(torch.LongTensor)
            self.model.cuda()
        super().__init__(config)
