import torch
from copy import deepcopy
import random
import pandas as pd
from tqdm import tqdm

# import os
# os.environ["MODIN_ENGINE"] = "ray"
# import modin.pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EvaluationDataset(Dataset):
    def __init__(
        self,
        positive_interactions_users,
        test_items,
        negative_interactions_users,
        negative_items,
    ):
        self.positive_interactions_users = positive_interactions_users
        self.test_items = test_items
        self.negative_interactions_users = negative_interactions_users
        self.negative_items = negative_items

    def __getitem__(self, index):
        return (
            self.positive_interactions_users[index],
            self.test_items[index],
            self.negative_interactions_users[index],
            self.negative_items[index],
        )

    def __len__(self):
        return self.positive_interactions_users.size(0)


class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor that corresponds to (user, item) pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        print(user_tensor)
        print(target_tensor)

    def __getitem__(self, index):
        """
        Map-style datasets https://pytorch.org/docs/stable/data.html#map-style-datasets
        """
        return (
            self.user_tensor[index],
            self.item_tensor[index],
            self.target_tensor[index],
        )

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator:
    def __init__(self, ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert "userId" in ratings.columns
        assert "itemId" in ratings.columns
        assert "rating" in ratings.columns

        self.ratings = ratings

        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings["userId"].unique())
        self.item_pool = set(self.ratings["itemId"].unique())
        # create negative item samples for NCF learning
        # self.negatives = self._sample_negatives(ratings)
        self.negatives = self._sample_negatives_low_mem(ratings)
        self.train_ratings, self.test_ratings = self._leave_last_out(
            self.preprocess_ratings
        )

    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        logger.info("binarizing data")
        # ratings.rating.applymap(lambda x: 1.0 if (x>0) else 0.0)
        # ratings = deepcopy(ratings)
        # ratings['rating'][ratings['rating'] > 0] = 1.0
        ratings["rating"] = ratings.rating.map(lambda x: 1.0 if (x > 0) else 0.0)
        logger.info("data is binarized")
        return ratings

    def _leave_last_out(self, ratings):
        """leave one out train/test split """
        logger.info("leaving last out")
        ratings["rank_latest"] = ratings.groupby(["userId"])["timestamp"].rank(
            method="first", ascending=False
        )
        test = ratings[ratings["rank_latest"] == 1]
        train = ratings[ratings["rank_latest"] > 1]
        assert train["userId"].nunique() == test["userId"].nunique()
        return (
            train[["userId", "itemId", "rating"]],
            test[["userId", "itemId", "rating"]],
        )

    def _sample_negatives_low_mem(self, ratings):
        """return all negative items & 100 sampled negative items"""
        logger.info("sampling negatives with low mem")
        # raitings = dd.from_pandas(ratings,npartitions=4)

        def sample_negatives_per_user(x):
            set_interacted = set(x)
            set_non_interacted = self.item_pool.difference(set_interacted)
            return random.sample(set_non_interacted, 99)

        interact_status = (
            ratings.groupby("userId")["itemId"]
            .apply(set)
            .reset_index()
            .rename(columns={"itemId": "interacted_items"})
        )
        # interact_status['negative_items'] = interact_status['interacted_items'].apply(
        #     lambda x: self.item_pool - x)
        interact_status["negative_samples"] = interact_status["interacted_items"].apply(
            sample_negatives_per_user
        )
        return interact_status[["userId", "negative_samples"]]

    def _sample_negatives(self, ratings):
        # from dask.distributed import LocalCluster, Client
        # cluster = LocalCluster(memory_limit='2GB', processes=False,
        #         n_workers=1, threads_per_worker=2)
        # client = Client(cluster)
        # logger.info(cluster)
        # import dask
        # import dask.dataframe as dd
        # from dask.diagnostics import ProgressBar
        # pbar = ProgressBar()
        # pbar.register()
        # To see where the port of the dashboard is, use this command
        # print(client.scheduler_info()['services'])
        # {'dashboard': 8787} --> means you can access it at localhost:8787
        # tqdm.pandas()
        """return all negative items & 100 sampled negative items"""
        logger.info("sampling negatives")
        # raitings = dd.from_pandas(ratings,npartitions=4)
        interact_status = (
            ratings.groupby("userId")["itemId"]
            .apply(set)
            .reset_index()
            .rename(columns={"itemId": "interacted_items"})
        )
        interact_status["negative_items"] = interact_status["interacted_items"].apply(
            lambda x: self.item_pool - x
        )
        interact_status["negative_samples"] = interact_status["negative_items"].apply(
            lambda x: random.sample(x, 99)
        )
        return interact_status[["userId", "negative_items", "negative_samples"]]

    def _prepare_epoch_low_mem(self, num_negatives, batch_size):
        tqdm.pandas()

        def sample_negatives_per_user_per_epoch(x):
            set_interacted = set(x)
            set_non_interacted = self.item_pool.difference(set_interacted)
            return random.sample(set_non_interacted, num_negatives)

        users, items, ratings = [], [], []
        # TODO: this is very bad to do it before every epoch, but, for now this makes it readable
        interact_status = (
            self.ratings.groupby("userId")["itemId"]
            .apply(set)
            .reset_index()
            .rename(columns={"itemId": "interacted_items"})
        )
        train_epoch_with_negatives = pd.merge(
            self.train_ratings,
            interact_status[["userId", "interacted_items"]],
            on="userId",
        )
        train_epoch_with_negatives[
            "epoch_sampled_negatives"
        ] = train_epoch_with_negatives["interacted_items"].apply(
            sample_negatives_per_user_per_epoch
        )
        for row in train_epoch_with_negatives.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.epoch_sampled_negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        logger.info("Epoch data is being prepared via pandas. Have a coffee")
        dataset = UserItemRatingDataset(
            user_tensor=torch.LongTensor(users),
            item_tensor=torch.LongTensor(items),
            target_tensor=torch.FloatTensor(ratings),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def _prepare_epoch(self, num_negatives, batch_size):
        tqdm.pandas()

        def sample_negatives_per_user_per_epoch(x):
            set_interacted = set(x)
            set_non_interacted = self.item_pool.difference(set_interacted)
            return random.sample(set_non_interacted, 99)

        users, items, ratings = [], [], []
        # TODO: this is very bad to do it before every epoch, but, for now this makes it readable
        train_epoch_with_negatives = pd.merge(
            self.train_ratings,
            self.negatives[["userId", "negative_items"]],
            on="userId",
        )
        train_epoch_with_negatives[
            "epoch_sampled_negatives"
        ] = train_epoch_with_negatives["negative_items"].apply(
            lambda x: random.sample(x, num_negatives)
        )
        logger.info("Epoch data is being prepared via pandas. Have a coffee")
        for row in train_epoch_with_negatives.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.epoch_sampled_negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = UserItemRatingDataset(
            user_tensor=torch.LongTensor(users),
            item_tensor=torch.LongTensor(items),
            target_tensor=torch.FloatTensor(ratings),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def evaluation_data_data_loader(self, batch_size):
        logger.info("preparing evaluation data")
        # TODO: this shoudl be unified in names:
        logger.info(
            "preparing evaluation data via dataloader, perhaps it shoudl be written to disk once and for all"
        )
        test_ratings = pd.merge(
            self.test_ratings,
            self.negatives[["userId", "negative_samples"]],
            on="userId",
        )
        (
            positive_interactions_users,
            test_items,
            negative_interactions_users,
            negative_items,
        ) = ([], [], [], [])
        for row in test_ratings.itertuples():
            positive_interactions_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            # It is worth describing what we mean under negative users here TODO:
            # why do we predict negative items with negative users in the evaluate epoch stage?
            for i in range(len(row.negative_samples)):
                negative_interactions_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))

        evaluation_dataset = EvaluationDataset(
            positive_interactions_users=torch.LongTensor(positive_interactions_users),
            test_items=torch.LongTensor(test_items),
            negative_interactions_users=torch.LongTensor(negative_interactions_users),
            negative_items=torch.LongTensor(negative_items),
        )
        return DataLoader(evaluation_dataset, batch_size=batch_size, shuffle=True)
        #  test_ratings, positive_interactions_users, test_items, negative_interactions_users, negative_items

    @property
    def evaluation_data(self):
        logger.info("preparing evaluation data")
        # TODO: this shoudl be unified in names:
        logger.info("preparing evaluation data. this is sloooow")
        test_ratings = pd.merge(
            self.test_ratings,
            self.negatives[["userId", "negative_samples"]],
            on="userId",
        )
        (
            positive_interactions_users,
            test_items,
            negative_interactions_users,
            negative_items,
        ) = ([], [], [], [])
        for row in test_ratings.itertuples():
            positive_interactions_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            # It is worth describing what we mean under negative users here TODO:
            # why do we predict negative items with negative users in the evaluate epoch stage?
            for i in range(len(row.negative_samples)):
                negative_interactions_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        return (
            torch.LongTensor(positive_interactions_users),
            torch.LongTensor(test_items),
            torch.LongTensor(negative_interactions_users),
            torch.LongTensor(negative_items),
        )
        #  test_ratings, positive_interactions_users, test_items, negative_interactions_users, negative_items
