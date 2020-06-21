import torch
from copy import deepcopy
import random
import pandas as pd
from tqdm import tqdm
import sys
# import os
# os.environ["MODIN_ENGINE"] = "ray"
# import modin.pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        # TODO: This is plain wrong since the dimentions are different
        return (self.positive_interactions_users[index],
                self.test_items[index],
                self.negative_interactions_users[index],
                self.negative_items[index])

    def __len__(self):
        return self.positive_interactions_users.size(0)


class EvaluationDatasetV2(Dataset):
    def __init__(self, test_ratings: pd.DataFrame):
        # TODO. do i have to make a copy here?
        self._test_ratings = test_ratings.copy()
        logger.debug('we are in the EvalDataSetV2 %s',
                     self._test_ratings.head())
        logger.debug('what is the shape of the self._test_ratings %s',
                     self._test_ratings.shape)
        logger.debug('do we have an index %s', self._test_ratings.index)
        self._positive_interactions_users = []
        self._test_items = []
        self._negative_interactions_users = []
        self._negative_items = []
        logger.debug("positive interactions users should be an empty array".format(
            self._positive_interactions_users))

        # persist dataframe
        #self._test_ratings.to_pickle("misc/_test_ratings.pkl")

    def __getitem__(self, index):
        row = self._test_ratings.iloc[index]
        return (torch.LongTensor([row.userId]),
                torch.LongTensor([row.itemId]),
                torch.LongTensor(row.negative_samples))

    def __len__(self):
        return self._test_ratings.shape[0]


class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor that corresponds to (user, item) pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        """
        Map-style datasets https://pytorch.org/docs/stable/data.html#map-style-datasets
        """
        return (self.user_tensor[index], self.item_tensor[index],
                self.target_tensor[index])

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
            self.preprocess_ratings)

    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        logger.info("binarizing data")
        # ratings.rating.applymap(lambda x: 1.0 if (x>0) else 0.0)
        # ratings = deepcopy(ratings)
        # ratings['rating'][ratings['rating'] > 0] = 1.0
        ratings["rating"] = ratings.rating.map(lambda x: 1.0
                                               if (x > 0) else 0.0)
        logger.info("data is binarized")
        return ratings

    def _leave_last_out(self, ratings):
        """leave one out train/test split """
        logger.info("leaving last out")
        ratings["rank_latest"] = ratings.groupby(["userId"])["timestamp"].rank(
            method="first", ascending=False)
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

        interact_status = (ratings.groupby("userId")["itemId"].apply(
            set).reset_index().rename(columns={"itemId": "interacted_items"}))
        # interact_status['negative_items'] = interact_status['interacted_items'].apply(
        #     lambda x: self.item_pool - x)
        interact_status["negative_samples"] = interact_status[
            "interacted_items"].apply(sample_negatives_per_user)
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
        interact_status = (ratings.groupby("userId")["itemId"].apply(
            set).reset_index().rename(columns={"itemId": "interacted_items"}))
        interact_status["negative_items"] = interact_status[
            "interacted_items"].apply(lambda x: self.item_pool - x)
        interact_status["negative_samples"] = interact_status[
            "negative_items"].apply(lambda x: random.sample(x, 99))
        return interact_status[[
            "userId", "negative_items", "negative_samples"
        ]]

    def _prepare_epoch_low_mem(self, num_negatives, batch_size):
        tqdm.pandas()

        def sample_negatives_per_user_per_epoch(x):
            set_interacted = set(x)
            set_non_interacted = self.item_pool.difference(set_interacted)
            return random.sample(set_non_interacted, num_negatives)

        users, items, ratings = [], [], []
        # TODO: this is very bad to do it before every epoch, but, for now this makes it readable
        interact_status = (self.ratings.groupby("userId")["itemId"].apply(
            set).reset_index().rename(columns={"itemId": "interacted_items"}))
        train_epoch_with_negatives = pd.merge(
            self.train_ratings,
            interact_status[["userId", "interacted_items"]],
            on="userId",
        )
        train_epoch_with_negatives[
            "epoch_sampled_negatives"] = train_epoch_with_negatives[
                "interacted_items"].apply(sample_negatives_per_user_per_epoch)
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
            "epoch_sampled_negatives"] = train_epoch_with_negatives[
                "negative_items"].apply(
                    lambda x: random.sample(x, num_negatives))
        logger.info("Epoch data is being prepared via pandas. Have a coffee")
        # TODO: This is the slowest part of the whole pipeline,
        # and it is _not_ the training.
        # This MUSt be somehow extracted to the Dataset() part, but
        # HOW do I handle the random sampling of negatives in every epoch?
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
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    def evaluation_data_loader_v2(self, batch_size):
        logger.info("preparing evaluation data")
        # TODO: this shoudl be unified in names:
        logger.info(
            "preparing evaluation data via dataloader v2, perhaps it shoudl be written to disk once and for all"
        )
        test_ratings = pd.merge(self.test_ratings,
                                self.negatives[["userId", "negative_samples"]],
                                on="userId")
        # logger.info(test_ratings.head())
        # positive_interactions_users, test_items, negative_interactions_users, negative_items = [], [], [], []
        # logger.info("check if i have correct positive inter users {}".format(
        #     positive_interactions_users))

        evaluation_dataset = EvaluationDatasetV2(test_ratings)

        return DataLoader(evaluation_dataset,
                          batch_size=batch_size,
                          shuffle=True, num_workers=4)

    def evaluation_data_data_loader(self, batch_size):
        logger.info("preparing evaluation data")
        # TODO: this shoudl be unified in names:
        logger.info(
            "preparing evaluation data via dataloader, perhaps it shoudl be written to disk once and for all"
        )
        test_ratings = pd.merge(self.test_ratings,
                                self.negatives[["userId", "negative_samples"]],
                                on="userId")
        logger.info(test_ratings.head())
        positive_interactions_users, test_items, negative_interactions_users, negative_items = [], [], [], []
        logger.info("check if i have correct positive inter users {}".format(
            positive_interactions_users))
        ####
        # In reality i am not sure that this should be done here
        # we can
        # 1. extract creation of the test_ratings to the top (if it is not epoch dependent , must check here)
        # theretically negative samples should be epoch dep in order not to overfit to those
        # 2. when we do have a join we can save it as a tensor and pass out in batches
        # 3. further processing can be done on that

        # this logic for creation of tensors should be moved to `evaluate_epoch()` function
        for row in test_ratings.itertuples():
            positive_interactions_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            # It is worth describing what we mean under negative users here TODO:
            # why do we predict negative items with negative users in the evaluate epoch stage?
            for i in range(len(row.negative_samples)):
                negative_interactions_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))

        logger.info("positive inter users shpae {}".format(
            len(positive_interactions_users)))
        logger.info("test users shpae {}".format(len(test_items)))
        logger.info("negative_interactions_users inter users shpae {}".format(
            len(negative_interactions_users)))
        logger.info("negative items shape {}".format(len(negative_items)))

        ############################
        # TODO: there is a bug here, since i am not correctly supplying data of different lneght to the DataLoader as
        # evaluation dataset
        ############################

        # here i will save those for debugging purposes
        #torch.save(tensor, 'file.pt') and torch.load('file.pt')
        torch.save(torch.LongTensor(positive_interactions_users),
                   'misc/positive_interactions_users.pt')
        torch.save(torch.LongTensor(test_items), 'misc/test_items.pt')
        torch.save(torch.LongTensor(negative_interactions_users),
                   'misc/negative_interactions_users.pt')
        torch.save(torch.LongTensor(negative_items), 'misc/negative_items.pt')
        sys.exit()

        evaluation_dataset = EvaluationDataset(
            positive_interactions_users=torch.LongTensor(
                positive_interactions_users),
            test_items=torch.LongTensor(test_items),
            negative_interactions_users=torch.LongTensor(
                negative_interactions_users),
            negative_items=torch.LongTensor(negative_items))
        return DataLoader(evaluation_dataset,
                          batch_size=len(evaluation_dataset),
                          shuffle=False)
        #  test_ratings, positive_interactions_users, test_items, negative_interactions_users, negative_items

    @property
    def evaluation_data(self):
        logger.info("preparing evaluation data")
        # TODO: this shoudl be unified in names:
        logger.info("preparing evaluation data. this is sloooow")
        test_ratings = pd.merge(self.test_ratings,
                                self.negatives[["userId", "negative_samples"]],
                                on="userId")

        positive_interactions_users, test_items, negative_interactions_users, negative_items = [], [], [], []
        for row in test_ratings.itertuples():
            positive_interactions_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            # It is worth describing what we mean under negative users here TODO:
            # why do we predict negative items with negative users in the evaluate epoch stage?
            for i in range(len(row.negative_samples)):
                negative_interactions_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        return (torch.LongTensor(positive_interactions_users),
                torch.LongTensor(test_items),
                torch.LongTensor(negative_interactions_users),
                torch.LongTensor(negative_items))
        #  test_ratings, positive_interactions_users, test_items, negative_interactions_users, negative_items
