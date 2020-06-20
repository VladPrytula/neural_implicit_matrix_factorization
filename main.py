import pandas as pd
import os
import sys

# os.environ["MODIN_ENGINE"] = "ray"
# import modin.pandas as pd
import numpy as np
import yaml
from pathlib import Path
from collections import defaultdict
from data_processor import SampleGenerator
from gimf import GIMFDeepRecommender
from mlp import MLPDeepRecommender
from neu_mf import NeuIMFDeepRecommender
import logging
from tqdm import tqdm
import torch

# from pandarallel import pandarallel
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_zpls_data(data_dir, MIN_RATINGS=4):
    ml_dir = data_dir
    sales_df = pd.read_csv(spain_sales_raw,
                           parse_dates=True,
                           dtype={
                               "customer_id": int,
                               "product_id": int
                           })
    sales_df["rating"] = 1.0
    sales_df.date = pd.to_datetime(sales_df.date)
    sales_df.rename(columns={
        "customer_id": "uid",
        "product_id": "mid",
        "date": "timestamp"
    },
        inplace=True)
    df = sales_df

    # first let us filter out the users with less than MIN_RATINGS interations
    logger.info(
        "Filtering out users with less than {} ratings".format(MIN_RATINGS))
    grouped = df.groupby("uid")
    df = grouped.filter(lambda x: len(x) >= MIN_RATINGS).copy()

    # now let us factoriyze (re-index users)
    logger.info("Mapping original user and item IDs to new sequential IDs")
    df["userId"] = pd.factorize(df["uid"])[0]
    df["itemId"] = pd.factorize(df["mid"])[0]

    logger.info("Range of userId is [{}, {}]".format(df.userId.min(),
                                                     df.userId.max()))
    logger.info("Range of itemId is [{}, {}]".format(df.userId.min(),
                                                     df.itemId.max()))

    num_users = len(df["userId"].unique())
    num_items = len(df["itemId"].unique())
    logger.info("num_users is {}, num_items is {}".format(
        num_users, num_items))

    return df, num_users, num_items


def load_data(data_dir, MIN_RATINGS=4):
    ml_dir = data_dir
    df = pd.read_csv(ml_dir,
                     sep="::",
                     header=None,
                     names=["uid", "mid", "rating", "timestamp"],
                     engine="python")

    # first let us filter out the users with less than MIN_RATINGS interations
    logger.info(
        "Filtering out users with less than {} ratings".format(MIN_RATINGS))
    grouped = df.groupby("uid")
    df = grouped.filter(lambda x: len(x) >= MIN_RATINGS)

    # now let us factoriyze (re-index users)
    logger.info("Mapping original user and item IDs to new sequential IDs")
    df["userId"] = pd.factorize(df["uid"])[0]
    df["itemId"] = pd.factorize(df["mid"])[0]

    logger.info("Range of userId is [{}, {}]".format(df.userId.min(),
                                                     df.userId.max()))
    logger.info("Range of itemId is [{}, {}]".format(df.userId.min(),
                                                     df.itemId.max()))

    num_users = len(df["userId"].unique())
    num_items = len(df["itemId"].unique())
    logger.info("num_users is {}, num_items is {}".format(
        num_users, num_items))

    return df, num_users, num_items


if __name__ == "__main__":
    # pandarallel.initialize()
    config_dir = Path("./config/")
    config_location = config_dir / "config.yml"
    c_list = defaultdict()
    with open(config_location) as file:
        c_list = yaml.load(file, Loader=yaml.FullLoader)
    gmf_config = c_list["gmf"]
    mlp_config = c_list["mlp"]
    neu_mf_conifg = c_list["neumf"]
    # gmf_config  # neu_mf_conifg  # mlp_config  # neu_mf_conifg
    current_config = mlp_config
    logger.debug("curretn config is {}".format(current_config))

    ml1m_dir = "/data/ml-1m/ratings.dat"
    RAW_DATA_FOLDER = Path("data/raw/")
    PROCESSED_DATA_FOLDER = Path("data/processed/")
    spain_sales_raw = RAW_DATA_FOLDER / "spanish_sales.csv"
    # data, num_users, num_items = load_zpls_data(spain_sales_raw)
    data, num_users, num_items = load_data(ml1m_dir)

    # prepare data for training
    sample_generator = SampleGenerator(ratings=data)
    evaluation_data = sample_generator.evaluation_data

    logger.info("eval data is ready")
    # print(len(evaluation_data))
    # print(len(evaluation_data[0]))
    # print(type(evaluation_data))
    # # deep_recommender = NeuIMFDeepRecommender(
    #     config=current_config, num_users=num_users, num_items=num_items)
    # training
    # for epoch in range(mlp_config['num_epochs']):
    #     logger.info("Epoch {}".format(epoch))
    #     train_data = sample_generator._prepare_epoch(
    #         mlp_config['num_negatives'], mlp_config['batch_size'])
    #     deep_recommender.train_epoch(train_data, epoch_num=epoch)
    #     hit_ratio = deep_recommender.evaluate_epoch(evaluation_data, epoch_num=epoch)
    #     deep_recommender.persist_model(mlp_config['alias'], epoch, hit_ratio)

    # deep_recommender = NeuIMFDeepRecommender(
    #     config=current_config, num_users=num_users, num_items=num_items
    # )

    deep_recommender = MLPDeepRecommender(
        config=current_config, num_users=num_users, num_items=num_items)
    # deep_recommender = GIMFDeepRecommender(config=current_config,
    #                                        num_users=num_users,
    #                                        num_items=num_items)
    dummy_users = torch.tensor([1, 2, 3]).cuda()
    dummy_items = torch.tensor([1, 2, 3]).cuda()
    deep_recommender._writer.add_graph(deep_recommender.model,
                                       (dummy_users, dummy_items))

    eval_data_loader_v2 = sample_generator.evaluation_data_loader_v2(
        batch_size=current_config["batch_size"])
    # training
    # for epoch in range(gmf_config['num_epochs']):
    #     logger.info("Epoch {}".format(epoch))
    #     train_data = sample_generator._prepare_epoch(
    #         gmf_config['num_negatives'], gmf_config['batch_size'])
    #     deep_recommender.train_epoch(train_data, epoch_num=epoch)
    #     hit_ratio = deep_recommender.evaluate_epoch(
    #         evaluation_data, epoch_num=epoch)
    #     deep_recommender.persist_model(gmf_config['alias'], epoch, hit_ratio)
    for epoch in range(current_config["num_epochs"]):
        logger.info("Epoch {}".format(epoch))
        # train_data = sample_generator._prepare_epoch(
        #     current_config["num_negatives"], current_config["batch_size"])

        # sys.exit()

        train_data = sample_generator._prepare_epoch_low_mem(
            current_config["num_negatives"], current_config["batch_size"])
        deep_recommender.train_epoch(train_data, epoch_num=epoch)
        # i want to use the batchced eval data loader to prevetn out of mem
        # hit_ratio = deep_recommender.evaluate_epoch(evaluation_data, epoch_num=epoch)

        # evaluation_data_loader = sample_generator.evaluation_data_data_loader(
        #     batch_size=current_config["batch_size"] * 10)

        hit_ratio = deep_recommender.evaluate_epoch(
            eval_data_loader_v2,
            epoch_num=epoch
        )
        deep_recommender.persist_model(current_config["alias"], epoch,
                                       hit_ratio)
    torch.cuda.empty_cache()
