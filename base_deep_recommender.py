import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from typing import DefaultDict
import logging
import datetime
from eval_metrics import RecoMetrics
from statistics import mean

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseDeepRecommender:
    def __init__(self, config: DefaultDict):
        logger.info("Base Deep Recommender got the config {}".format(config))
        self._config = config
        self._evaluator = RecoMetrics(top_k=30)
        # self.model = None
        self._writer = SummaryWriter(
            log_dir="runs/{}".format(
                config["alias"]
                + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
        )
        self._writer.add_text("config", str(config), 0)
        self.opt = BaseDeepRecommender.get_optimizer(self.model, config)
        self.criterion = torch.nn.BCELoss()
        self.criterion = self.criterion.cuda()
        # TODO: I'm not sure that it should be here
        # TODO: I'm not sure where i have to close the

    def train_batch(self, users, items, ratings):
        assert hasattr(self, "model"), "We need some model to train, please load/define"
        if self._config["use_cuda"] and torch.cuda.is_available():
            logger.debug("moving users, items, ratings to cuda")
            users, items, ratings = (
                users.cuda(non_blocking=True),
                items.cuda(non_blocking=True),
                ratings.cuda(non_blocking=True),
            )
        self.opt.zero_grad()
        ratings_predicted = self.model(users, items)
        # view(-1) is simplier than flatten()
        loss = self.criterion(ratings_predicted.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        del users, items, ratings
        return loss

    def train_epoch(self, data_loader, epoch_num):
        logger.info("starting training Epoch {}".format(epoch_num))
        assert hasattr(self, "model"), "We need some model to train, please load/define"
        logger.info("self.model.num_users is {}".format(self.model.num_users))
        self.model.train()
        epoch_loss = 0.0
        for batch_idx, batch_data in enumerate(data_loader):
            # those are the types that should be returned by the DataLoader
            # assert isinstance(batch_data[0], torch.LongTensor)
            # assert isinstance(batch_data[1], torch.LongTensor)
            # assert isinstance(batch_data[2], torch.FloatTensor)
            batch_loss = self.train_batch(batch_data[0], batch_data[1], batch_data[2])
            if batch_idx % 500 == 0:
                logger.info(
                    "Epoch {} has batch {} loss {}".format(
                        epoch_num, batch_idx, batch_loss
                    )
                )
            epoch_loss += batch_loss
        self._writer.add_scalar("model/epoch_loss", epoch_loss, epoch_num)
        self._writer.flush()
        del batch_data

    def evaluate_epoch(self, eval_data, epoch_num):
        assert hasattr(self, "model"), "We need some model to train, please load/define"

        self.model.eval()
        # let us also disable autograd engine to speed things up and reduce memory footprint
        epoch_hit_ratio = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(eval_data):
                # TODO: this is ridiculous piece of shit
                test_users, test_items, negative_interactions_users, negative_items = (
                    batch_data[0],
                    batch_data[1],
                    batch_data[2],
                    batch_data[3],
                )
                # TODO: I have to do normal batch loader.
                if self._config["use_cuda"]:
                    test_users = test_users.cuda(non_blocking=True)
                    test_items = test_items.cuda(non_blocking=True)
                    negative_interactions_users = negative_interactions_users.cuda(
                        non_blocking=True
                    )
                    negative_items = negative_items.cuda(non_blocking=True)
                    self.model.cuda()
                elif not self._config["use_cuda"]:
                    test_users = test_users.cpu()
                    test_items = test_items.cpu()
                    negative_users = negative_users.cpu()
                    negative_items = negative_items.cpu()
                    self.model.cpu()
                test_interactions_scores = self.model(test_users, test_items)
                negative_interactions_scores = self.model(
                    negative_interactions_users, negative_items
                )
                # TODO: TODO, TODO : this is TERIBLY SLOW ALSO
                self._evaluator.set_interactions(
                    test_users.view(-1).tolist(),
                    test_items.view(-1).tolist(),
                    test_interactions_scores.view(-1).tolist(),
                    negative_interactions_users.view(-1).tolist(),
                    negative_items.view(-1).tolist(),
                    negative_interactions_scores.view(-1).tolist(),
                )
                hit_ratio = self._evaluator.cal_hit_ratio()
                logger.info("eval batch value HR is {}".format(hit_ratio))
                epoch_hit_ratio.append(hit_ratio)
            hit_ratio_avg = mean(epoch_hit_ratio)
        self._writer.add_scalar("performance/HR", hit_ratio_avg, epoch_num)
        self._writer.flush()
        logger.info("[Evluating Epoch {}] HR = {:.4f}".format(epoch_num, hit_ratio))

        return hit_ratio

    # def evaluate_epoch(self, eval_data, epoch_num):
    #     assert hasattr(self, "model"), "We need some model to train, please load/define"

    #     self.model.eval()
    #     # let us also disable autograd engine to speed things up and reduce memory footprint
    #     with torch.no_grad():
    #         # TODO: this is ridiculous piece of shit
    #         test_users, test_items, negative_interactions_users, negative_items = (
    #             eval_data[0][:10000],
    #             eval_data[1][:10000],
    #             eval_data[2][:10000],
    #             eval_data[3][:10000],
    #         )
    #         # TODO: I have to do normal batch loader.
    #         if self._config["use_cuda"]:
    #             test_users = test_users.cuda(non_blocking=True)
    #             test_items = test_items.cuda(non_blocking=True)
    #             negative_interactions_users = negative_interactions_users.cuda(
    #                 non_blocking=True
    #             )
    #             negative_items = negative_items.cuda(non_blocking=True)
    #             self.model.cuda()
    #         elif not self._config["use_cuda"]:
    #             test_users = test_users.cpu()
    #             test_items = test_items.cpu()
    #             negative_users = negative_users.cpu()
    #             negative_items = negative_items.cpu()
    #             self.model.cpu()
    #         test_interactions_scores = self.model(test_users, test_items)
    #         negative_interactions_scores = self.model(
    #             negative_interactions_users, negative_items
    #         )
    #         # TODO: TODO, TODO : this is TERIBLY SLOW ALSO
    #         self._evaluator.set_interactions(
    #             test_users.view(-1).tolist(),
    #             test_items.view(-1).tolist(),
    #             test_interactions_scores.view(-1).tolist(),
    #             negative_interactions_users.view(-1).tolist(),
    #             negative_items.view(-1).tolist(),
    #             negative_interactions_scores.view(-1).tolist(),
    #         )
    #     hit_ratio = self._evaluator.cal_hit_ratio()
    #     self._writer.add_scalar("performance/HR", hit_ratio, epoch_num)
    #     self._writer.flush()
    #     logger.info("[Evluating Epoch {}] HR = {:.4f}".format(epoch_num, hit_ratio))

    #     return hit_ratio

    def persist_model(self, alias, epoch_num, hit_ratio):
        assert hasattr(self, "model"), "I can not save model when there is no model!"
        model_dir = self._config["model_dir"].format(alias, epoch_num, hit_ratio)
        torch.save(self.model, model_dir)

    @staticmethod
    def get_optimizer(model, params):
        assert model is not None, "We need some model to train, please load/define"

        if params["optimizer"] == "sgd":
            raise NotImplementedError
            # optimizer = torch.optim.SGD(network.parameters(),
            #                             lr=params['sgd_lr'],
            #                             momentum=params['sgd_momentum'],
            #                             weight_decay=params['l2_regularization'])
        elif params["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(params["adam_lr"]),
                weight_decay=float(params["l2_regularization"]),
            )
        elif params["optimizer"] == "rmsprop":
            raise NotImplementedError
            # optimizer = torch.optim.RMSprop(network.parameters(),
            #                                 lr=params['rmsprop_lr'],
            #                                 alpha=params['rmsprop_alpha'],
            #                                 momentum=params['rmsprop_momentum'])
        return optimizer

    @staticmethod
    def use_cuda(enabled=True):
        if enabled:
            assert torch.cuda.is_available(), "CUDA is not available"
            torch.cuda.set_device(0)
            # torch.set_default_tensor_type('torch.cuda.LongTensor')
