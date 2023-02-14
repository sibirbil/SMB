from dataclasses import dataclass
from typing import Tuple, Dict, Optional

from .testoptimizers import TestOptimizer
from smbexperiments.utils.networks import get_model
from smbexperiments.utils.datatools import get_dataset
from smbexperiments.utils.metrics import compute_loss, compute_accuracy
from smbexperiments.utils import use_GPU
import time


@dataclass
class TrainingSetup:
    dataset_and_model: Tuple
    optimizer_with_config: Tuple[TestOptimizer, Dict]
    epochs: int
    batch_size: Optional[int] = None
    n_train_samples: Optional[int] = None

    def dataset(self):
        return self.dataset_and_model[0]

    def model(self):
        model=get_model(self.dataset_and_model[1])
        return model

    def organize_training_data(self):
        return get_dataset(dataset_name=self.dataset(),
                           batch_size=self.batch_size,
                           n_samples=self.n_train_samples)

    def test_optimizer(self):
        return self.optimizer_with_config[0]

    def set_optimizer_with_model_parameters(self, params, config):
        return self.test_optimizer().config_from_dict(
            params=params,
            data=config
        )

    def optimizer_configuration(self):
        return self.optimizer_with_config[1]

    def optimizer_name(self):
        test_optim = self.test_optimizer()
        independent_batch = self.optimizer_configuration().get('independent_batch')
        autoschedule = self.optimizer_configuration().get('autoschedule')

        _name = test_optim.name
        if independent_batch is not None:
            if independent_batch:
                _name = f"{_name}_i"

        if autoschedule is not None:
            if autoschedule:
                _name = f"{_name}_autosched"

        return _name

    def init_result_dict(self):
        dataset, model_name = self.dataset_and_model

        return {'data': dataset,
                'model': model_name,
                'name': self.optimizer_name()
                }

    def train_epoch_with_sls(self, model, optimizer, train_set, test_set, train_loader, criterion, train_iter_loss_list,
                             metric_lists: Dict):

        begin = time.time()

        # training steps
        model.train()

        for batch_index, (data, target) in enumerate(train_loader):

            # moves tensors to GPU
            if use_GPU:
                data, target = data.cuda(), target.cuda()

                # create loss closure for sls algorithm
            closure = lambda: criterion(model, data, target)
            # clears gradients
            optimizer.zero_grad()

            loss = optimizer.step(closure=closure)

            train_iter_loss_list.append(loss.item())

        end = time.time()

        train_loss = compute_loss(model, train_set)
        test_acc = compute_accuracy(model, test_set)

        self.update_evaluation_metrics(metric_lists=metric_lists,
                                       train_loss=train_loss,
                                       test_acc=test_acc,
                                       time_per_epoch=end - begin)

    def train_epoch_with_smb(self, model, train_set, test_set, train_loader, optimizer, criterion, metric_lists):

        begin = time.time()

        # training steps
        model.train()

        for batch_index, (data, target) in enumerate(train_loader):

            # moves tensors to GPU if available
            if use_GPU:
                data, target = data.cuda(), target.cuda()

                # create loss closure for smb algorithm

            def closure():
                optimizer.zero_grad()
                loss = criterion(model, data, target)
                return loss

            # forward pass
            optimizer.step(closure=closure)

        end = time.time()

        train_loss = compute_loss(model, train_set)
        test_acc = compute_accuracy(model, test_set)

        self.update_evaluation_metrics(metric_lists=metric_lists,
                                       train_loss=train_loss,
                                       test_acc=test_acc,
                                       time_per_epoch=end - begin)

    def train_epoch_with_adam(self, model, train_set, test_set, train_loader, optimizer, criterion, metric_lists):
        begin = time.time()

        # training steps
        model.train()
        for batch_index, (data, target) in enumerate(train_loader):

            # moves tensors to GPU
            if use_GPU:
                data, target = data.cuda(), target.cuda()
                # clears gradients
            optimizer.zero_grad()
            # loss in batch
            loss = criterion(model, data, target)
            # backward pass for loss gradient
            loss.backward()

            # update paremeters
            optimizer.step()

        end = time.time()

        # Calculate metrics
        train_loss = compute_loss(model, train_set)
        test_acc = compute_accuracy(model, test_set)
        self.update_evaluation_metrics(metric_lists=metric_lists,
                                       train_loss=train_loss,
                                       test_acc=test_acc,
                                       time_per_epoch=end - begin)

    def train_epoch_with_sgd(self, model, train_set, test_set, train_loader, optimizer, criterion, metric_lists):
        begin = time.time()

        # training steps
        model.train()
        for batch_index, (data, target) in enumerate(train_loader):

            # moves tensors to GPU
            if use_GPU:
                data, target = data.cuda(), target.cuda()
                # clears gradients
            optimizer.zero_grad()
            # loss in batch
            loss = criterion(model, data, target)
            # backward pass for loss gradient
            loss.backward()

            # update paremeters
            optimizer.step()

        end = time.time()

        # Calculate metrics
        train_loss = compute_loss(model, train_set)
        test_acc = compute_accuracy(model, test_set)
        self.update_evaluation_metrics(metric_lists=metric_lists,
                                       train_loss=train_loss,
                                       test_acc=test_acc,
                                       time_per_epoch=end - begin)

    @staticmethod
    def update_evaluation_metrics(metric_lists: Dict, train_loss, test_acc, time_per_epoch):
        metric_lists["train_loss_list"].append(train_loss)
        metric_lists["test_acc_list"].append(test_acc)
        metric_lists["run_time_list"].append(time_per_epoch)
