import os
import sys
import time
from utils.utils import calculate_screw_accuracy

import matplotlib
import numpy as np
import torch
from tensorboardX import SummaryWriter

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


class ModelTrainer(object):
    def __init__(self, **kwargs):

        super(ModelTrainer, self).__init__()
        self.model = kwargs.get("model")
        self.model_type = kwargs.get("model_type")
        self.trainloader = kwargs.get("train_loader")
        self.testloader = kwargs.get("test_loader")
        self.optimizer = kwargs.get("optimizer")
        self.scheduler = kwargs.get("scheduler")
        self.criterion = kwargs.get("criterion")
        self.epochs = kwargs.get("epochs")
        self.name = kwargs.get("name")
        self.test_freq = kwargs.get("test_freq")
        self.ndof = kwargs.get("ndof")
        self.grad_th = kwargs.get("grad_th")
        self.training_stage = kwargs.get("training_stage")

        self.losses = []
        self.tlosses = []

        # float model as push to GPU/CPU
        self.device = kwargs.get("device")
        self.model.float().to(self.device)
        self.wts_dir = os.path.join(os.path.abspath("."), "saved", "models")
        os.makedirs(self.wts_dir, exist_ok=True)

        # plots dir
        self.plots_dir = os.path.join(
            os.path.abspath("."), "saved", kwargs.get("plots_dir"), self.name
        )
        os.makedirs(self.plots_dir, exist_ok=True)

        # Tensorboard
        logs_dir = os.path.join(os.path.abspath("."), "saved", kwargs.get("logs_dir"))
        os.makedirs(logs_dir, exist_ok=True)
        self.writer = SummaryWriter(logs_dir + self.name)
        # self.writer.add_graph(self.model, self.trainloader)

    def train(self):
        best_tloss = 1e8

        for epoch in range(self.epochs + 1):
            # sys.stdout.flush()
            loss = self.train_epoch(epoch)
            self.losses.append(loss)
            self.writer.add_scalar("Loss/train", loss, epoch)

            if epoch % self.test_freq == 0:
                tloss = self.test_epoch(epoch)
                self.tlosses.append(tloss)
                self.plot_losses()
                self.writer.add_scalar("Loss/validation", tloss, epoch)

                if tloss < best_tloss:
                    print("saving model.")
                    net_fname = os.path.join(self.wts_dir, str(self.name) + ".pt")
                    # torch.save(self.model.state_dict(), net_fname)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.scheduler.state_dict(),
                            "loss": tloss,
                        },
                        net_fname,
                    )
                    best_tloss = tloss

                self.scheduler.step(np.mean(self.tlosses))
            # self.scheduler.step()

            # Visualize gradients
            total_norm = 0.0
            nan_count = 0
            for tag, parm in self.model.named_parameters():
                if torch.isnan(parm.grad).any():
                    print("Encountered NaNs in gradients at {} layer".format(tag))
                    nan_count += 1
                else:
                    self.writer.add_histogram(tag, parm.grad.data.cpu().numpy(), epoch)
                    param_norm = parm.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2

            total_norm = total_norm ** (1.0 / 2)
            self.writer.add_scalar("Gradient/2-norm", total_norm, epoch)
            if nan_count > 0:
                raise ValueError("Encountered NaNs in gradients")

        # plot losses one more time
        self.plot_losses()
        # re-load the best state dictionary that was saved earlier.
        # self.model.load_state_dict(torch.load(net_fname, map_location="cpu"))
        checkpoint = torch.load(net_fname)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json("./all_scalars.json")
        self.writer.close()
        return self.model

    def train_epoch(self, epoch):
        start = time.time()
        running_loss = 0
        batches_per_dataset = (
            len(self.trainloader.dataset) / self.trainloader.batch_size
        )
        self.model.train()  # Put model in training mode

        for i, X in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            depth, labels = X["depth"].to(self.device), X["label"].to(self.device)
            y_pred = self.model(depth)
            loss = self.criterion(
                target=labels, prediction=y_pred, training_stage=self.training_stage
            )
            if loss.data == -float("inf"):
                print("inf loss caught, not backpropping")
                running_loss += -1000
            else:
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_th)
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.)

                self.optimizer.step()
                running_loss += loss.item()

        stop = time.time()
        print(
            "Epoch %s -  Train  Loss: %.5f Time: %.5f"
            % (str(epoch).zfill(3), running_loss / batches_per_dataset, stop - start)
        )
        return running_loss / batches_per_dataset

    def test_epoch(self, epoch):
        start = time.time()
        running_loss = 0
        screw_err = np.array([0.0] * 5, dtype=np.float32)
        batches_per_dataset = len(self.testloader.dataset) / self.testloader.batch_size
        self.model.eval()  # Put batch norm layers in eval mode
        with torch.no_grad():
            for i, X in enumerate(self.testloader):
                depth, labels = X["depth"].to(self.device), X["label"].to(self.device)
                y_pred = self.model(depth)
                loss = self.criterion(
                    target=labels, prediction=y_pred, training_stage=self.training_stage
                )
                running_loss += loss.item()

                ## Accuracy
                screw_err += (
                    torch.tensor(
                        calculate_screw_accuracy(
                            target=labels, pred=y_pred, model_type=self.model_type
                        )
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

        stop = time.time()
        print(
            "Epoch %s -  Test  Loss: %.5f Euc. Time: %.5f"
            % (str(epoch).zfill(3), running_loss / batches_per_dataset, stop - start)
        )

        screw_err /= batches_per_dataset
        print(
            "  Screw loss: Ori:{:.4f}  Dist:{:.4f}  th:{:.4f}  d:{:.4f}  ortho:{:.4f}".format(
                screw_err[0], screw_err[1], screw_err[2], screw_err[3], screw_err[4]
            )
        )

        self.writer.add_scalar("Metric/Screw/Orientation", screw_err[0], epoch)
        self.writer.add_scalar("Metric/Screw/Distance", screw_err[1], epoch)
        self.writer.add_scalar("Metric/Screw/Config-Theta", screw_err[2], epoch)
        self.writer.add_scalar("Metric/Screw/Config-d", screw_err[3], epoch)
        self.writer.add_scalar("Metric/Screw/Orthogonality", screw_err[4], epoch)

        return running_loss / batches_per_dataset

    def plot_losses(self):
        x = np.arange(len(self.losses))
        tx = np.arange(0, len(self.losses), self.test_freq)
        plt.plot(x, np.array(self.losses), color="b", label="train")
        plt.plot(tx, np.array(self.tlosses), color="r", label="test")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.plots_dir, "curve.png"))
        plt.close()
        np.save(os.path.join(self.plots_dir, "losses.npy"), np.array(self.losses))
        np.save(os.path.join(self.plots_dir, "tlosses.npy"), np.array(self.tlosses))

    def plot_grad_flow(self, named_parameters):
        """Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
        from link: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063
        """
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend(
            [
                Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4),
            ],
            ["max-gradient", "mean-gradient", "zero-gradient"],
        )
        plt.savefig(os.path.join(self.plots_dir, "grad_flow.png"))
