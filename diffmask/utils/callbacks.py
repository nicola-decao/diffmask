import torch
import pytorch_lightning as pl


class CallbackSST(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        print(
            "Epoch {}: Validation accuracy = {:.2f}, F1 = {:.2f}".format(
                trainer.callback_metrics["epoch"] + 1,
                trainer.callback_metrics["val_acc"] * 100,
                trainer.callback_metrics["val_f1"] * 100,
            )
        )


class CallbackSSTDiffMask(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        print(
            "Epoch {}: Validation accuracy = {:.2f}, F1 = {:.2f}, gates at zero = {:.2%}, constraint = {:.5f}".format(
                trainer.callback_metrics["epoch"] + 1,
                trainer.callback_metrics["val_acc"] * 100,
                trainer.callback_metrics["val_f1"] * 100,
                1 - trainer.callback_metrics["val_l0"],
                trainer.callback_metrics["val_loss_c"],
            )
        )


class CallbackSquadDiffMask(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        print(
            "Epoch {}: Validation accuracy = {:.2f}, gates at zero = {:.2%}, constraint = {:.5f}".format(
                trainer.callback_metrics["epoch"] + 1,
                trainer.callback_metrics["val_acc"] * 100,
                1 - trainer.callback_metrics["val_l0"],
                trainer.callback_metrics["val_loss_c"],
            )
        )


class CallbackToyTask(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        print(
            "Epoch {}: Validation accuracy = {:.2f}".format(
                trainer.callback_metrics["epoch"] + 1,
                trainer.callback_metrics["val_acc"],
            )
        )


class CallbackToyTaskDiffMask(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        print(
            "Epoch {}: Validation accuracy = {:.2f}, gates at zero = {:.2%}, constraint = {:.5f}".format(
                trainer.callback_metrics["epoch"] + 1,
                trainer.callback_metrics["val_acc"],
                1 - trainer.callback_metrics["val_l0"],
                trainer.callback_metrics["val_loss_c"],
            )
        )
