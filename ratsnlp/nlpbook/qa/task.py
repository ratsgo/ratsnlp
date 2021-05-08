from transformers import PreTrainedModel
from transformers.optimization import AdamW
from ratsnlp.nlpbook.metrics import accuracy
from pytorch_lightning import LightningModule
from ratsnlp.nlpbook.qa import QATrainArguments
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts


class QATask(LightningModule):

    def __init__(self,
                 model: PreTrainedModel,
                 args: QATrainArguments,
    ):
        super().__init__()
        self.model = model
        self.args = args

    def configure_optimizers(self):
        if self.args.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.args.learning_rate)
        else:
            raise NotImplementedError('Only AdamW is Supported!')
        if self.args.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.args.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def training_step(self, inputs, batch_idx):
        # outputs: QuestionAnsweringModelOutput
        outputs = self.model(**inputs)
        start_preds = outputs.start_logits.argmax(dim=-1)
        start_positions = inputs["start_positions"]
        end_preds = outputs.end_logits.argmax(dim=-1)
        end_positions = inputs["end_positions"]
        acc = (accuracy(start_preds, start_positions) + accuracy(end_preds, end_positions)) / 2
        self.log("loss", outputs.loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("acc", acc, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return outputs.loss

    def validation_step(self, inputs, batch_idx):
        # outputs: QuestionAnsweringModelOutput
        outputs = self.model(**inputs)
        start_preds = outputs.start_logits.argmax(dim=-1)
        start_positions = inputs["start_positions"]
        end_preds = outputs.end_logits.argmax(dim=-1)
        end_positions = inputs["end_positions"]
        acc = (accuracy(start_preds, start_positions) + accuracy(end_preds, end_positions)) / 2
        self.log("val_loss", outputs.loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return outputs.loss
