from transformers.optimization import AdamW
from ratsnlp.nlpbook.metrics import accuracy
from pytorch_lightning import LightningModule
from transformers import BertPreTrainedModel
from ratsnlp.nlpbook.ner import NERTrainArguments, NER_PAD_ID
from torch.optim.lr_scheduler import ExponentialLR


class NERTask(LightningModule):

    def __init__(self,
                 model: BertPreTrainedModel,
                 args: NERTrainArguments,
    ):
        super().__init__()
        self.model = model
        self.args = args

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def training_step(self, inputs, batch_idx):
        # outputs: TokenClassifierOutput
        outputs = self.model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
        labels = inputs["labels"]
        acc = accuracy(preds, labels, ignore_index=NER_PAD_ID)
        self.log("loss", outputs.loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("acc", acc, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return outputs.loss

    def validation_step(self, inputs, batch_idx):
        # outputs: TokenClassifierOutput
        outputs = self.model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
        labels = inputs["labels"]
        acc = accuracy(preds, labels, ignore_index=NER_PAD_ID)
        self.log("val_loss", outputs.loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return outputs.loss
