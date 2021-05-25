from transformers import PreTrainedModel
from transformers.optimization import AdamW
from pytorch_lightning import LightningModule
from ratsnlp.nlpbook.generation.arguments import GenerationTrainArguments
from torch.optim.lr_scheduler import ExponentialLR


class GenerationTask(LightningModule):

    def __init__(self,
                 model: PreTrainedModel,
                 args: GenerationTrainArguments,
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
        # outputs: CausalLMOutputWithCrossAttentions
        outputs = self.model(**inputs)
        self.log("loss", outputs.loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return outputs.loss

    def validation_step(self, inputs, batch_idx):
        # outputs: CausalLMOutputWithCrossAttentions
        outputs = self.model(**inputs)
        self.log("val_loss", outputs.loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return outputs.loss
