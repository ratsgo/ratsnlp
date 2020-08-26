import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.optimization import AdamW
from ratsnlp.nlpbook.metrics import accuracy
from pytorch_lightning import LightningModule
from ratsnlp.nlpbook.ner.corpus import NER_PAD_ID
from ratsnlp.nlpbook.arguments import TrainArguments
from transformers import BertPreTrainedModel, BertModel
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts


class ModelForNER(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=NER_PAD_ID)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class NERTask(LightningModule):

    def __init__(self,
                 model: BertPreTrainedModel,
                 args: TrainArguments,
    ):
        super().__init__()
        self.model = model
        self.args = args
        self.running_accuracy = TensorRunningAccum(window_length=args.stat_window_length)

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

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def step(self, inputs, mode="train"):
        loss, logits = self.model(**inputs)
        preds = logits.argmax(dim=-1)
        labels = inputs["labels"]
        acc = accuracy(preds, labels, ignore_index=NER_PAD_ID)
        self.running_accuracy.append(acc)
        logs = {f"{mode}_loss": loss, f"{mode}_acc": acc}
        return {"loss": loss, "log": logs}

    def training_step(self, inputs, batch_idx):
        return self.step(inputs, mode="train")

    def validation_step(self, inputs, batch_idx):
        return self.step(inputs, mode="val")

    def test_step(self, inputs, batch_idx):
        return self.step(inputs, mode="test")

    def epoch_end(self, outputs, mode="train"):
        loss_mean, acc_mean = 0, 0
        for output in outputs:
            loss_mean += output['loss']
            acc_mean += output['log'][f'{mode}_acc']
        acc_mean /= len(outputs)
        results = {
            'log': {
                f'{mode}_loss': loss_mean,
                f'{mode}_acc': acc_mean,
            },
            'progress_bar': {f'{mode}_acc': acc_mean},
        }
        return results

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, mode="val")

    def test_epoch_end(self, outputs):
        return self.epoch_end(outputs, mode="test")

    def get_progress_bar_dict(self):
        running_train_loss = self.trainer.running_loss.mean()
        running_train_accuracy = self.running_accuracy.mean()
        tqdm_dict = {
            'tr_loss': '{:.3f}'.format(running_train_loss.cpu().item()),
            'tr_acc': '{:.3f}'.format(running_train_accuracy.cpu().item()),
        }
        return tqdm_dict
