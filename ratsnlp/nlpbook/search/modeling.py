import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


def _cosine_scores(questions: torch.Tensor, answers: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    questions: n1 x D, answers: n2 x D, result n1 x n2
    pairwise cosine similarity matrix 계산
    이 행렬의 대각성분은 각각 q0-a0, q1-a1, ... 사이의 코사인 유사도가 된다
    이는 F.cosine_similarity(questions, answers, dim=1)와 일치한다
    """
    questions_l2_norm = questions.norm(p=2, dim=1, keepdim=True)
    answers_l2_norm = answers.norm(p=2, dim=1, keepdim=True)
    numerator = torch.matmul(questions, torch.transpose(answers, 0, 1))
    denominator = (questions_l2_norm * torch.transpose(answers_l2_norm, 0, 1)).clamp(min=eps)
    return numerator / denominator


class SearchModelForTrain(torch.nn.Module):

    def __init__(
            self,
            question_tower: PreTrainedModel,
            passage_tower: PreTrainedModel,
            score_fn=_cosine_scores,
    ):
        super().__init__()
        self.question_tower = question_tower
        self.passage_tower = passage_tower
        self.score_fn = score_fn

    def forward(
            self,
            question_input_ids=None,
            question_attention_mask=None,
            question_token_type_ids=None,
            passage_input_ids=None,
            passage_attention_mask=None,
            passage_token_type_ids=None,
            labels=None,
    ):
        question_embeddings = self.question_tower(
            input_ids=question_input_ids,
            token_type_ids=question_token_type_ids,
            attention_mask=question_attention_mask,
        )[1]
        passage_embeddings = self.passage_tower(
            input_ids=passage_input_ids,
            token_type_ids=passage_attention_mask,
            attention_mask=passage_token_type_ids,
        )[1]
        scores = self.score_fn(question_embeddings, passage_embeddings)
        softmax_scores = F.log_softmax(scores, dim=1)
        loss = F.nll_loss(softmax_scores, labels, reduction='mean')
        return loss, softmax_scores


class SearchModelForInference(torch.nn.Module):

    def __init__(
            self,
            question_tower: PreTrainedModel,
            passage_tower: PreTrainedModel,
            score_fn=_cosine_scores,
    ):
        super().__init__()
        self.question_tower = question_tower
        self.passage_tower = passage_tower
        self.score_fn = score_fn

    def forward(
            self,
            question_input_ids=None,
            question_attention_mask=None,
            question_token_type_ids=None,
            passage_input_ids=None,
            passage_attention_mask=None,
            passage_token_type_ids=None,
            passage_embeddings=None,
            mode="inference",
    ):
        if mode == "inference":
            question_embeddings = self.question_tower(
                input_ids=question_input_ids,
                token_type_ids=question_token_type_ids,
                attention_mask=question_attention_mask,
            )[1]
            scores = self.score_fn(question_embeddings, passage_embeddings)
            return scores
        else:
            passage_embeddings = self.passage_tower(
                input_ids=passage_input_ids,
                token_type_ids=passage_attention_mask,
                attention_mask=passage_token_type_ids,
            )[1]
            return passage_embeddings
