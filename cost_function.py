from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel
import torch
import torch.nn as nn
from sentence_transformers import CrossEncoder
import math


class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # Initialize the base transformer model.
        self.model = AutoModel.from_config(config)
        # Define a classifier head.
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights (handled by PreTrainedModel)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the transformer
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        # Mean pooling
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Classifier
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output


# Initialize models
model_directory = "desklib/ai-text-detector-v1.01"
tokenizer = AutoTokenizer.from_pretrained(model_directory)
ai_detector = DesklibAIDetectionModel.from_pretrained(model_directory)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ai_detector.to(device)

coherence_model = CrossEncoder("enochlev/coherence-all-mpnet-base-v2")


def predict_single_text(text, model, tokenizer, device, max_len=768, threshold=0.5):
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probability = torch.sigmoid(logits).item()

    label = 1 if probability >= threshold else 0
    return probability, label


def get_ai_detection_score(text, max_len=768):
    """Get the probability that the text is AI generated using Desklib model."""
    probability, _ = predict_single_text(text, ai_detector, tokenizer, device, max_len)
    return probability


def get_coherence_score(context, response):
    """Get the coherence score between context and response."""
    coherence_score = coherence_model.predict([[context, response]])
    normalized_score = 1 / (1 + math.exp(-coherence_score))
    return normalized_score


def evaluate_text(conversation):
    """
    Evaluate the latest message in a conversation using both AI detection and coherence.
    Returns a combined score that is high when the text appears human-written and coherent.
    """
    if len(conversation) < 2:
        raise ValueError("Conversation must have at least 2 messages")

    # Get the latest message and its context
    latest_message = conversation[-1]["content"]
    context = conversation[-2]["content"]

    # Get individual scores
    ai_score = get_ai_detection_score(latest_message)
    coherence_score = get_coherence_score(context, latest_message)

    # Calculate final score (high coherence and low AI probability is best)
    final_score = coherence_score * (1 - ai_score)

    return {
        "final_score": final_score,
        "ai_detection_score": ai_score,
        "coherence_score": coherence_score,
    }


# Example usage
if __name__ == "__main__":
    # Test conversation
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's your favorite book?"},
        {
            "role": "assistant",
            "content": "I mean thats an impossible choice man, its like picking your favorite child. I guess fiction i'd go with mice and men",
        },
    ]

    # Evaluate the last message
    results = evaluate_text(conversation)

    print("Evaluation Results:")
    print(f"Final Score: {results['final_score']:.3f}")
    print(f"AI Detection Score: {results['ai_detection_score']:.3f}")
    print(f"Coherence Score: {results['coherence_score']:.3f}")
