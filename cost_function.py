from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel
import torch
import torch.nn as nn
from sentence_transformers import CrossEncoder
import math
import os
import numpy

# Set the TOKENIZERS_PARALLELISM environment variable to avoid the warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    # Extract the scalar value to avoid the deprecation warning
    if isinstance(coherence_score, (list, tuple, torch.Tensor, numpy.ndarray)):
        coherence_score = float(coherence_score[0])
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

    # Calculate human-likeness score (inverse of AI score with higher weight)
    human_score = 1 - ai_score

    # Length-based penalty for extremely short answers (less than 5 words)
    word_count = len(latest_message.split())
    length_factor = min(1.0, word_count / 5)

    # Penalize responses with very low coherence more severely
    coherence_factor = coherence_score**2 if coherence_score < 0.6 else coherence_score

    # Calculate final score with more emphasis on human score and a minimum coherence threshold
    # This formula gives more weight to human-written text and heavily penalizes incoherent text
    final_score = (human_score**1.5) * coherence_factor * length_factor

    # Normalize to 0-1 range
    final_score = min(1.0, max(0.0, final_score))

    return {
        "final_score": final_score,
        "ai_detection_score": ai_score,
        "coherence_score": coherence_score,
        "human_score": human_score,
        "length_factor": length_factor,
    }


# Example usage
if __name__ == "__main__":
    # Test conversation
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's your favorite book?"},
    ]
    print("Evaluation Results:")
    ai_answers = [
        "I'm not sure if I have a favorite book. I like to read a variety of books, depending on my mood and interests.",
        "I love reading 1984 by George Orwell because it makes me think deeply about society and human nature.",
        "That's tough—there are so many—but I think I'll go with To Kill a Mockingbird.",
        "I really love 1984 by George Orwell; it changed the way I see the world.",
        "Probably The Alchemist. I find something new each time I read it.",
        "Right now, I'd say The Great Gatsby—it's beautifully tragic.",
        "Hands down, Pride and Prejudice. It's timeless.",
        "It's always changing, but lately, I'm really into Sapiens by Yuval Noah Harari.",
        "Definitely Harry Potter and the Sorcerer's Stone; it made me love reading.",
        "I think Crime and Punishment is my all-time favorite; it really challenged my perspective.",
        "I can't pick just one, but The Catcher in the Rye always resonates with me.",
        "Honestly, The Lord of the Rings series is unbeatable for me.",
        "Beloved by Toni Morrison—it's incredibly powerful and moving.",
        "The Little Prince—it might seem simple, but it's deep and meaningful.",
        "Right now, I'd say Dune. It's got everything: politics, philosophy, adventure.",
        "Meditations by Marcus Aurelius—it's the kind of book I keep returning to.",
        "The Hitchhiker's Guide to the Galaxy because life shouldn't be taken too seriously!",
    ]

    human_answers = [
        "Maybe Lonesome Dove. It had me for around a thousand pages and I'm not a massive reader.",
        "East of Eden",
        "Count of Monte Cristo",
        "The Prince of Tides by Pat Conroy",
        "The stand, I've read the extended version about 10 times, and it always feels to me like meeting up with old friends again.",
        "Hyperion - a blend of six separate genres in one and each as compelling narrative as if they were standalone stories.",
        "11/22/63 was the first book I read when I started reading again. Haven't found anything better yet",
        "Jane Eyre. Made me feel less alone",
        "My favorite book of all time is J.R.R. Tolkien's The Lord of the Rings. It transports me to another world like little else can.",
    ]

    rubbish_answers = [
        "Madonna pig evangelical TOTALITY 123678 #$.",
        "transformers>=4.18.0 sentence-transformers>=2.2.0",
        "However, according to Zuckerberg, the era of the smartphone may be drawing to a close. In a recent conversation, he described how the way we interact with technology is evolving towards a more natural and social experience. He believes that soon, our de",
        "I am a pigeon going through midlife crisis",
    ]

    # Store results for each category
    ai_scores = []
    human_scores = []
    rubbish_scores = []

    # Process all answers and collect scores per category
    for category_name, category_answers, score_list in [
        ("AI ANSWERS", ai_answers, ai_scores),
        ("HUMAN ANSWERS", human_answers, human_scores),
        ("RUBBISH ANSWERS", rubbish_answers, rubbish_scores),
    ]:
        print(f"\n===== {category_name} =====")
        for response in category_answers:
            final_answer = {
                "role": "assistant",
                "content": response,
            }
            # Evaluate the last message
            results = evaluate_text(conversation + [final_answer])
            score_list.append(results["final_score"])

            # Classify based on final score
            if results["final_score"] > 0.5:
                category = "LIKELY HUMAN"
            elif results["final_score"] > 0.2:
                category = "UNCERTAIN"
            else:
                category = "LIKELY AI/RUBBISH"

            print("________")
            print(f"Final Answer: {final_answer['content']}")
            print(f"Classification: {category}")
            print(f"Final Score: {results['final_score']:.3f}")
            print(f"AI Detection Score: {results['ai_detection_score']:.3f}")
            print(f"Human Score: {results['human_score']:.3f}")
            print(f"Coherence Score: {results['coherence_score']:.3f}")
            print(f"Length Factor: {results['length_factor']:.3f}")

    # Print summary statistics
    print("\n===== SUMMARY STATISTICS =====")
    print("AI Answers Final Scores:", [f"{score:.3f}" for score in ai_scores])
    print(
        f"AI Answers - Mean: {sum(ai_scores)/len(ai_scores):.3f}, Min: {min(ai_scores):.3f}, Max: {max(ai_scores):.3f}"
    )

    print("\nHuman Answers Final Scores:", [f"{score:.3f}" for score in human_scores])
    print(
        f"Human Answers - Mean: {sum(human_scores)/len(human_scores):.3f}, Min: {min(human_scores):.3f}, Max: {max(human_scores):.3f}"
    )

    print(
        "\nRubbish Answers Final Scores:", [f"{score:.3f}" for score in rubbish_scores]
    )
    print(
        f"Rubbish Answers - Mean: {sum(rubbish_scores)/len(rubbish_scores):.3f}, Min: {min(rubbish_scores):.3f}, Max: {max(rubbish_scores):.3f}"
    )

    # Calculate classification accuracy
    ai_correct = sum(1 for score in ai_scores if score <= 0.2)
    human_correct = sum(1 for score in human_scores if score > 0.5)
    rubbish_correct = sum(1 for score in rubbish_scores if score <= 0.2)

    print("\n===== CLASSIFICATION ACCURACY =====")
    print(
        f"AI Answers correctly classified: {ai_correct}/{len(ai_scores)} ({ai_correct/len(ai_scores)*100:.1f}%)"
    )
    print(
        f"Human Answers correctly classified: {human_correct}/{len(human_scores)} ({human_correct/len(human_scores)*100:.1f}%)"
    )
    print(
        f"Rubbish Answers correctly classified: {rubbish_correct}/{len(rubbish_scores)} ({rubbish_correct/len(rubbish_scores)*100:.1f}%)"
    )
    print(
        f"Overall accuracy: {(ai_correct + human_correct + rubbish_correct)/(len(ai_scores) + len(human_scores) + len(rubbish_scores)):.3f}"
    )
