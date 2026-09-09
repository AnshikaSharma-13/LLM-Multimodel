"""
DistilBERT Tokenization Demo (Phase 2, Step 5)

This script demonstrates how Hugging Face AutoTokenizer (distilbert-base-uncased)
converts a raw text sentence into subword tokens, numerical token IDs, and an
attention mask, and decodes those IDs back into human-readable text.
"""

from transformers import AutoTokenizer


def demo_tokenization():
    # Step 1: Load the pre-trained DistilBERT tokenizer
    print("Loading distilbert-base-uncased tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Step 2: Define the input sentence
    text = "I am feeling nervous today."

    # Step 3: Tokenize the sentence to produce model input tensors (input_ids and attention_mask)
    encoded_inputs = tokenizer(text)

    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]

    # Convert token IDs back to human-readable subword tokens (including special tokens like [CLS] and [SEP])
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Step 4: Decode token IDs back into text string
    decoded_text = tokenizer.decode(input_ids)

    # Step 5: Display tokenization results
    print("\n--- DistilBERT Tokenization Demo ---")
    print(f"Original Text    : {text}")
    print(f"Tokens           : {tokens}")
    print(f"Token IDs        : {input_ids}")
    print(f"Attention Mask   : {attention_mask}")
    print(f"Token ID Count   : {len(input_ids)}")
    print(f"Decoded Text     : {decoded_text}")


if __name__ == "__main__":
    demo_tokenization()
