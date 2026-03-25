from transformers import BartTokenizer, BartForConditionalGeneration

SAVE_PATH = "models/bart-large-cnn"

print("Downloading and saving model locally...")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model     = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"Model saved to {SAVE_PATH}")
print("You can now run without internet.")