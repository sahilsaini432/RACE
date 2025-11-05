import torch
import json
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)  # Assuming a classification task; adjust if different

# Assuming the model is a Hugging Face model (common for JSONL data like text classification).
# If it's a custom PyTorch model, replace with your model class and loading logic.
# Adjust paths as needed.


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]  # Assuming each JSON line has a 'text' field; adjust if different
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "text": text,
        }


def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def predict(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
    return predictions


# Main script
if __name__ == "__main__":
    # Paths (update these)
    model_path = "path/to/your/trained/model"  # e.g., 'model.pth' or Hugging Face model name
    new_data_path = "path/to/new/data.jsonl"  # New data file
    tokenizer_name = "bert-base-uncased"  # Adjust to match your model's tokenizer

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)  # For custom model, use torch.load
    model.to(device)

    # Load new data
    new_data = load_data(new_data_path)

    # Create dataset and dataloader
    dataset = TextDataset(new_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)  # Adjust batch size as needed

    # Get predictions
    preds = predict(model, dataloader, device)

    # Output predictions (e.g., save to file or print)
    with open("predictions.txt", "w") as f:
        for i, pred in enumerate(preds):
            f.write(f"{i}: {pred}\n")  # Adjust format as needed
    print("Predictions saved to predictions.txt")
