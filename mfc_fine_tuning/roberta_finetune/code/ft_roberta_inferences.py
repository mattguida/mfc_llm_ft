import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import json

# Load the model
model_path = "/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/roberta_finetune/output/final_model"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

# Load the label mapping
with open(f"{model_path}/label_map.json", 'r') as f:
    label_map = json.load(f)

# Invert the label map for easy lookup
inv_label_map = {v: k for k, v in label_map.items()}

# Set the model to evaluation mode
model.eval()

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    # Map the predicted class back to the original label
    predicted_label = inv_label_map[predicted_class]
    
    return predicted_label

text = "Same sex marriage is an immoral act and should not be allowed. It is against the natural order of things."
prediction = predict(text)
print(f"Predicted label: {prediction}")
