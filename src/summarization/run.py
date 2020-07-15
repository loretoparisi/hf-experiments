from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

model = AutoModel.from_pretrained("facebook/bart-large-cnn")
