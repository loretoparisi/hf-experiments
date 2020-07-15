import os
from transformers import AutoTokenizer, AutoModelWithLMHead
'''
FutureWarning: The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. 
  Please use `AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and `AutoModelForSeq2SeqLM` for encoder-decoder models.
'''
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")

model = AutoModelWithLMHead.from_pretrained(
    "mrm8488/t5-base-finetuned-emotion", cache_dir=os.getenv("cache_dir", "model"))

def get_emotion(text):
  input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')

  output = model.generate(input_ids=input_ids,
               max_length=2)

  dec = [tokenizer.decode(ids) for ids in output]
  label = dec[0]
  return label

res = get_emotion("i feel as if i havent blogged in ages are at least truly blogged i am doing an update cute") # Output: 'joy'
print(res)

res = get_emotion("i have a feeling i kinda lost my best friend") # Output: 'sadness'
print(res)