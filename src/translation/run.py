# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020-2021 Loreto Parisi (loretoparisi at gmail dot com)
# HF: https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt

import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

article_en = "The head of the United Nations says there is no military solution in Syria"
model = MBartForConditionalGeneration.from_pretrained(
    "facebook/mbart-large-50-one-to-many-mmt", cache_dir=os.getenv("cache_dir", "../../models"))
tokenizer = MBart50TokenizerFast.from_pretrained(
    "facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX", 
    cache_dir=os.getenv("cache_dir", "../../models"))

model_inputs = tokenizer(article_en, return_tensors="pt")

# translate from English to Hindi
generated_tokens = model.generate(
    **model_inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => 'संयुक्त राष्ट्र के नेता कहते हैं कि सीरिया में कोई सैन्य समाधान नहीं है'

# translate from English to Chinese
generated_tokens = model.generate(
    **model_inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
)
decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => '联合国首脑说,叙利亚没有军事解决办法'
print(decoded)
