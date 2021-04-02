# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020-2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", 
    cache_dir=os.getenv("cache_dir", "../../models"))

model = AutoModel.from_pretrained("facebook/bart-large-cnn", 
    cache_dir=os.getenv("cache_dir", "../../models"))

LONG_BORING_TENNIS_ARTICLE = """
 Andy Murray  came close to giving himself some extra preparation time for his w
edding next week before ensuring that he still has unfinished tennis business to
 attend to. The world No 4 is into the semi-finals of the Miami Open, but not be
fore getting a scare from 21 year-old Austrian Dominic Thiem, who pushed him to 
4-4 in the second set before going down 3-6 6-4, 6-1 in an hour and three quarte
rs. Murray was awaiting the winner from the last eight match between Tomas Berdy
ch and Argentina's Juan Monaco. Prior to this tournament Thiem lost in the secon
d round of a Challenger event to soon-to-be new Brit Aljaz Bedene. Andy Murray p
umps his first after defeating Dominic Thiem to reach the Miami Open semi finals
 . Muray throws his sweatband into the crowd after completing a 3-6, 6-4, 6-1 vi
ctory in Florida . Murray shakes hands with Thiem who he described as a 'strong 
guy' after the game . And Murray has a fairly simple message for any of his fell
ow British tennis players who might be agitated about his imminent arrival into 
the home ranks: don't complain. Instead the British No 1 believes his colleagues
 should use the assimilation of the world number 83, originally from Slovenia, a
s motivation to better themselves. At present any grumbles are happening in priv
ate, and Bedene's present ineligibility for the Davis Cup team has made it less 
of an issue, although that could change if his appeal to play is allowed by the 
International Tennis Federation. Murray thinks anyone questioning the move, now 
it has become official, would be better working on getting their ranking closer 
to his.
""".replace('\n','')

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
article_input_ids = tokenizer.batch_encode_plus(
    [LONG_BORING_TENNIS_ARTICLE], 
    return_tensors='pt', 
    max_length=1024, 
    truncation=True)['input_ids'].to(torch_device)

summary_ids = model.generate(article_input_ids,num_beams=4,length_penalty=2.0,max_length=142,min_length=56,no_repeat_ngram_size=3)

# Generate Summary
#summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
#summary_txt = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#print('> **Summary: **'+summary_txt)
