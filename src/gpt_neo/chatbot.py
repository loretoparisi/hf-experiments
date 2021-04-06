# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os,sys,codecs
#import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

# To control logging level for various modules used in the application:
import logging
import re
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

def read_stdin():
    '''
        read standard input
        yeld next line
    '''
    try:
        readline = sys.stdin.readline()
        while readline:
            yield readline
            readline = sys.stdin.readline()
    except:
        # LP: avoid to exit(1) at stdin end
        pass

# control logging level: ["transformers", "nlp", "torch", "tensorflow", "tensorboard", ...]
set_global_logging_level(logging.ERROR, ["transformers"])

# tokenizer and model with cache_dir
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M",
    cache_dir=os.getenv("cache_dir", "../../models"))
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M", 
    cache_dir=os.getenv("cache_dir", "../../models"))
# text generation pipeline
generator = pipeline('text-generation',
    model=model,
    tokenizer=tokenizer)

ofp = codecs.getwriter('utf8')(sys.stdout.buffer)
print(">> You (type something):")
for sequence in read_stdin():
    # generate
    response = generator(sequence, do_sample=True, min_length=50, max_length=100)
    # [{"generated_text": "I feel sad\nA new year begins when a thousand\nLions go up and kill\nA"}]
    out = "GPTNeo: {}".format(response[0]['generated_text'].replace('\n', ' ')) # remove newlines
    
    # write to stdout
    try:
        ofp.writelines([out, '\n'])
        sys.stdout.flush()
        print(">> You (type something):")
    except Exception as ex:
        pass
    except KeyboardInterrupt:
        # close files when process ends
        ofp.close()
        # gracefully exit
        sys.exit(0)

# code adapted from https://www.kdnuggets.com/2021/02/hugging-face-transformer-basics.html
#for step in range(5):
   # encode the new user input, add the eos_token and return a tensor in Pytorch
   # new_user_input_ids = tokenizer.encode(input(">> You:") + tokenizer.eos_token, return_tensors='pt')
   # append the new user input tokens to the chat history
   # bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
   # generated a response while limiting the total chat history to 1000 tokens,
   # chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
   # pretty print last output tokens from bot
   # print("GPTNeo: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

