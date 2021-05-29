# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
import spacy,re,json
from spacy.tokens import Doc
from skweak import heuristics, gazetteers, aggregation, utils

def get_entities(doc: Doc, layer=None):
    """write the entities annotated in a spacy document, based on the
    provided annotation layer(s). If layer is None, the method displays
    the entities from Spacy
    """
    if layer is None:
        spans = doc.ents
    elif type(layer) is list:
        spans = utils.get_spans(doc, layer)
    elif type(layer) == str:
        if "*" in layer:
            matched_layers = [l for l in doc.spans
                              if re.match(layer.replace("*", ".*?")+"$", l)]
            spans = utils.get_spans(doc, matched_layers)
        else:
            spans = doc.spans[layer]
    else:
        raise RuntimeError("Layer type not accepted")

    entities = {}
    for span in spans:
        start_char = doc[span.start].idx
        end_char = doc[span.end-1].idx + len(doc[span.end-1])

        if (start_char, end_char) not in entities:
            entities[(start_char, end_char)] = span.label_

        # If we have several alternative labels for a span, join them with +
        elif span.label_ not in entities[(start_char, end_char)]:
            entities[(start_char, end_char)] = entities[(
                start_char, end_char)] + "+" + span.label_

    entities = [{"start": start, "end": end, "label": label}
                for (start, end), label in entities.items()]

    for item in entities:
        item['term']=doc.text[item['start']:item['end']]

    doc2 = { "text": doc.text, "entities": entities}
    return doc2

# LF 1: heuristic to detect occurrences of MONEY entities
def money_detector(doc):
   for tok in doc[1:]:
      if tok.text[0].isdigit() and tok.nbor(-1).is_currency:
          yield tok.i-1, tok.i+1, "MONEY"


# spacy load or download cached model
cache_dir=os.getenv("cache_dir", "../../models")
try:
    nlp = spacy.load(os.path.join(cache_dir,'en_core_web_sm'))
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')
    nlp.to_disk(os.path.join(cache_dir,'en_core_web_sm'))

lf1 = heuristics.FunctionAnnotator("money", money_detector)

# LF 2: detection of years with a regex
lf2= heuristics.TokenConstraintAnnotator("years", lambda tok: re.match("(19|20)\d{2}$", tok.text), "DATE")

# LF 3: a gazetteer with a few names
NAMES = [("Barack", "Obama"), ("Donald", "Trump"), ("Joe", "Biden")]
trie = gazetteers.Trie(NAMES)
lf3 = gazetteers.GazetteerAnnotator("presidents", {"PERSON":trie})

# We create a corpus (here with a single text)
doc = nlp("Donald Trump paid $750 in federal income taxes in 2016")

# apply the labelling functions
doc = lf3(lf2(lf1(doc)))

# and aggregate them
hmm = aggregation.HMM("hmm", ["PERSON", "DATE", "MONEY"])
hmm.fit_and_aggregate([doc])

# retrieve entities span
entities_list = get_entities(doc, "hmm")
print(json.dumps(entities_list, indent=2))