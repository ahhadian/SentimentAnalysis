import re
from hazm import Normalizer
hazm_normalizer = Normalizer()
link_regex = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"


def preprocess(text: str):
    text = re.sub(r"<(/?p/?|/?br/?|)>", '', text)
    text = re.sub(link_regex, " ", text)
    text = re.sub("&?amp", " ", text)
    text = re.sub(r"[.,،?؟:;'\"/«»@!٬﷼#()<>*_]", " ", text)
    text = re.sub("[\r\n]+", ' ', text)
    text = re.sub("‌", " ", text)    # turn half-space to space in persian
    text = re.sub("(ن?می)\s+", r"\1", text) # attach prefixes of verbs to the verb
    text = hazm_normalizer.character_refinement(text)
    wierd_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0001f926-\U0001f937"
    u'\U00010000-\U0010ffff'
    u"\u200d"
    u"\u2640-\u2642"
    u"\u2600-\u2B55"
    u"\u23cf"
    u"\u23e9"
    u"\u231a"
    u"\u3030"
    u"\ufe0f"
    u"\u2069"
    u"\u2066"
    u"\u2068"
    u"\u2067"
    "]+", flags=re.UNICODE)
    text = wierd_pattern.sub(r'', text)
    text = re.sub("\s+", " ", text)
    return text.strip()
