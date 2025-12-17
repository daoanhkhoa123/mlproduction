import re
from collections import Counter
from typing import Callable, Optional

import torch
# NOTE: this will be downloaded in runtime by cloud, i do not want to explode my computer
from sentence_transformers import SentenceTransformer  # type: ignore


def prep_remove_common_words_with_counting(text1, text2):
    word1  = re.findall(r"\w+", text1)
    word2 = re.findall(r"\w+",text2)

    c1= Counter(w.lower() for w in word1)
    c2 = Counter(w.lower() for w in word2)
    common = c1& c2 

    rm_c = dict(common)
    clean1 = []
    for w in word1:
        lw = w.lower()
        if lw in rm_c and rm_c[lw] > 0:
            rm_c[lw] -= 1
        else:
            clean1.append(w)

    rm_c = dict(common)
    clean2 = []
    for w in word2:
        lw = w.lower()
        if lw  in rm_c and rm_c[lw] > 0:
            rm_c[lw] -= 1
        else:
            clean2.append(w)

    return " ".join(clean1), " ".join(clean2)

def agg_differential(x,y):
    return x-y

class Preprop:
    def __init__(self, prepropfn:Callable, aggfn:Optional[Callable]=None, encoder_name="bkai-foundation-models/vietnamese-bi-encoder", device=None) -> None:
        self.prepropfn = prepropfn
        self.aggfn = aggfn if aggfn is not None else lambda x,y : (x,y)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.encoder = SentenceTransformer(encoder_name, device=self.device) # type: ignore

    def __call__(self, text1, text2, batch_size, verbose = True):
        t1, t2 = self.prepropfn(text1,text2)
        v1 =  self.encoder.encode(t1, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=verbose)
        v2 =  self.encoder.encode(t2, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=verbose)
        return self.aggfn(v1,v2)
    
    def call_df(self, df, batch_size, verbose =True):
        t1, t2 = self.prepropfn(df)
        v1 =  self.encoder.encode(t1, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=verbose)
        v2 =  self.encoder.encode(t2, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=verbose)
        return self.aggfn(v1,v2)
