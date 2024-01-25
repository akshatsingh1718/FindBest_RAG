import psutil

print(f"Before imports= {psutil.virtual_memory().percent} ")

from llama_index.indices.postprocessor import (
    SentenceTransformerRerank,
)
import gc

print(f"After imports= {psutil.virtual_memory().percent} ")


class AB:

    # def __init__(cls):
    @staticmethod
    def froms():
        rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")
        return rerank

    def delete(cls):
        del cls.rerank


print(f"After class= {psutil.virtual_memory().percent} ")

l = list()

for i in range(5):
    print("==============", i)
    print(f"\t before instantiating= {psutil.virtual_memory().percent} ")
    r = AB.froms()
    l.append(r)
    print(f"\t after {r} instantiating= {psutil.virtual_memory().percent} ")

    del r
    gc.collect()
    print(l)
    print(f"\tafter del= {psutil.virtual_memory().percent} ")

