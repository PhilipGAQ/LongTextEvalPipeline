from mteb import MTEB
from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from sentence_transformers import SentenceTransformer
from mteb import AbsTaskRetrieval
from datasets import load_dataset, DatasetDict
from collections import defaultdict

def load_retrieval_data(hf_hub_name, eval_splits):
    eval_split = eval_splits[0]
    dataset = load_dataset(hf_hub_name)
    qrels = load_dataset(hf_hub_name + '-qrels')[eval_split]

    corpus = {e['id']: {'text': e['text']} for e in dataset['corpus']}
    queries = {e['id']: e['text'] for e in dataset['queries']}
    relevant_docs = defaultdict(dict)
    for e in qrels:
        relevant_docs[e['qid']][e['pid']] = e['score']

    corpus = DatasetDict({eval_split:corpus})
    queries = DatasetDict({eval_split:queries})
    relevant_docs = DatasetDict({eval_split:relevant_docs})
    return corpus, queries, relevant_docs

class LongDocRetrieval(AbsTaskRetrieval):
    
    @property
    def description(self):
        return {
            'name':'LongDocRetrieval',
            'hf_hub_name':'xxxx',
            'eval_splits':['dev'],
            'type':'retrieval',
            'category':'s2p'
        }
        
    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True
        
class Long

model = SentenceTransformer("average_word_embeddings_komninos")
evaluation = MTEB(tasks=[LongDocReranking()])
evaluation.run(model)