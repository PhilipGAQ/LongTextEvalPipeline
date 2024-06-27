import argparse
from LMTEB_retrieval import *
from flag_dres_model import FlagDRESModel
from mteb import MTEB

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="BAAI/bge-large-zh", type=str)
    parser.add_argument('--task_type', default=None, type=str)
    parser.add_argument('--add_instruction', action='store_true', help="whether to add instruction for query")
    parser.add_argument('--pooling_method', default='cls', type=str)
    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()

    model = FlagDRESModel(model_name_or_path=args.model_name_or_path,
                            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                            pooling_method=args.pooling_method)   

    if args.add_instruction:
        instruction="为这个句子生成表示以用于检索相关文章："
    else:
        instruction=None

    model.query_instruction_for_retrieval = instruction

    evaluation = MTEB(tasks=[LongDocRetrieval()])
    evaluation.run(model, output_folder=f"zh_results/{args.model_name_or_path.split('/')[-1]}")
        
        
        
