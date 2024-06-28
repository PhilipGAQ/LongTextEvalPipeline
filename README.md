# LongTextEvalPipeline
A pipeline for longtext embedding tasks (retrieval & reranking)
- Implemented from MTEB & C-MTEB & MLDR 

## To do:
- Collect Long text datasets.

- Test correctness.

- 添加load data的逻辑；不应该使用huggingface的链接，应该在本地load data；还是说通过切分策略生成数据集上传到huggingface？
    - 可以modify encode_corpus

- how to use the modified evaluation function?
    - Done