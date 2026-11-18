import torch.cuda
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain.docstore.document import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)
# print(f'rag_qa_path--》{rag_qa_path}')
core_path = os.path.join(rag_qa_path, 'core')
sys.path.insert(0, core_path)
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

from document_processor import *
from base import logger, Config


conf = Config()


# core/vector_store.py
# 定义 VectorStore 类，封装向量存储和检索功能
class VectorStore:
    # 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 collection_name=conf.MILVUS_COLLECTION_NAME,
                 host=conf.MILVUS_HOST,
                 port=conf.MILVUS_PORT,
                 database=conf.MILVUS_DATABASE_NAME):
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database
        # 设置日志记录器
        self.logger = logger
        # 检查CUDA是否可用
        self.device ='cuda' if torch.cuda.is_available() else 'cpu'
        # 日志提醒使用的是什么设备
        self.logger.info(f"使用设置：{self.device}")
        # 初始化 BGE-Reranker 模型，用于重排序检索结果
        reranker_path = os.path.join(rag_qa_path, 'models', 'bge-reranker-large')
        # print(f'reranker_path--》{reranker_path}')
        self.reranker = CrossEncoder(reranker_path, device=self.device)
        # 初始化 BGE-M3 嵌入函数，使用 CPU 设备，不启用 FP16
        m3_path = os.path.join(rag_qa_path, 'models', 'bge-m3')
        self.embedding_function = BGEM3EmbeddingFunction(model_name_or_path=m3_path, use_fp16=(self.device == 'cuda'), device=self.device)
        # 获取稠密向量的维度# 1024
        self.dense_dim = self.embedding_function.dim["dense"]
        # 初始化 Milvus 客户端，连接到指定主机和数据库
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()
    def _create_or_load_collection(self):
        if not self.client.has_collection(self.collection_name):
            schema=self.client.create_schema(auto_id=False,enable_dynamic_field=True)
            schema.add_field(field_name='id',datatype=DataType.VARCHAR,is_primary=True,max_length=100)
            schema.add_field(field_name='text',datatype=DataType.VARCHAR,max_length=65535)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)
            index_params=self.client.prepare_index_params()
            index_params.add_index(
                field_name='dense_vector',
                index_name='dense_index',
                index_type='IVF_FLAT',
                metric_type='IP',
                params={'nlist':128}
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )
            self.client.create_collection(collection_name=self.collection_name,schema=schema,index_params=index_params)
            logger.info(f'已创建集合{self.collection_name}')
        else:
            logger.info(f"已加载集合 {self.collection_name}")
            # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)
    def add_documents(self, documents):
        texts=[doc.page_content for doc in documents]
        embeddings=self.embedding_function.encode(texts)
        data=[]
        for i,doc in enumerate(documents):
            text_hash=hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            sparse_vector={}
            row=embeddings['sparse'].getrow(i)
            indices=row.indices
            values=row.data
            for idx,value in zip(indices, values):
                sparse_vector[idx]=value
            data.append(
                {
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings["dense"][i],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            }
            )
        if data:
            self.client.upsert(collection_name=self.collection_name,data=data)
        logger.info(f"已插入或更新 {len(data)} 个文档")
    def hybrid_search_with_rerank(self,query,k=conf.RETRIEVAL_K, source_filter=None):
        embeddings=self.embedding_function([query])
        dense_query_vector=embeddings["dense"][0]
        sparse_query_vector={}
        row=embeddings["sparse"].getrow(0)
        indices=row.indices
        values=row.data
        for idx,value in zip(indices, values):
            sparse_query_vector[idx]=value
        filter_expr=f"source=='{source_filter}'" if source_filter else ""
        dense_request=AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={'metric_type': 'IP','params': {'nprobe':10}},
            limit=k,
            expr=filter_expr
        )
        sparse_request=AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={'metric_type':'IP','params':{}},
            limit=k,
            expr=filter_expr
        )
        ranker=WeightedRanker(1.0,0.7)
        results=self.client.hybrid_search(collection_name=self.collection_name,
                                          reqs=[dense_request,sparse_request],
                                          ranker=ranker,
                                          limit=k,
                                          output_fields=['text','parent_id','parent_content','source','timestamp']
                                          )[0]
        sub_chunks=[self._doc_from_hit(hit['entity']) for hit in results]
        parent_docs=self._get_unique_parent_docs(sub_chunks)
        if len(parent_docs)<2:
            return parent_docs[:conf.RETRIEVAL_K]
        if parent_docs:
            pairs=[[query,doc.page_content]for doc in parent_docs]
            scores=self.reranker.predict(pairs)
            ranked_parent_docs =[doc for _,doc in sorted(zip(scores,parent_docs),reverse=True)]
        else:
            ranked_parent_docs = []

        # 返回前 k 个重排序后的文档
        return ranked_parent_docs[:conf.CANDIDATE_M]

    def _doc_from_hit(self,hit):
        return Document(
            page_content=hit.get('text'),
            metadata={
                'parent_id':hit.get('parent_id'),
                'parent_content':hit.get('parent_content'),
                'source':hit.get('source'),
                'timestamp':hit.get('timestamp')
            }
        )
    def _get_unique_parent_docs(self,sub_chunks):
        parent_contents=set()
        unique_docs=[]
        for chunk in sub_chunks:
            parent_content=chunk.metadata.get('parent_content',chunk.page_content)
            if parent_content and parent_content not in parent_contents:
                parent_contents.add(parent_content)
                unique_docs.append(Document(page_content=parent_content,metadata=chunk.metadata))
        return unique_docs



















