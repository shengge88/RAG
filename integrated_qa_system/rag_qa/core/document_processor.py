import os
import sys
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_RAG_QA_DIR = os.path.dirname(_CURRENT_DIR)
_PROJECT_ROOT = os.path.dirname(_RAG_QA_DIR)
for _p in (_PROJECT_ROOT, _RAG_QA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from datetime import datetime
from edu_text_spliter import AliTextSplitter, ChineseRecursiveTextSplitter
from edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
from base import logger, Config

conf = Config()
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader,
}


def load_documents_from_directory(directory_path):
    documents = []
    supported_extensions = document_loaders.keys()
    # 从目录名提取学科类别（如 "ai_data" -> "ai"）
    source = os.path.basename(directory_path).replace("_data", "")
    for root, _, files in os.walk(directory_path):
        for filename in files:
            file_path=os.path.join(root, filename)
            file_extension=os.path.splitext(filename)[1].lower()
            if file_extension in supported_extensions:
                try:
                    loader_class=document_loaders[file_extension]
                    if file_extension == ".txt":
                        loader=loader_class(file_path,encoding="utf-8")
                    else:
                        loader=loader_class(file_path)
                    loaded_docs=loader.load()
                    for doc in loaded_docs:
                        doc.metadata['source']=source
                        doc.metadata['file_path']=file_path
                        doc.metadata['timestamp']=datetime.now().isoformat()
                    documents.extend(loaded_docs)
                    logger.info(f"成功加载文件: {file_path}")
                except Exception as e:
                    # 记录加载失败的日志，包含错误信息
                    logger.error(f"加载文件 {file_path} 失败: {str(e)}")
                    # 如果文件类型不在支持列表中
            else:
                # 记录警告日志，提示不支持的文件类型
                logger.warning(f"不支持的文件类型: {file_path}")
                # 返回加载的所有文档列表
    return documents
def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                     child_chunk_size=conf.CHILD_CHUNK_SIZE,
                     chunk_overlap=conf.CHUNK_OVERLAP):
    documents=load_documents_from_directory(directory_path)
    logger.info(f'加载的文档数量: {len(documents)}')
    parent_splitter=ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter=ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    markdown_parent_splitter=MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter=MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    child_chunks=[]
    for i,doc in enumerate(documents):
        file_extension=os.path.splitext(doc.metadata.get('file_path'))[1].lower()
        is_markdown=file_extension==".md"
        parent_splitter_to_use=markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use=markdown_child_splitter if is_markdown else child_splitter
        logger.info(f"处理文档: {doc.metadata['file_path']}, 使用切分器: {'Markdown' if is_markdown else 'ChineseRecursive'}")
        parent_docs=parent_splitter_to_use.split_documents([doc])
        for j,parent_doc in enumerate(parent_docs):
            parent_id=f'doc_{i}_parent_{j}'
            parent_doc.metadata['parent_id']=parent_id
            parent_doc.metadata['parent_content']=parent_doc.page_content
            sub_chunks=child_splitter_to_use.split_documents([parent_doc])
            for k,sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata['parent_id']=parent_id
                sub_chunk.metadata['parent_content']=parent_doc.page_content
                sub_chunk.metadata['id']=f'{parent_id}_child_{k}'
                child_chunks.append(sub_chunk)
    logger.info(f"子块数量: {len(child_chunks)}")
    return child_chunks
if __name__ == '__main__':
    chunks = process_documents(
        r'C:\Users\Lenovo\PycharmProjects\LLMproject\integrated_qa_system\rag_qa\data\ai_data',
        conf.PARENT_CHUNK_SIZE,
        conf.CHILD_CHUNK_SIZE,
        conf.CHUNK_OVERLAP,
    )
    print(chunks)