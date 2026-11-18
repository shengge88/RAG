from langchain.text_splitter import CharacterTextSplitter
import os
import re
from typing import List

# 默认模型目录：项目内 rag_qa/nlp_bert_document-segmentation_chinese-base
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RAG_QA_DIR = os.path.dirname(_THIS_DIR)
_DEFAULT_MODEL_DIR = os.path.join(
    _RAG_QA_DIR, "nlp_bert_document-segmentation_chinese-base"
)


class AliTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, model_path: str = None, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.model_path = model_path or os.environ.get(
            "EDU_DOC_SEG_MODEL", _DEFAULT_MODEL_DIR
        )
        self.device = device
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            # 延迟导入，避免仅导入类就拉起 modelscope/torch
            from modelscope.pipelines import pipeline

            if os.path.isdir(self.model_path):
                model = self.model_path
            else:
                # 本地没有模型时走 ModelScope 模型 id
                model = "damo/nlp_bert_document-segmentation_chinese-base"
            self._pipeline = pipeline(
                task="document-segmentation",
                model=model,
                device=self.device,
            )
        return self._pipeline

    def split_text(self, text: str) -> List[str]:
        # use_document_segmentation参数指定是否用语义切分文档，此处采取的文档语义分割模型为达摩院开源的nlp_bert_document-segmentation_chinese-base，论文见https://arxiv.org/abs/2107.09278
        # 如果使用模型进行文档语义切分，那么需要安装modelscope[nlp]：pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
        # 考虑到使用了三个模型，可能对于低配置gpu不太友好，因此这里将模型load进cpu计算，有需要的话可以替换device为自己的显卡id
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub(r"\s", " ", text)
            text = re.sub(r"\n\n", "", text)
        p = self._get_pipeline()
        result = p(documents=text)
        sent_list = [i for i in result["text"].split("\n\t") if i]
        return sent_list


if __name__ == "__main__":
    model_split = AliTextSplitter()
    result = model_split.split_text(
        text="移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。"
    )
    print(result)
