import pandas as pd
import numpy as np

# 단계 1: 문서 로드(Load Documents)
from langchain_core.documents import Document
# from langchain_community.document_loaders import CSVLoader

# 단계 2: 문서 분할(Split Documents)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 단계 3: 임베딩 & 벡터스토어 생성(Create Vectorstore)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 단계 4: 검색(Search)
# from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# 단계 5: 프롬프트 생성(Create Prompt)

# 단계 6: 언어모델 생성(Create LLM)
from langchain_community.chat_models import ChatOllama

# 단계 7: 체인 생성(Create Chain)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 단계 8: 체인 실행(Run Chain)

# 단계 9: 결과 출력


##########################################################################
# Inference Function
def generate_answer(context, question):
    # 단계 1: 문서 로드(Load Documents) - 함수 파라미터

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    splits = text_splitter.split_text(context)
    
    # 벡터스토어 생성
    faiss_vectorstore = FAISS.from_texts(splits,
                                   embedding = embeddings
                                  )

    # 단계 4: 검색(Search)
    # 문서 포함되어 있는 정보를 검색하고 생성합니다.
    # 리트리버는 구조화되지 않은 쿼리가 주어지면 문서를 반환하는 인터페이스입니다.
    # initialize the bm25 retriever and faiss retriever

    bm25_retriever = BM25Retriever.from_texts(splits)
    bm25_retriever.k = 10
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})

    # initialize the ensemble retriever
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )

    ## 7-2: rag chain 생성
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(question)
    print(response)
    return response
##########################################################################





##########################################################################
# Main Code
# CSV Load -> Call Inference function -> Save Output -> Make output.csv ##
# Read the CSV file into a Pandas DataFrame
input_df = pd.read_csv("/home/Chatbot/LangChain/assets/open/test.csv")

input_df = input_df.head(30)

# Initialize an empty list to store the results
results = []

# Initialize relevenant with Model Inferencing
# 단계 3: 임베딩 & 벡터스토어 생성(Create Vectorstore)
# 임베딩 모델 선택
# 3- (나) - BAAI로 임베딩
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs = {'device': 'cuda'}, # 모델이 CPU에서 실행되도록 설정. GPU를 사용할 수 있는 환경이라면 'cuda'로 설정할 수도 있음
    encode_kwargs = {'normalize_embeddings': True}, # 임베딩 정규화. 모든 벡터가 같은 범위의 값을 갖도록 함. 유사도 계산 시 일관성을 높여줌
)

# 단계 6: 언어모델 생성(Create LLM)
# 모델(LLM) 을 생성합니다.
llm = ChatOllama(model="0704:latest")


# 단계 7: 체인 생성(Create Chain)
## 7-1: 프롬프트 임의 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", "Please answer the question in one word according to the following context."),
    ("user", "{question}")
])


# Iterate over each row in the DataFrame
for index, row in input_df.iterrows():
    id = row['id']
    context = row['context']
    question = row['question']
    answer = generate_answer(context, question)
    
    # Append the result to the list
    results.append({'id': id, 'answer': answer})

# Create a new DataFrame with the results
output_df = pd.DataFrame(results)

# Write the new DataFrame to a CSV file
output_df.to_csv('output.csv', index=False)

print("Output CSV file has been created.")
##########################################################################
