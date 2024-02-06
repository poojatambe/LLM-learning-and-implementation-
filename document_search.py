from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


def document_embedding(pdf_path):
    """
    Load and split the document. After store document embeddings
    to vectorestore.
    """
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                chunk_overlap=50,
                                                add_start_index=True)
    splits = text_splitter.split_documents(data)
    print(f"We have {len(splits)} chunks in memory")
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore, embeddings


path = r"D:/Winjit_training/LLMs/1409.0473.pdf"
query = "What is rnn encoder-decoder?"

# document  and query embedding
vectorstore, embeddings = document_embedding(path)
query_embed = embeddings.embed_query(query)

# With similarity search
data = vectorstore.similarity_search(query, 3)
print('Siilarity search documents:\n')
for i in range(len(data)):
    print(data[i].page_content)
    print("--------------------------------")

data_1 = vectorstore.similarity_search_with_score(query, 3)
print('Similarity search with score :\n')
for i in range(len(data_1)):
    print(data_1[i])
    print("--------------------------------")

data_2 = vectorstore.similarity_search_by_vector(query_embed, 3)
print('Similarity search using query vector :\n')
for i in range(len(data_2)):
    print(data_2[i])
    print("--------------------------------")

data_3 = vectorstore.similarity_search_with_score_by_vector(query_embed, 3)
print('Similarity search with score using query vector:\n')
for i in range(len(data_3)):
    print(data_3[i])
    print("--------------------------------")

data_4 = vectorstore.similarity_search_with_relevance_scores(query, 3)
print('Similarity search with relevance score:\n')
for i in range(len(data_4)):
    print(data_4[i])
    print("--------------------------------")

# max marginal relevance serach
data_5 = vectorstore.max_marginal_relevance_search(query, 3)
print('Max marginal relevance search:\n')
for i in range(len(data_5)):
    print(data_5[i])
    print("--------------------------------")

data_6 = vectorstore.max_marginal_relevance_search_by_vector(query_embed, 3)
print('Max marginal relevance search using query vector:\n')
for i in range(len(data_6)):
    print(data_6[i].page_content)
    print("--------------------------------")

data_7 = vectorstore.max_marginal_relevance_search_with_score_by_vector(query_embed, k=3)
print('Max marginal relevance search using query vector:\n')
for i in range(len(data_7)):
    print(data_7[i])
    print("--------------------------------")

# as_retriever
retriever = vectorstore.as_retriever(search_type="mmr")
retrieved_docs = retriever.get_relevant_documents(query)
print('max maximal relevance retrived documents:\n')
for i in range(len(retrieved_docs)):
    print(retrieved_docs[i].page_content)
    print("--------------------------------")

retriever_1 = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                         search_kwargs={"score_threshold": 0.5})
retrieved_docs_1 = retriever_1.get_relevant_documents(query)
print('Similarity score threshold retrived documents:\n')
for i in range(len(retrieved_docs_1)):
    print(retrieved_docs_1[i].page_content)
    print("--------------------------------")

retriever_2 = vectorstore.as_retriever(search_kwargs={"k": 2})
retrieved_docs_2 = retriever_2.get_relevant_documents(query)
print('top 2 retrived documents:\n')
for i in range(len(retrieved_docs_2)):
    print(retrieved_docs_2[i].page_content)
    print("--------------------------------")
