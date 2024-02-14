from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import CohereEmbeddings
from langchain.llms import Cohere
from langchain.chains import RetrievalQA


def pdf_QA(key, file, query):
    """
    Question answer with PDF file.
    """
    cohere_key = key
    # data loading and splitting
    loader = PyPDFLoader(file)
    data = loader.load()
    splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=['\n\n', '.']
            )
    docs = splitter.split_documents(data)

    # embedding and storing in vectorstore
    embeddings = CohereEmbeddings(model='embed-english-v3.0', 
                                  cohere_api_key=cohere_key)

    vectorstore = FAISS.from_documents(docs, embeddings)
    print('Done embedding and storing')

    # retrieve documents based on query
    retriever = vectorstore.as_retriever(
                                search_type='mmr',
                                search_kwargs={'k': 3}
                            )
    relevant_docs = retriever.get_relevant_documents(query)
    # print('retrieved docs: ', relevant_docs[0])

    # question-answer

    llm = Cohere(model='command-light', 
                    cohere_api_key=cohere_key, 
                    temperature=0.0,
            )

    qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever
                )
    results = qa({'query': query})
    return results

