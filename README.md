# LLM Learnings

* **Document search**:
  It is way to find relevant documents in response to query. This is a vector search approach.
  
  After loading and chuncking data, it is embedded and stored in vectorstore.
  When user asks query, it gets embedded and used to search related document from vectorstore.

  Here, FAISS vectorstore has used with langchain to perform document search.

  Search methods:

  1. Similarity search: Based on some similarity measure, most similar documents to query are found out.
  2. MMR (Maximum Marginal Relevance): It extracts similar to query and diverse documents.

  To implement document search use command
  ```
  !python document_search.py --pdf_path PDF_PATH --que "QUESTION"
  ```
  or use this for sample example
  ```
  !python document_search.py
  ```
  The output will be retrived documents based on question for different methods.

  Change top_k value, score threshold based on output retrieval.
  

* **Document Question-Answer**:

  After retriving documents based on query, LLM is used to generate answer for asked question refering those documents.
  
  In this, cohere embeddings and cohere model is used to generate the answer.
  To implement document_QA.py, generate your own cohere key and run command
  ```
  uvicorn main:app --reload
  ```
  On swagger UI, need to input cohere key, query, and pdf file. This will return JSON response with query and corresponding answer.

* **Langchain's Runnable Interface**:

  The notebook ```Implementation of Langchain's Runnable Interface``` contains each runnable method's example to understand its functionality.

  The methods covered are:
  * Pipe operator
  * RunnableLambda
  * RunnablePassthrough
  * RunnableParallel
  * RunnablePassthrough.assign

   
  
**References**:
1. https://python.langchain.com/docs/get_started/introduction
2. https://docs.cohere.com/docs/the-cohere-platform
3. https://python.langchain.com/v0.1/docs/expression_language/interface/
