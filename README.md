# RAG-Chat-with-data
* Document search:
  It is way to find relevant documents in response to query.
  
  After loading and chuncking data, it is embedded and stored in vectorstore.
  When user asks query, it gets embedded and used to search related document from vectorstore.

  Here, FAISS vectorstore has used with langchain to perform document search.

  Search methods:

  1. Similarity search: Based on some similarity measure, most similar documents to query are found out.
  2. MMR (Maximum Marginal Relevance): It extracts similar to query and diverse documents.
