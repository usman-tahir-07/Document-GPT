
# Document-GPT
This project is basically a RAG app where you can upload your documents ( pdf, csv) and can ask any question from them.

I made this project using langchain which is a popular and powerful framework for making AI base applications. I have used Groq API keys for the llm which is lamma3-8b.

I have used streamlit for the web interface of this application.


## Live Video Overview

https://github.com/user-attachments/assets/553be780-36b7-4376-93f7-d0d73d61f210

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`GROQ_API_KEY`

## Working

Firstly the uploaded documents are going to be converted into chunks becuase of the context window of LLMs and then these chunks are going to be stored in Vector DB which is FAISS and vectors are stored locally in it.

Streamlit runs the code from start everytime I ask question from the same stored documents, So I made the session state for the vector store. Now it will not repeat the same indexing process everytime anyone ask some question.

Now when anyone ask the question, it will do similarity search and provide the best results from the documents.

![rag-process](https://github.com/user-attachments/assets/7b824afc-d005-468f-b294-07d22e770f03)

## Future Improvements

1-More supported file formats

2-Advance retrieval methods for complex datasets

3-Advance RAG such as Self-RAG for more effecient results

## Connect with Me
If you have any feedback, please let me know!

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/usman-tahir-676a51291)

