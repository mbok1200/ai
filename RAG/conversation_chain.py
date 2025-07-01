from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_aws import ChatBedrock

class ConversationChain:
    def __init__(self, config, vectorstore, memory):
        self.llm = ChatBedrock(
            model_id=config.BEDROCK_MODEL,
            model_kwargs={"temperature": config.TEMPERATURE},
            region_name=config.AWS_REGION
        )
        self.vectorstore = vectorstore
        self.memory = memory
        
    def chain(self):
        template = """You are a helpful and friendly AI assistant.
            Your tasks:
            1. Answer questions using the information provided in the documents below.
            2. If the information in the documents is insufficient, honestly state that you cannot find the answer.
            3. If the question is unrelated to the documents, feel free to engage in general friendly conversation.
            4. Remember the context of the conversation to provide better answers.
            5. Use the context from the documents to answer the question.
            6. Provide sources for your answers when possible.
            7. if the question is not related to the documents, you can use Google Search to find the answer.
            8. If you use Google Search, provide the answer in a friendly manner.
            9. If user ask give them answer in not Ukrainian, you can use the answer in Ukrainian.
            10. Always answer in Ukrainian, regardless of the language of the question.
            Current conversation:
            {chat_history}

            Context from documents:
            {context}

            Human: {question}
            Assistant:"""

        PROMPT = ChatPromptTemplate.from_template(template)
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 6, 'lambda_mult': 0.25}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=True
        )
