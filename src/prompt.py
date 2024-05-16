# prompt_template="""
# Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """


prompt_template="""
Answer the user's question using the following information:

Context: {context}
Question: {question}

Provide only the most relevant answer.
"""