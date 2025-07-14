import streamlit as st
from langchain_community.tools import WikipediaQueryRun 
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser

# WIKIPEDIA
wikipedia_search= WikipediaQueryRun(api_wrapper= WikipediaAPIWrapper())

# PROMPT
prompt= PromptTemplate(
    template= """
    Take the following text and summarize it in a short, simple, and funny story format, like you're telling it to a 10-year-old with a big imagination. Add playful analogies, silly metaphors, and keep it light and entertaining — but still accurate.If possible then try to Generate the response in Hinglish Language. Limit the number of words to 150. Also don't repeat my words (like this: Okay, I need to help create a fun story for a 10-year-old based on the info from these pages.), only print the story in the output. That's it!!,
    Here is the text:{content}
    """,
    input_variables= ['content']
)

# LLM MODEL & PARSER
llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Meta-Llama-3-8B-Instruct',
    task= 'text-generation'
)
model= ChatHuggingFace(llm= llm)
parser= StrOutputParser()

# App title
st.title("WikiPedia Summarizer")

# Text input from user
user_input = st.text_area("Enter your text here:")

# Initialize the output variable
summary_output = ""

# Button to trigger summarization
if st.button("Summarize"):
    # === Your summarization code goes here ===
    # For now, we use a placeholder logic:
    # Replace this with your own summarization code
    FinalChain= wikipedia_search| prompt | model | parser
    FunnySummary= FinalChain.invoke(user_input)

    summary_output = FunnySummary

# Display the output board
st.subheader("Summary Output:")
st.write(summary_output if summary_output else "Your summary will appear here after clicking the button.")







