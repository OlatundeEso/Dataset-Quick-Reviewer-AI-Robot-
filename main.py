
import os
# import langchain
# import openai
# from langchain import HuggingFaceHub
import pandas as pd
import tabulate
import streamlit as st
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# my_prompt = PromptTemplate(input_variables=["country"], template="Tell me the capital of {country}")
load_dotenv()  # Load variables from .env file
my_llm = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"), temperature=0.6)
# result = my_llm("Tell me what the capital of Turkey is")
# print(result)
# result_2 = my_llm(my_prompt.format(country="Nineveh"))  # You can prompt the llm straight
# print(result_2)
# result_3 = my_llm.predict(my_prompt.format(country="Ghana"))  # You can use .predict
# print(result_3)
# Title
st.title('AI Assistant for Data Science 🤖😇')
# Welcoming message
st.write("### Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")

# Designing a Sidebar into the WebApp

# Keeping the Sidebar "Open By Default"

st.markdown(
    """
    <style>
        div[data-testid="stSidebar"][aria-expanded="false"] {
            display: block !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# The Explanation-sidebar Proper
with st.sidebar:
    st.write('*Let Us Begin By Asking You to Share Your FIle (Preferable a csv file)*')
    st.caption('''**You may already know that every exciting data science journey starts with a dataset.
    That's why I'd love for you to upload a CSV file.
    Once we have your data in hand, we'll dive into understanding it and have some fun exploring it.
    Then, we'll work together to shape your business challenge into a data science framework.
    I'll introduce you to the coolest machine learning models, and we'll use them to tackle your problem. Sounds fun right?**
    ''')
    st.divider()

    st.caption("<p style ='text-align:center'> Proudly Created by Olatunde Eso</p>", unsafe_allow_html=True)

# Initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}


# Function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True


st.button("Let's get started", on_click=clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)
        st.success("Looks Good!✌")

        # llm model
        # llm = OpenAI(temperature=0)

        # Function sidebar
        @st.cache_data
        def steps_eda():
            # steps_eda = my_llm('What are the steps of EDA')
            eda_steps = my_llm('What are the steps of EDA')
            # return steps_eda
            return eda_steps


        # Pandas agent
        pandas_agent = create_pandas_dataframe_agent(my_llm, df, verbose=True)

        # Functions main
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(columns_df)
            missing_values = pandas_agent.run(
                "How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(duplicates)
            st.write("**Data Summarization**")
            st.write(df.describe())
            correlation_analysis = pandas_agent.run(
                "Calculate correlations between numerical variables to identify potential relationships.")
            st.write(correlation_analysis)
            outliers = pandas_agent.run(
                "Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            st.write(outliers)
            new_features = pandas_agent.run("What new features would be interesting to create?.")
            st.write(new_features)
            return


        @st.cache_data
        def function_question_variable():
            st.line_chart(df, y=[user_question_variable])
            summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
            st.write(summary_statistics)
            normality = pandas_agent.run(
                f"Check for normality or specific distribution shapes of {user_question_variable}")
            st.write(normality)
            outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
            st.write(outliers)
            trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
            st.write(trends)
            missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
            st.write(missing_values)
            return


        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return


        # Main

        st.header('Exploratory data analysis')
        st.subheader('General information about the dataset')

        with st.sidebar:
            with st.expander('What are the steps of EDA'):
                st.write(steps_eda())

        function_agent()

        st.subheader('Variable of study')
        user_question_variable = st.text_input('What variable are you interested in')
        if user_question_variable is not None and user_question_variable != "":
            function_question_variable()

            st.subheader('Further study')

        if user_question_variable:
            user_question_dataframe = st.text_input(
                "Is there anything else you would like to know about your dataframe?")
            if user_question_dataframe is not None and user_question_dataframe not in ("", "no", "No"):
                function_question_dataframe()
            if user_question_dataframe in ("no", "No"):
                st.write("")



# Refresh button
if st.button("Refresh"):
    # Use JavaScript to refresh the page
    st.markdown(
        """<script>
        function refreshPage() {
            location.reload();
        }
        </script>
        """,
        unsafe_allow_html=True,
    )





#
#
#
# # Import required libraries
# import os
# from apikey import apikey
#
# import streamlit as st
# import pandas as pd
#
# from langchain.my_llm import OpenAI
# from langchain.agents import create_pandas_dataframe_agent
# from dotenv import load_dotenv, find_dotenv
#
# # OpenAIKey
# os.environ['OPENAI_API_KEY'] = apikey
# load_dotenv(find_dotenv())
#
# # Title
# st.title('AI Assistant for Data Science 🤖')
#
# # Welcoming message
# st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")
#
# # Explanation sidebar
# with st.sidebar:
#     st.write('*Your Data Science Adventure Begins with an CSV File.*')
#     st.caption('''**You may already know that every exciting data science journey starts with a dataset.
#     That's why I'd love for you to upload a CSV file.
#     Once we have your data in hand, we'll dive into understanding it and have some fun exploring it.
#     Then, we'll work together to shape your business challenge into a data science framework.
#     I'll introduce you to the coolest machine learning models, and we'll use them to tackle your problem. Sounds fun right?**
#     ''')
#
#     st.divider()
#
#     st.caption("<p style ='text-align:center'> made with ❤️ by Ana</p>", unsafe_allow_html=True)
#
# # Initialise the key in session state
# if 'clicked' not in st.session_state:
#     st.session_state.clicked = {1: False}
#
#
# # Function to update the value in session state
# def clicked(button):
#     st.session_state.clicked[button] = True
#
#
# st.button("Let's get started", on_click=clicked, args=[1])
# if st.session_state.clicked[1]:
#     user_csv = st.file_uploader("Upload your file here", type="csv")
#     if user_csv is not None:
#         user_csv.seek(0)
#         df = pd.read_csv(user_csv, low_memory=False)
#
#         # llm model
#         llm = OpenAI(temperature=0)
#
#
#         # Function sidebar
#         @st.cache_data
#         def steps_eda():
#             steps_eda = llm('What are the steps of EDA')
#             return steps_eda
#
#
#         # Pandas agent
#         pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True)
#
#
#         # Functions main
#         @st.cache_data
#         def function_agent():
#             st.write("**Data Overview**")
#             st.write("The first rows of your dataset look like this:")
#             st.write(df.head())
#             st.write("**Data Cleaning**")
#             columns_df = pandas_agent.run("What are the meaning of the columns?")
#             st.write(columns_df)
#             missing_values = pandas_agent.run(
#                 "How many missing values does this dataframe have? Start the answer with 'There are'")
#             st.write(missing_values)
#             duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
#             st.write(duplicates)
#             st.write("**Data Summarization**")
#             st.write(df.describe())
#             correlation_analysis = pandas_agent.run(
#                 "Calculate correlations between numerical variables to identify potential relationships.")
#             st.write(correlation_analysis)
#             outliers = pandas_agent.run(
#                 "Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
#             st.write(outliers)
#             new_features = pandas_agent.run("What new features would be interesting to create?.")
#             st.write(new_features)
#             return
#
#
#         @st.cache_data
#         def function_question_variable():
#             st.line_chart(df, y=[user_question_variable])
#             summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
#             st.write(summary_statistics)
#             normality = pandas_agent.run(
#                 f"Check for normality or specific distribution shapes of {user_question_variable}")
#             st.write(normality)
#             outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
#             st.write(outliers)
#             trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
#             st.write(trends)
#             missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
#             st.write(missing_values)
#             return
#
#
#         @st.cache_data
#         def function_question_dataframe():
#             dataframe_info = pandas_agent.run(user_question_dataframe)
#             st.write(dataframe_info)
#             return
#
#
#         # Main
#
#         st.header('Exploratory data analysis')
#         st.subheader('General information about the dataset')
#
#         with st.sidebar:
#             with st.expander('What are the steps of EDA'):
#                 st.write(steps_eda())
#
#         function_agent()
#
#         st.subheader('Variable of study')
#         user_question_variable = st.text_input('What variable are you interested in')
#         if user_question_variable is not None and user_question_variable != "":
#             function_question_variable()
#
#             st.subheader('Further study')
#
#         if user_question_variable:
#             user_question_dataframe = st.text_input(
#                 "Is there anything else you would like to know about your dataframe?")
#             if user_question_dataframe is not None and user_question_dataframe not in ("", "no", "No"):
#                 function_question_dataframe()
#             if user_question_dataframe in ("no", "No"):
#                 st.write("")
# """
# Stateful button
# If you want a clicked button to continue to be True, create a value in st.session_state and use the button to set that value to True in a callback.
#
# """

# import streamlit as st
#
# if 'clicked' not in st.session_state:
#     st.session_state.clicked = False
#
# def click_button():
#     st.session_state.clicked = True
#
# st.button('Click me', on_click=click_button)
#
# if st.session_state.clicked:
#     # The message and nested widget will remain on the page
#     st.write('Button clicked!')
#     st.slider('Select a value')
