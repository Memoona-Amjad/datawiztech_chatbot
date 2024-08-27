import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import json
import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import CSVLoader
from io import StringIO
import docx2txt
import PyPDF2
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import base64
# Load environment variables
#load_dotenv()

def decode_api_key(encoded_api_key):
    decoded_bytes = base64.b64decode(encoded_api_key.encode('utf-8'))
    decoded_str = str(decoded_bytes, 'utf-8')
    return decoded_str

api_key = decode_api_key("QUl6YVN5QUdmQTh5SnpqUkRjbV80YmJ3SDJsWmE2R1I2blJUTFR3")

# Configure Streamlit page settings
gemini_llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=api_key, temperature=0)
st.set_page_config(
    page_title="datawiztech",
    page_icon="📈",
    layout="centered",
)

# Load existing users from the JSON file
def load_user_db():
    if os.path.exists("users_db.json"):
        with open("users_db.json", "r") as file:
            return json.load(file)
    else:
        return {}

# Save user data to JSON file
def save_user_db(users_db):
    with open("users_db.json", "w") as file:
        json.dump(users_db, file)

# Function for user authentication
def authenticate(username, password):
    users = load_user_db()
    return users.get(username) == password

# Function for user signup
def signup(username, password):
    users_db = load_user_db()
    if username in users_db:
        st.error("Username already exists! Please choose a different one.")
    else:
        users_db[username] = password
        save_user_db(users_db)  # Save the updated user database to JSON
        st.success("Signup successful! You can now log in.")
        st.session_state.page = "login"

# Function for user login
def login(username, password):
    if authenticate(username, password):
        st.session_state.logged_in = True
        st.session_state.page = "chatbot"
    else:
        st.error("Invalid username or password")

# Function for user logout
def logout():
    st.session_state.logged_in = False
    st.session_state.page = "login"
    st.success("Logged out successfully!")

# Manage page navigation
def navigate_to(page):
    st.session_state.page = page

# Initialize session state for login status and navigation
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"  # Default page is login
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "General Chatbot"

# Login Page
def login_page():
    st.markdown("""
        <style>
        .stTextInput > div > input {
            border-radius: 12px;
            padding: 12px;
            border: 2px solid #4CAF50;
        }
        .stButton > button {
            background-color: #388E3C;
            color: white;
            border-radius: 12px;
            padding: 12px;
            width: 100%;
            border: none;
            font-size: 1.1rem;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #2E7D32;
        }
        .stTitle {
            text-align: center;
            color: white;
        }
        .stMarkdown {
            text-align: center;
            color: black;
        }
        .logo-container {
            text-align: center;
            margin-bottom: 2rem;
            justify-content: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Add a logo
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image("image.png", width=350)
    st.markdown('</div>', unsafe_allow_html=True)

    st.title("Login")
    st.markdown("## Please log in to access the chatbot.")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        login(username, password)
    
    st.markdown("---")
    st.markdown("Don't have an account?")
    st.button("Sign up", on_click=lambda: setattr(st.session_state, "page", "signup"))
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced UI for the signup page
def signup_page():
    st.markdown("""
        <style>
        .login-signup-container {
            background: linear-gradient(135deg, #a3d9a5, #4CAF50);
            padding: 3rem;
            border-radius: 15px;
            max-width: 400px;
            margin: auto;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
        }
        .stTextInput > div > input {
            border-radius: 12px;
            padding: 12px;
            border: 2px solid #4CAF50;
        }
        .stButton > button {
            background-color: #388E3C;
            color: white;
            border-radius: 12px;
            padding: 12px;
            width: 100%;
            border: none;
            font-size: 1.1rem;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #2E7D32;
        }
        .stTitle {
            text-align: center;
            color: white;
        }
        .stMarkdown {
            text-align: center;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)

    #st.markdown('<div class="login-signup-container">', unsafe_allow_html=True)
    st.title("Sign Up")
    st.markdown("## Create a new account.")
    
    username = st.text_input("Create a Username")
    password = st.text_input("Create a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Sign Up"):
        if password == confirm_password:
            signup(username, password)
        else:
            st.error("Passwords do not match.")
    
    st.markdown("---")
    st.markdown("Already have an account?")
    st.button("Login", on_click=lambda: setattr(st.session_state, "page", "login"))
    st.markdown('</div>', unsafe_allow_html=True)

# Chatbot UI
def chatbot_page():
    st.title("🤖 datawiztech chat bot")

    # Sidebar with logout, mode selection, and upload PDF button
    with st.sidebar:
        st.subheader("User Options")
        
        # Dropdown for selecting chat mode
        st.session_state.chat_mode = st.selectbox(
            "Select Chat Mode",
            [ 
                "Enterprenureship Chatbot", 
                "Chat with Files" 
            ],
            index=0
        )

       
        
        if st.session_state.chat_mode == "Chat with Files":
            st.header("Upload your file")
            uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "docx", "pdf", "txt", "md"])
            
         # Logout button    
        if st.button("Logout", key="logout"):
            logout()
            

    # Apply custom CSS for chat UI
    st.markdown(
        """
        <style>
        body {
            background-color: #eaf4ea;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
        }
        .stTextInput>div>input {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 10px;
        }
        .stTitle {
            color: #4CAF50;
            text-align: center;
        }
        .stMarkdown {
            color: black;
        }
        .stFileUploader>label {
            color: #4CAF50;
        }
        .user-message {
            background-color: #8db698;
            border-radius: 15px;
            padding: 10px;
            margin: 5px;
            text-align: right;
            width: fit-content;
            float: right;
            clear: both;
        }
        .assistant-message {
            background-color: #bddabb;
            border-radius: 15px;
            padding: 10px;
            margin: 5px;
            text-align: left;
            width: fit-content;
            float: left;
            clear: both;
        }
        .chat-container {
            max-width: 600px;
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display chat UI based on selected mode
    if st.session_state.chat_mode == "Enterprenureship Chatbot":
        entrepeneurship_chatbot()
    elif st.session_state.chat_mode == "Chat with Files":
        fileschatbot(uploaded_file)

def entrepeneurship_chatbot():
    # Set up Google Gemini-Pro AI model
    gen_ai.configure(api_key=api_key)
    model = gen_ai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction="""
        Stay in the following context, if user types in a specific language reply him in that language.
        You are a specialized AI chatbot for entrepreneurs with only the following functions:

        1. Task Management: Help users manage their tasks efficiently by providing reminders and tracking progress.
        2. Business Advice: Offer actionable business advice based on user input and relevant data.
        3. Data Analysis and Reporting: Analyze business data and generate detailed reports.
        4. Expense Tracking: Allow users to track and categorize their expenses.
        5. Market Research: Perform market research by querying relevant APIs or analyzing business documents.
        6. Multilingual Support: Communicate in any language the user prefers, including local languages, to ensure effective interaction with a diverse audience.
        Important: Focus solely on these tasks, ensuring accuracy and efficiency in each function. also ask user about the above mentioned functionality after greeting or in your first interaction so that user can know whyare you here""")

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    # Display chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display the chat history
    for message in st.session_state.chat_session.history:
        role = message.role
        if role == 'user':
            st.markdown(f'<div class="user-message">{message.parts[0].text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message.parts[0].text}</div>', unsafe_allow_html=True)

    # Input field for user's message
    user_prompt = st.chat_input("Ask datawiztechaAI...")
    if user_prompt:
        # Add user's message to chat and display it
        st.markdown(f'<div class="user-message">{user_prompt}</div>', unsafe_allow_html=True)

        # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        # Display Gemini-Pro's response
        st.markdown(f'<div class="assistant-message">{gemini_response.text}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def fileschatbot(uploaded_file):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    #uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "docx", "pdf", "txt", "md"])
    if uploaded_file is not None:
        # Determine the file type
        file_extension = uploaded_file.name.split('.')[-1]

        # Handle CSV and Excel files separately
        if file_extension in ['csv', 'xlsx']:
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'xlsx':
                df = pd.read_excel(uploaded_file)

            # Save the uploaded data to a temporary CSV file
            csv_file_path = 'temp_uploaded_file.csv'
            df.to_csv(csv_file_path, index=False)

            # Pass the CSV file path directly to create_csv_agent
            agent = create_csv_agent(gemini_llm, csv_file_path, allow_dangerous_code=True, verbose=True)

        # Handle other file types (Word, PDF, Text, Markdown)
        else:
            if file_extension == 'docx':
                text = docx2txt.process(uploaded_file)
            elif file_extension == 'pdf':
                reader = PyPDF2.PdfReader(uploaded_file)
                text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
            elif file_extension in ['txt', 'md']:
                text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()

            prompt_template = "You are a helpful assistant. Here is the content:\n{text}\nAnswer the question: {question}"
            prompt = PromptTemplate(template=prompt_template, input_variables=["text", "question"])

            chain = LLMChain(llm=gemini_llm, prompt=prompt)

        # Display all previous chat messages
        for message in st.session_state.chat_history:
            st.markdown(message, unsafe_allow_html=True)

        # Message input area at the bottom
        user_query = st.chat_input("Type your question here:", key="query_input")
        if user_query:
            if file_extension in ['csv', 'xlsx']:
                response = agent.run(user_query)
            else:
                response = chain.run({"text": text, "question": user_query})

            # Format messages for right-aligned user input and left-aligned bot response
            user_message = f'''
            <div style="text-align:right;">
                <div style="display:inline-block;background-color:#E8F5E9;border-radius:10px;padding:10px;margin:10px 0;max-width:80%;">{user_query}</div>
            </div>'''
            
            bot_message = f'''
            <div style="text-align:left;">
                <div style="display:inline-block;background-color:#bddabb;border-radius:10px;padding:10px;margin:10px 0;max-width:80%;">{response}</div>
            </div>'''

            # Append new messages to chat history
            st.session_state.chat_history.append(user_message)
            st.session_state.chat_history.append(bot_message)

            # Display the updated chat history
            st.markdown(user_message, unsafe_allow_html=True)
            st.markdown(bot_message, unsafe_allow_html=True)

    else:
        st.write("Please upload a CSV, Excel, Word, PDF, Text, or Markdown file to begin.")

# Page Navigation Logic
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "signup":
    signup_page()
elif st.session_state.page == "chatbot" and st.session_state.logged_in:
    chatbot_page()
else:
    navigate_to("login")
