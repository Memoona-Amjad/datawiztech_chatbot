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
import bcrypt  # Added for password hashing

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
gemini_llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=os.environ["GOOGLE_API_KEY"], temperature=0)
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

# Function to hash the password
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Function to check if the provided password matches the hashed password
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# Function for user authentication (now with email and hashed password)
def authenticate(email, password):
    users = load_user_db()
    user = users.get(email)
    if user and check_password(password, user['password']):
        return True
    return False

# Function for user signup (with additional fields and password hashing)
def signup(first_name, last_name, email, password):
    with st.spinner("Creating account..."):  # Spinner added here
        users_db = load_user_db()
        if email in users_db:
            st.error("Email already registered! Please log in.")
        elif len(password) < 6:
            st.error("Password must be at least 6 characters long.")
        else:
            hashed_password = hash_password(password)
            users_db[email] = {
                "first_name": first_name,
                "last_name": last_name,
                "password": hashed_password
            }
            save_user_db(users_db)
            st.success("Signup successful! You can now log in.")
            st.session_state.page = "login"

# Function for user login
def login(email, password):
    with st.spinner("Authenticating..."):  # Spinner added here
        if authenticate(email, password):
            st.session_state.logged_in = True
            st.session_state.page = "chatbot"
        else:
            st.error("Invalid email or password")

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
    
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        login(email, password)
    
    st.markdown("---")
    st.markdown("Don't have an account?")
    st.button("Sign up", on_click=lambda: setattr(st.session_state, "page", "signup"))
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced UI for the signup page
def signup_page():
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
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-signup-container">', unsafe_allow_html=True)
    st.title("Sign Up")
    st.markdown("## Create a new account.")
    
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    email = st.text_input("Email Address")
    password = st.text_input("Create a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Sign Up"):
        if len(password) < 6:
            st.error("Password must be at least 6 characters long.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            signup(first_name, last_name, email, password)
    
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

        # Upload multiple files in Chat with Files mode
        if st.session_state.chat_mode == "Chat with Files":
            st.header("Upload your files")
            uploaded_files = st.file_uploader("Choose files", type=["csv", "xlsx", "docx", "pdf", "txt", "md"], accept_multiple_files=True)
            
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
            clear: both.
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
        fileschatbot(uploaded_files)

def entrepeneurship_chatbot():
    # Set up Google Gemini-Pro AI model
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    gen_ai.configure(api_key=GOOGLE_API_KEY)
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

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for message in st.session_state.chat_session.history:
        role = message.role
        if role == 'user':
            st.markdown(f'<div class="user-message">{message.parts[0].text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message.parts[0].text}</div>', unsafe_allow_html=True)

    user_prompt = st.chat_input("Ask datawiztechaAI...")
    if user_prompt:
        st.markdown(f'<div class="user-message">{user_prompt}</div>', unsafe_allow_html=True)

        with st.spinner("Thinking..."):  # Spinner added here
            gemini_response = st.session_state.chat_session.send_message(user_prompt)

        st.markdown(f'<div class="assistant-message">{gemini_response.text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def fileschatbot(uploaded_files):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    combined_text = ""

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split('.')[-1]

            if file_extension in ['csv', 'xlsx']:
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                elif file_extension == 'xlsx':
                    df = pd.read_excel(uploaded_file)
                combined_text += df.to_string(index=False) + "\n"
            else:
                if file_extension == 'docx':
                    combined_text += docx2txt.process(uploaded_file) + "\n"
                elif file_extension == 'pdf':
                    reader = PyPDF2.PdfReader(uploaded_file)
                    combined_text += "".join([page.extract_text() for page in reader.pages if page.extract_text()]) + "\n"
                elif file_extension in ['txt', 'md']:
                    combined_text += StringIO(uploaded_file.getvalue().decode("utf-8")).read() + "\n"

        prompt_template = "You are a helpful assistant. Here is the combined content:\n{text}\nAnswer the question: {question}"
        prompt = PromptTemplate(template=prompt_template, input_variables=["text", "question"])
        chain = LLMChain(llm=gemini_llm, prompt=prompt)

        for message in st.session_state.chat_history:
            st.markdown(message, unsafe_allow_html=True)

        user_query = st.chat_input("Type your question here:", key="query_input")
        if user_query:
            with st.spinner("Analyzing your files and generating a response..."):  # Spinner added here
                response = chain.run({"text": combined_text, "question": user_query})

            user_message = f'''
            <div style="text-align:right;">
                <div style="display:inline-block;background-color:#E8F5E9;border-radius:10px;padding:10px;margin:10px 0;max-width:80%;">{user_query}</div>
            </div>'''
            
            bot_message = f'''
            <div style="text-align:left;">
                <div style="display:inline-block;background-color:#bddabb;border-radius:10px;padding:10px;margin:10px 0;max-width:80%;">{response}</div>
            </div>'''

            st.session_state.chat_history.append(user_message)
            st.session_state.chat_history.append(bot_message)

            st.markdown(user_message, unsafe_allow_html=True)
            st.markdown(bot_message, unsafe_allow_html=True)

    else:
        st.write("Please upload CSV, Excel, Word, PDF, Text, or Markdown files to begin.")

# Page Navigation Logic
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "signup":
    signup_page()
elif st.session_state.page == "chatbot" and st.session_state.logged_in:
    chatbot_page()
else:
    navigate_to("login")
