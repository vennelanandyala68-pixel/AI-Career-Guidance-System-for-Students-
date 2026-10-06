import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
st.set_page_config(
    page_title="AI Career Guidance",
    page_icon="🚀",
    layout="wide"
)
st.markdown("""
<style>
.stApp {
    background: #07152f;
    color: white;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #38bdf8;
}
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 30px;
}
.card {
    background: #10254a;
    padding: 22px;
    border-radius: 15px;
    margin: 15px 0;
}
.result {
    background: #12345b;
    padding: 18px;
    border-radius: 12px;
    margin: 10px 0;
}
.stButton > button {
    width: 100%;
    background: #1683e8;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)
st.markdown(
    '<div class="title">🚀 AI Career Guidance System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'Find the right career based on your skills and interests'
    '</div>',
    unsafe_allow_html=True
)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
client = chromadb.PersistentClient(
    path="./career_db"
)
collection = client.get_or_create_collection(
    name="careers"
)
career_data = [
    {
        "id": "software",
        "name": "Software Developer",
        "text": """
        Software Developer: Python, Java, C++, SQL,
        data structures, algorithms, Git, programming,
        problem solving and application development.
        """
    },
    {
        "id": "aiml",
        "name": "AI / Machine Learning Engineer",
        "text": """
        AI Machine Learning Engineer: Python, machine
        learning, deep learning, NLP, computer vision,
        mathematics, statistics and artificial intelligence.
        """
    },
    {
        "id": "data",
        "name": "Data Scientist",
        "text": """
        Data Scientist: Python, pandas, statistics,
        machine learning, SQL, data visualization,
        mathematics and data analysis.
        """
    },
    {
        "id": "analyst",
        "name": "Data Analyst",
        "text": """
        Data Analyst: SQL, Excel, Python, pandas,
        statistics, Power BI, Tableau and visualization.
        """
    },
    {
        "id": "cyber",
        "name": "Cybersecurity Analyst",
        "text": """
        Cybersecurity Analyst: networking, Linux, Python,
        ethical hacking, cryptography, security and
        vulnerability analysis.
        """
    },
    {
        "id": "cloud",
        "name": "Cloud Engineer",
        "text": """
        Cloud Engineer: AWS, Azure, Linux, networking,
        Docker, DevOps and cloud computing.
        """
    },
    {
        "id": "web",
        "name": "Web Developer",
        "text": """
        Web Developer: HTML, CSS, JavaScript, React,
        Node.js, frontend and backend development.
        """
    },
    {
        "id": "embedded",
        "name": "Embedded Systems Engineer",
        "text": """
        Embedded Systems Engineer: C, C++, Arduino,
        ESP32, microcontrollers, sensors, electronics
        and IoT.
        """
    }
]
if collection.count() == 0:
    documents = [
        item["text"] for item in career_data
    ]
    ids = [
        item["id"] for item in career_data
    ]
    embeddings = model.encode(
        documents
    ).tolist()
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )
st.markdown("""
<div class="card">
<h2>👨‍🎓 Student Profile</h2>
</div>
""", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    name = st.text_input(
        "Student Name"
    )
    degree = st.selectbox(
        "Degree",
        [
            "B.Tech",
            "B.Sc",
            "BCA",
            "MCA",
            "M.Tech",
            "MBA",
            "Other"
        ]
    )
with col2:
    branch = st.text_input(
        "Branch",
        placeholder="CSE / ECE / EEE"
    )
    level = st.selectbox(
        "Experience Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )
skills = st.text_input(
    "💻 Your Skills",
    placeholder="Python, C, SQL, HTML..."
)
interests = st.text_input(
    "❤️ Your Interests",
    placeholder="AI, Web Development, IoT..."
)
if st.button(
    "🚀 Find My Career",
    use_container_width=True
):
    if not skills or not interests:
        st.warning(
            "Please enter your skills and interests."
        )
    else:
        # Student profile
        profile = f"""
        Degree: {degree}
        Branch: {branch}
        Level: {level}
        Skills: {skills}
        Interests: {interests}
        """
        # Create embedding
        profile_embedding = model.encode(
            [profile]
        ).tolist()
        # Search ChromaDB
        results = collection.query(
            query_embeddings=profile_embedding,
            n_results=3
        )
        matched = results["documents"][0]
        st.markdown("""
        <div class="card">
        <h2>🎯 Recommended Careers</h2>
        </div>
        """, unsafe_allow_html=True)
        career_names = []
        for document in matched:
            for item in career_data:
                if item["text"].strip() == document.strip():
                    career_names.append(
                        item["name"]
                    )
                    st.markdown(
                        f"""
                        <div class="result">
                        💼 <b>{item["name"]}</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        context = "\n".join(matched)
        prompt = f"""
You are an AI Career Guidance Assistant.
Student Profile:
{profile}
Recommended Career Information:
{context}
Give simple and practical career guidance.
Include:
1. Best career choice
2. Why it suits the student
3. Skills to learn
4. 6-month learning roadmap
5. Project ideas
6. Job roles
7. Higher study options
8. Final advice
Do not invent information about the student's
skills or education.
"""
        with st.spinner(
            "🤖 AI is preparing your career guidance..."
        ):
            try:
                response = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                answer = response[
                    "message"
                ]["content"]
                st.markdown("""
                <div class="card">
                <h2>🤖 AI Career Guidance</h2>
                </div>
                """, unsafe_allow_html=True)
                st.write(answer)
            except Exception as error:
                st.error(
                    "Ollama connection failed."
                )
                st.code(str(error))
st.markdown("""
<hr>
<center>
🚀 <b>AI Career Guidance System</b><br>
Python • Streamlit • Sentence Transformers •
ChromaDB • Ollama • Llama 3.2
</center>
""", unsafe_allow_html=True)