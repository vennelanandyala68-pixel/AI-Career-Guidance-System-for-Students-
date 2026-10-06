print("Checking installations...\n")

packages = [
    "streamlit",
    "sentence_transformers",
    "chromadb",
    "pypdf",
    "ollama",
    "pandas",
    "sklearn"
]

for package in packages:
    try:
        __import__(package)
        print(f"✅ {package} - Installed")
    except ImportError:
        
        print(f"❌ {package} - NOT Installed")

print("\nDone!")