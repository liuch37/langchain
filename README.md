# langchain
LangChain APP building examples. Note that using OpenAI API IS NOT FREE.

## installation
1. Create Python environment.
```
conda create --name lamgchain python=3.10
```

2. Install libraries.
```
pip install -r requirements.txt
```

3. Create your environmental variables by adding/editing ```.env```. Add your Google API key, LangChain API key, OpenAI API key, etc in it. If you don't have the keys, go and apply them online for free.

## run app
```
streamlit run gemini_app_qa.py
```