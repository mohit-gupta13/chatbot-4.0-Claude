import langchain
import langchain.agents
import sys

with open("debug_langchain.txt", "w") as f:
    f.write(f"LangChain Version: {langchain.__version__}\n")
    f.write(f"LangChain Path: {langchain.__file__}\n")
    f.write(f"LangChain Agents Path: {langchain.agents.__file__}\n")
    f.write("Attributes in langchain.agents:\n")
    for item in dir(langchain.agents):
        f.write(f"{item}\n")
