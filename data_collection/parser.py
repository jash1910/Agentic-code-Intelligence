import os
import ast
import json
import random
import re

# Base directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.join(BASE_DIR, "data_collection", "langchain_repo")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Target directories within LangChain repo to extract code
CODE_TARGET_PATHS = [
    os.path.join("libs", "core", "langchain_core"),
    os.path.join("libs", "langchain", "langchain"),
]

# Target directories to extract docs
DOC_TARGET_PATHS = [
    os.path.join("docs", "docs"),
]

class CodeVisitor(ast.NodeVisitor):
    def __init__(self, file_rel_path):
        self.file_rel_path = file_rel_path
        self.qa_pairs = []

    def visit_ClassDef(self, node):
        class_name = node.name
        docstring = ast.get_docstring(node) or ""
        
        # We only want classes with some meaningful structure/docstring
        if len(docstring.strip()) > 10:
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            bases_str = f" inherits from {', '.join(bases)}" if bases else ""
            
            methods_info = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_name = item.name
                    # Skip private/internal methods except __init__
                    if method_name.startswith("_") and method_name != "__init__":
                        continue
                    m_docstring = ast.get_docstring(item) or "No description available."
                    # Extract arg names
                    args = [arg.arg for arg in item.args.args]
                    methods_info.append(f"- `{method_name}({', '.join(args)})`: {m_docstring.split(chr(10))[0]}")
            
            methods_str = "\n".join(methods_info) if methods_info else "No public methods documented."
            
            instruction = f"Explain the structure, base classes, and core interface of the `{class_name}` class in LangChain."
            response = (
                f"The `{class_name}` class is defined in `{self.file_rel_path}`{bases_str}.\n\n"
                f"### Purpose & Description:\n{docstring}\n\n"
                f"### Core Methods and Interface:\n{methods_str}"
            )
            self.qa_pairs.append({"instruction": instruction, "response": response})
            
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Only parse top-level functions (not nested or class methods)
        if hasattr(node, 'parent') and not isinstance(node.parent, ast.Module):
            return
        
        func_name = node.name
        if func_name.startswith("_"):
            return
            
        docstring = ast.get_docstring(node) or ""
        if len(docstring.strip()) > 10:
            args = [arg.arg for arg in node.args.args]
            instruction = f"Explain the usage and signature of the utility function `{func_name}` in LangChain."
            response = (
                f"The function `{func_name}({', '.join(args)})` is defined in `{self.file_rel_path}`.\n\n"
                f"### Description:\n{docstring}"
            )
            self.qa_pairs.append({"instruction": instruction, "response": response})


def parse_python_file(file_path, rel_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        
        tree = ast.parse(code)
        
        # Add parent references to AST nodes to help identify top-level functions
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent
                
        visitor = CodeVisitor(rel_path)
        visitor.visit(tree)
        return visitor.qa_pairs
    except Exception as e:
        print(f"[Parser] Warning: Failed to parse Python file {rel_path}: {e}")
        return []


def parse_markdown_file(file_path, rel_path):
    qa_pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Split markdown by sections (h1, h2, h3 headings)
        sections = re.split(r'\n(#{1,3}\s+.*)\n', content)
        
        # The first item might be intro text before any heading
        intro = sections[0].strip()
        if len(intro) > 100:
            instruction = f"What is the introduction and overview of the topic covered in `{rel_path}`?"
            response = f"Overview of `{rel_path}`:\n\n{intro}"
            qa_pairs.append({"instruction": instruction, "response": response})
            
        # Parse the remaining headings and their content
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                heading_line = sections[i].strip()
                section_content = sections[i+1].strip()
                
                # Clean up heading text
                heading = heading_line.replace("#", "").strip()
                
                if len(section_content) > 100 and heading:
                    instruction = f"What does the LangChain documentation explain regarding '{heading}' in the context of `{rel_path}`?"
                    response = f"According to `{rel_path}`, here is the explanation for **{heading}**:\n\n{section_content}"
                    qa_pairs.append({"instruction": instruction, "response": response})
                    
    except Exception as e:
        print(f"[Parser] Warning: Failed to parse Markdown file {rel_path}: {e}")
    return qa_pairs


def main():
    if not os.path.exists(REPO_DIR):
        print(f"[Parser] Error: Repository directory {REPO_DIR} does not exist. Please run scraper.py first.")
        return

    qa_pairs = []
    print("[Parser] Starting parsing of python files...")
    
    # 1. Parse Py files
    for target_path in CODE_TARGET_PATHS:
        full_target_path = os.path.join(REPO_DIR, target_path)
        if not os.path.exists(full_target_path):
            print(f"[Parser] Path {target_path} not found inside cloned repo. Skipping.")
            continue
            
        for root, _, files in os.walk(full_target_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, REPO_DIR)
                    qa_pairs.extend(parse_python_file(file_path, rel_path))
                    
    print(f"[Parser] Extracted {len(qa_pairs)} pairs from Python files.")
    python_count = len(qa_pairs)

    # 2. Parse Markdown files
    doc_count = 0
    for target_path in DOC_TARGET_PATHS:
        full_target_path = os.path.join(REPO_DIR, target_path)
        if not os.path.exists(full_target_path):
            print(f"[Parser] Path {target_path} not found inside cloned repo. Skipping.")
            continue
            
        for root, _, files in os.walk(full_target_path):
            for file in files:
                if file.endswith('.md') or file.endswith('.mdx'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, REPO_DIR)
                    doc_pairs = parse_markdown_file(file_path, rel_path)
                    qa_pairs.extend(doc_pairs)
                    doc_count += len(doc_pairs)
                    
    print(f"[Parser] Extracted {doc_count} pairs from Markdown documentation.")
    print(f"[Parser] Total dataset size: {len(qa_pairs)} QA pairs.")

    # Fallback to avoid empty datasets if repo doesn't contain expected directories or is empty
    if not qa_pairs:
        print("[Parser] Warning: No QA pairs extracted. Creating synthetic/fallback QA pairs for testing.")
        qa_pairs = [
            {
                "instruction": "Explain how to create a simple LLMChain in LangChain.",
                "response": "To create a simple LLMChain in LangChain, you pair an LLM with a PromptTemplate. Example:\n\n```python\nfrom langchain.chains import LLMChain\nfrom langchain.prompts import PromptTemplate\nfrom langchain_openai import OpenAI\n\nprompt = PromptTemplate.from_template('Tell me a joke about {topic}')\nllm = OpenAI()\nchain = LLMChain(llm=llm, prompt=prompt)\nresult = chain.run(topic='AI')\nprint(result)\n```"
            },
            {
                "instruction": "What is the role of BaseChatMessageHistory in LangChain?",
                "response": "The `BaseChatMessageHistory` class is an abstract base class that defines the core interface for storing and retrieving chat message history. It requires subclasses to implement methods to add messages (e.g. user or AI messages) and retrieve the complete message history list."
            },
            {
                "instruction": "How do you construct a ChatPromptTemplate in LangChain?",
                "response": "A `ChatPromptTemplate` is built using a sequence of messages. You can use from_messages:\n\n```python\nfrom langchain.prompts import ChatPromptTemplate\n\nprompt = ChatPromptTemplate.from_messages([\n    ('system', 'You are a helpful assistant.'),\n    ('user', 'Hello, my name is {name}'),\n])\n```"
            }
        ]

    # Shuffle and split into Train and Validation (90/10 split)
    random.shuffle(qa_pairs)
    split_idx = int(len(qa_pairs) * 0.9)
    train_pairs = qa_pairs[:split_idx]
    val_pairs = qa_pairs[split_idx:]

    # Create dataset directory
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    train_file = os.path.join(DATASET_DIR, "train.jsonl")
    val_file = os.path.join(DATASET_DIR, "val.jsonl")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + '\n')
            
    with open(val_file, 'w', encoding='utf-8') as f:
        for pair in val_pairs:
            f.write(json.dumps(pair) + '\n')
            
    print(f"[Parser] Wrote {len(train_pairs)} pairs to {train_file}")
    print(f"[Parser] Wrote {len(val_pairs)} pairs to {val_file}")

if __name__ == "__main__":
    main()
