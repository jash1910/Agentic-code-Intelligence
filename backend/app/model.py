import os
import sys
import time
import torch
from threading import Thread

# Add root folder to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from backend.app.config import settings

class LLMManager:
    def __init__(self):
        self.mode = settings.RUN_MODE
        self.model = None
        self.tokenizer = None
        
        print(f"[ModelManager] Initializing in {self.mode.upper()} mode...")
        
        if self.mode in ["gpu", "local"]:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                # Choose model based on mode
                model_id = settings.MODEL_ID
                if self.mode == "local" and "6.7B" in model_id:
                    # Fallback to smaller model for CPU/MPS local runs
                    model_id = "deepseek-ai/DeepSeek-Coder-1.3B-instruct"
                    print(f"[ModelManager] Local mode active. Downsizing model to {model_id} for performance.")
                
                print(f"[ModelManager] Loading tokenizer for {model_id}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                
                print(f"[ModelManager] Loading model {model_id}...")
                device_map = "auto" if self.mode == "gpu" else None
                
                # Check for MPS or CPU
                device = "cpu"
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                
                model_kwargs = {
                    "trust_remote_code": True,
                    "device_map": device_map
                }
                if device == "mps" and self.mode == "local":
                    model_kwargs["torch_dtype"] = torch.float16
                    
                self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
                
                if device == "mps" and self.mode == "local":
                    self.model = self.model.to("mps")
                
                # Load LoRA adapter if GPU mode and adapter weights exist
                if self.mode == "gpu" and os.path.exists(settings.PEFT_ADAPTER_PATH):
                    print(f"[ModelManager] Loading PEFT adapter weights from {settings.PEFT_ADAPTER_PATH}...")
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(self.model, settings.PEFT_ADAPTER_PATH)
                    print("[ModelManager] Adapter loaded and merged.")
                    
            except Exception as e:
                print(f"[ModelManager] Error initializing model/tokenizer: {e}", file=sys.stderr)
                print("[ModelManager] Falling back to MOCK mode.", file=sys.stderr)
                self.mode = "mock"

    def stream_generate(self, prompt: str):
        """
        Yields text chunks as they are generated.
        """
        if self.mode == "mock":
            yield from self._mock_stream_generate(prompt)
            return

        # Inference logic for transformers
        from transformers import TextIteratorStreamer
        
        # Format inputs
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Put inputs on same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            top_p=0.95
        )
        
        # Run generation in a separate thread so it is non-blocking
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text

    def _mock_stream_generate(self, prompt: str):
        """
        Generates structured, realistic coder assistant responses to test streaming and UI styling.
        """
        # Determine the user request context
        prompt_lower = prompt.lower()
        
        response = ""
        
        if "llmchain" in prompt_lower:
            response = (
                "### Understanding `LLMChain` in LangChain\n\n"
                "The `LLMChain` is one of the most foundational chain classes in LangChain. It combines a **Language Model (LLM)** with a **PromptTemplate** to execute a sequence of prompt formatting, model invocation, and response parsing.\n\n"
                "Here is how you initialize and run a basic `LLMChain`:\n\n"
                "```python\n"
                "from langchain.chains import LLMChain\n"
                "from langchain.prompts import PromptTemplate\n"
                "from langchain_openai import OpenAI\n\n"
                "# 1. Define your prompt structure\n"
                "prompt = PromptTemplate.from_template(\n"
                "    \"Provide a detailed, step-by-step explanation of the concept of {concept}.\"\n"
                ")\n\n"
                "# 2. Initialize the language model\n"
                "llm = OpenAI(temperature=0.7)\n\n"
                "# 3. Combine them into an LLMChain\n"
                "chain = LLMChain(llm=llm, prompt=prompt)\n\n"
                "# 4. Run the chain\n"
                "result = chain.run(concept=\"Retrieval-Augmented Generation (RAG)\")\n"
                "print(result)\n"
                "```\n\n"
                "#### Key Components:\n"
                "- **`prompt`**: Formats the input variables into a string prompt.\n"
                "- **`llm`**: The language model that receives the formatted prompt and returns the output.\n"
                "- **`output_parser`** *(Optional)*: Parsers that format raw string completions into structured types (like JSON or Python objects).\n\n"
                "> [!NOTE]\n"
                "> In LangChain's newer versions, `LLMChain` is deprecated in favor of **LCEL** (LangChain Expression Language). "
                "The equivalent LCEL syntax is:\n"
                "> `chain = prompt | llm`"
            )
        elif "chatprompttemplate" in prompt_lower or "prompttemplate" in prompt_lower:
            response = (
                "### Constructing `ChatPromptTemplate` in LangChain\n\n"
                "A `ChatPromptTemplate` is designed specifically for chat-based LLMs that expect a structured list of messages (System, Human, AI) rather than a single raw string prompt.\n\n"
                "Here is the standard implementation pattern:\n\n"
                "```python\n"
                "from langchain.prompts import ChatPromptTemplate\n\n"
                "# Create a prompt template with roles\n"
                "prompt = ChatPromptTemplate.from_messages([\n"
                "    (\"system\", \"You are a helpful database administrator named {name}.\"),\n"
                "    (\"human\", \"What is the SQL query to fetch the top 5 customers?\"),\n"
                "    (\"ai\", \"SELECT * FROM customers ORDER BY revenue DESC LIMIT 5;\"),\n"
                "    (\"human\", \"Explain this query in {language}.\")\n"
                "])\n\n"
                "# Format the messages with variables\n"
                "formatted_messages = prompt.format_messages(name=\"Dexter\", language=\"Spanish\")\n"
                "print(formatted_messages)\n"
                "```\n\n"
                "#### Why use `ChatPromptTemplate`?\n"
                "1. **Structured Inputs**: Keeps the system context clean and separate from user inputs.\n"
                "2. **Safety**: Makes it harder for user prompts to hijack the system instruction.\n"
                "3. **Multi-turn Dialogues**: Simplifies keeping track of the chat history by appending messages to the list."
            )
        elif "agent" in prompt_lower or "react" in prompt_lower:
            response = (
                "### Creating Custom Agents in LangChain\n\n"
                "Agents in LangChain use an LLM to decide a sequence of actions to take. Unlike chains where the sequence is hardcoded, agents use the LLM as a reasoning engine to dynamically determine which tools to call and in what order.\n\n"
                "Here is how you set up an agent using the modern **LangChain Expression Language (LCEL)** and `AgentExecutor`:\n\n"
                "```python\n"
                "from langchain.agents import AgentExecutor, create_tool_calling_agent\n"
                "from langchain_core.tools import tool\n"
                "from langchain_openai import ChatOpenAI\n"
                "from langchain.prompts import ChatPromptTemplate\n\n"
                "# 1. Define custom tools\n"
                "@tool\n"
                "def multiply(a: int, b: int) -> int:\n"
                "    \"\"\"Multiply two integers together.\"\"\"\n"
                "    return a * b\n\n"
                "tools = [multiply]\n\n"
                "# 2. Initialize LLM supporting tool calling\n"
                "llm = ChatOpenAI(model=\"gpt-3.5-turbo\", temperature=0)\n\n"
                "# 3. Set up the Prompt\n"
                "prompt = ChatPromptTemplate.from_messages([\n"
                "    (\"system\", \"You are a helpful assistant.\"),\n"
                "    (\"human\", \"{input}\"),\n"
                "    (\"placeholder\", \"{agent_scratchpad}\")\n"
                "])\n\n"
                "# 4. Construct Agent\n"
                "agent = create_tool_calling_agent(llm, tools, prompt)\n"
                "executor = AgentExecutor(agent=agent, tools=tools, verbose=True)\n\n"
                "# 5. Execute\n"
                "result = executor.invoke({\"input\": \"What is 432 multiplied by 54?\"})\n"
                "print(result[\"output\"])\n"
                "```"
            )
        else:
            response = (
                "### Welcome to the LangChain AI Coding Assistant!\n\n"
                "I am a fine-tuned model trained on LangChain internals, combined with active RAG context.\n\n"
                "You can ask me questions such as:\n"
                "1. **How do I use `LLMChain`?**\n"
                "2. **How to define a custom `ChatPromptTemplate`?**\n"
                "3. **What is the structure of a custom Agent or Tool?**\n\n"
                "I will retrieve the code from LangChain's repository and explain the implementations with proper citations."
            )

        # Simulate streaming by splitting response by word/chunk
        words = response.split(" ")
        chunk_size = 3
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size]) + " "
            yield chunk
            time.sleep(0.04) # Simulate network latency and token generation
