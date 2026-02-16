import logging
import torch
from smolagents import CodeAgent, Tool, Model, ChatMessage, MessageRole, ChatMessageStreamDelta, ActionStep, ToolCall, ToolOutput, FinalAnswerStep
from smolagents.models import get_clean_message_list
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer, AutoProcessor, AutoModelForVision2Seq
from threading import Thread, Lock as ThreadLock
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)

class LocalQwenModel(Model):
    """A local model wrapper for smolagents that handles 4-bit quantization correctly."""
    def __init__(self, model_id, device_map="auto", **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        # Handle smolagents input format (list of ChatMessage objects)
        # Use smolagents helper to clean/flatten messages logic
        
        # flatten_messages_as_text=True converts list content [{'type':'text', 'text':'...'}] to pure string
        messages_as_dicts = get_clean_message_list(
            messages,
            flatten_messages_as_text=True
        )
            
        # Convert messages to prompt
        prompt = self.tokenizer.apply_chat_template(messages_as_dicts, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                stop_strings=stop_sequences, # Transformers 4.39+
                tokenizer=self.tokenizer
            )
            
        # Decode only the new tokens
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Return a ChatMessage object as expected by smolagents
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response_text
        )

    def generate_stream(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        **kwargs,
    ):
        messages_as_dicts = get_clean_message_list(
            messages,
            flatten_messages_as_text=True
        )
        prompt = self.tokenizer.apply_chat_template(messages_as_dicts, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            inputs, 
            streamer=streamer, 
            max_new_tokens=512, 
            temperature=0.7, 
            do_sample=True,
            stop_strings=stop_sequences,
            tokenizer=self.tokenizer
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            # logger.info(f"Stream token: {new_text}") # Debug logging
            yield ChatMessageStreamDelta(
                content=new_text
            )

class VisionTool(Tool):
    name = "analyze_screen"
    description = "Analyze the latest screen frame to answer a question. Use this tool when the user asks you to 'see', 'look', or describe what is on the screen. CRITICAL USAGE RULE: DO NOT use 'import' statements to load this tool. It is already injected into your environment. You must call it directly like this: `description = analyze_screen(question='what do you see?')`"
    inputs = {
        "question": {
            "type": "string",
            "description": "The question to ask about the screen content (e.g., 'Describe the error message', 'What is the background color?')."
        },
        "answer": {
            "type": "string",
            "description": "Optional answer argument. (Ignored)",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, model, processor, get_image_func, image_lock, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.processor = processor
        self.get_image_func = get_image_func
        self.image_lock = image_lock # Store the lock passed from OmniAgent

    def update_vision_context(self, base64_image: str):
        """Updates the latest screen frame for the vision tool to use."""
        # This method is intended to be called by OmniAgent to update its latest_image
        # The actual update happens in OmniAgent, this method is a placeholder or for future expansion
        # For now, we assume get_image_func already handles the latest_image from OmniAgent
        pass # The actual update logic will be in OmniAgent, using its lock

    def forward(self, question: str, answer: str = None) -> str:
        # Thread-safe image retrieval
        # The get_image_func already encapsulates the lock from OmniAgent
        image_data = self.get_image_func()
        
        if not image_data:
            return "No screen frame available. Ask the user to share their screen first."
        
        try:
            logger.info(f"VisionTool: Analyzing screen with question: '{question}'")
            # Decode base64
            if "," in image_data:
                 image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # 1. IMAGE DOWNSCALING (Critical for VRAM/Perf)
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            logger.info("VisionTool: Image resized to 512x512.")
            
            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # SmolVLM Inference
            logger.info("VisionTool: Starting Inference (CPU)...")
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            # 2. EXCEPTION HANDLING
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            logger.info("VisionTool: Inference Complete.")
            
            input_len = inputs.input_ids.shape[1]
            new_tokens = generated_ids[:, input_len:]
            response = self.processor.decode(new_tokens[0], skip_special_tokens=True)
            
            # The ultimate SLM override template
            forced_template = f"""SUCCESSFUL OBSERVATION: {response}

CRITICAL SYSTEM DIRECTIVE: The task is complete. You are forbidden from writing conversational text. You MUST copy and paste the EXACT code block below to end the turn:

Code:
```python
final_answer('''{response}''')
```"""
            return forced_template
            
        except Exception as e:
            logger.error(f"VisionTool Error: {e}")
            return f"Error analyzing screen: {str(e)}"

class OmniAgent:
    def __init__(self):
        logger.info("Initializing Real AI Agent (Qwen2.5-Coder-1.5B) with Custom 4-bit Wrapper...")
        
        try:
            # Initialize our custom model wrapper with the SMALLER 1.5B model
            self.model = LocalQwenModel("Qwen/Qwen2.5-Coder-1.5B-Instruct")
            
            # Initialize Vision Model (SmolVLM-500M-Instruct) - CPU Mode (Fallback for Stability)
            # GPU/cuDNN is causing hard crashes ("Could not load symbol cudnnGetLibConfig")
            # likely due to conflict with bitsandbytes or VRAM fragmentation.
            logger.info("Loading Vision Model (SmolVLM-500M-Instruct) on CPU...")
            self.latest_image = None
            self.image_lock = ThreadLock()
            
            self.vision_model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-500M-Instruct",
                torch_dtype=torch.float32, # CPU prefers float32
                device_map="cpu", 
                trust_remote_code=True,
                _attn_implementation="eager"
            )
            self.vision_processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
            
            # Create the Vision Tool
            def get_latest_image_safe():
                with self.image_lock:
                    return self.latest_image
            
            self.vision_tool = VisionTool(self.vision_model, self.vision_processor, get_latest_image_safe, self.image_lock)
            
            # 1.5B model needs VERY explicit instructions to not just chat
            SYSTEM_PROMPT_ADDITION = """
You are a Python coding assistant, created by Nihan.
You MUST follow this format STRICTLY:

1. Think: "Thought: <plan>"
2. Code: A SINGLE Python code block.
3. Return: You MUST use `final_answer(variable)` to return the result.

IMPORTANT RULES:
- ALWAYS import libraries before using them (e.g. `import math`, `import os`, `import datetime`).
- NEVER use variables that are not defined in your code block.
- ALWAYS end your code with `final_answer(variable)`.
- Use `final_answer` ONLY ONCE at the very end of your code.
- PROHIBITED: Do not call `final_answer` multiple times.
- CRITICAL: If the user asks multiple questions, you MUST solve ALL of them and combine the results into a SINGLE string.
- DO NOT use `print()` for the final result.
- DO NOT import imaginary libraries (e.g., `whoami`, `my_library`, `addition`).
- If asked about yourself, just return the string.
- CRITICAL RULE: If the user asks you to look at the screen, see the screen, or describe the screen, you MUST immediately call the `analyze_screen` tool. DO NOT write placeholder code. DO NOT guess what is on the screen. You are blind until you call `analyze_screen`.
- CRITICAL RULE: `analyze_screen` is a BUILT-IN function. DO NOT import it. Just call `analyze_screen(...)`.

EXAMPLES:

User: "Calculate the factorial of 5 and tell me who created you."
You:
Thought: I need to calculate 5! and then state my creator. I will combine both into one answer.
```python
import math
fact = math.factorial(5)
creator = "Nihan"
final_answer(f"The factorial of 5 is {fact}. I was created by {creator}.")
```

User: "Calculate the factorial of 5"
You:
Thought: I need to calculate the factorial of 5 using the math library.
```python
import math
result = math.factorial(5)
final_answer(result)
```

User: "What is the current working directory?"
You:
Thought: I will use the os library to get the current working directory.
```python
import os
cwd = os.getcwd()
final_answer(cwd)
```

User: "Reverse the string 'hello'"
You:
Thought: I will reverse the string using slice notation.
```python
text = "hello"
reversed_text = text[::-1]
final_answer(reversed_text)
```

User: "Filter the list [1, 2, 3, 4, 5] to keep only even numbers"
You:
Thought: I will use a list comprehension to filter for even numbers.
```python
numbers = [1, 2, 3, 4, 5]
even_numbers = [n for n in numbers if n % 2 == 0]
final_answer(even_numbers)
```

User: "What is today's date?"
You:
Thought: I will use the datetime library to get the current date.
```python
import datetime
today = datetime.date.today()
final_answer(today)
```

User: "Write 'Hello World' to a file named 'output.txt'"
You:
Thought: I will open the file in write mode and write the string.
```python
with open('output.txt', 'w') as f:
    f.write('Hello World')
final_answer("File written successfully")
```

User: "Calculate the 10th Fibonacci number"
You:
Thought: I will use an iterative approach to find the 10th Fibonacci number.
```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

result = fibonacci(10)
final_answer(result)
```

DO NOT write conversational filler. JUST THE THOUGHT AND CODE.
"""
            
            # Create the CodeAgent
            self.agent = CodeAgent(
                tools=[self.vision_tool], 
                model=self.model, 
                add_base_tools=True,
                max_steps=3, # Fail fast if it loops
                verbosity_level=logging.INFO,
                planning_interval=None,
                name=None,
                description=None,
                prompt_templates=None,
                instructions=SYSTEM_PROMPT_ADDITION,
                additional_authorized_imports=["os", "datetime", "math", "sympy"],
                stream_outputs=True # Enable real-time token streaming
            )
            
            logger.info("Real AI Agent initialized successfully (1.5B 4-bit Custom).")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Agent: {e}")
            raise e

    def update_vision_context(self, base64_image: str):
        """Updates the latest screen frame for the vision tool to use."""
        with self.image_lock:
            self.latest_image = base64_image

    def execute_stream(self, task: str):
        logger.info(f"Agent thinking on task: {task} (Streaming)")
        try:
            # Run the agent in streaming mode!
            # The agent.run(stream=True) yields steps.
            # We need to filter for text content to send to frontend.
            
            for step in self.agent.run(task, stream=True):
                # We mainly care about the delta content (thinking/coding process)
                # and the final answer.
                
                if isinstance(step, ChatMessageStreamDelta):
                    if step.content:
                        yield step.content
                
                elif isinstance(step, ToolCall):
                    yield f"\n> Executing tool: {step.name}...\n"
                    # yield f"> Arguments: {step.arguments}\n"
                
                elif isinstance(step, ToolOutput):
                    yield f"\nObservation:\n{step.observation}\n"

                elif isinstance(step, ActionStep):
                    if step.error:
                        yield f"\n[Step Error]: {step.error}\n"
                    # Also check if this action step IS the final answer (sometimes used instead of FinalAnswerStep)
                    if step.is_final_answer and step.action_output is not None:
                        yield f"\nFinal Answer: {step.action_output}\n"

                elif isinstance(step, FinalAnswerStep):
                    # Yield the final answer clearly!
                    yield f"\nFinal Answer: {step.output}\n"
            
        except GeneratorExit:
            logger.info("Client disconnected, stopping agent execution.")
            return # Explicitly stop the generator
        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            yield f"Error: {e}"
        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            yield f"Error: {e}"
