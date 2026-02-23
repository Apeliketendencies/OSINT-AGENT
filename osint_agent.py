import os
import sys
import subprocess
import json
import requests
import lancedb
import pyarrow as pa
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown

# Initialize LanceDB connection and ensure table exists
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lancedb_data")
db = lancedb.connect(DB_PATH)

def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text:latest", "prompt": text}
    )
    return response.json().get("embedding", [])

try:
    table = db.open_table("osint_knowledge")
except:
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 768)),
        pa.field("content", pa.string()),
        pa.field("source", pa.string())
    ])
    table = db.create_table("osint_knowledge", schema=schema)

# Initialize rich console
console = Console()

# Configure OpenAI client to point to local Ollama instance
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'  # API key is not required for local Ollama, but the library needs a string
)

# Set the model to the requested gpt-oss model
MODEL = "gpt-oss:20b"

# Define the shell execution tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_shell_command",
            "description": "Execute a shell command on the local Nyarch Linux system. This gives you access to OSINT tools installed on the system (like social and recon packages). Always run commands non-interactively.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash shell command to execute."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "store_in_database",
            "description": "Store gathered information in a vector database for later reference or memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to store."
                    },
                    "source": {
                        "type": "string",
                        "description": "Where this information came from."
                    }
                },
                "required": ["content", "source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search the vector database for previously stored information using semantic similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The semantic search query."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def execute_shell_command(command: str) -> str:
    """Executes a shell command and returns its output or error."""
    console.print(f"[bold yellow]Executing command:[/bold yellow] {command}")
    try:
        # Run the command with a timeout to prevent hanging
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes timeout
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]:\n{result.stderr}"
        
        # Truncate very long outputs to avoid context window explosion
        if len(output) > 8000:
            console.print("[dim]Command output truncated before sending to model.[/dim]")
            output = output[:8000] + "\n...[OUTPUT TRUNCATED]..."
            
        return output if output.strip() else "Command executed successfully with no output."
    except subprocess.TimeoutExpired:
        return "Command timed out after 120 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

def store_in_database(content: str, source: str) -> str:
    """Stores information in LanceDB."""
    console.print(f"[bold green]Storing in DB:[/bold green] {source}")
    try:
        embedding = generate_embedding(content)
        if not embedding:
            return "Failed to generate embedding."
        table.add([{"vector": embedding, "content": content, "source": source}])
        return f"Successfully stored information from {source}."
    except Exception as e:
        return f"Error storing in database: {str(e)}"

def search_database(query: str) -> str:
    """Searches LanceDB for information."""
    console.print(f"[bold green]Searching DB for:[/bold green] {query}")
    try:
        embedding = generate_embedding(query)
        if not embedding:
            return "Failed to generate embedding."
        
        # Search the table
        results = table.search(embedding).limit(3).to_pandas()
        
        if results.empty:
            return "No relevant information found in the database."
            
        formatted_results = []
        for _, row in results.iterrows():
            formatted_results.append(f"Source: {row['source']}\nContent: {row['content']}\nDistance: {row['_distance']:.4f}")
            
        return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        return f"Error searching database: {str(e)}"

def run_agent_loop(initial_prompt: str = None):
    # System prompt sets the persona and rules for the agent
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert, automated OSINT (Open Source Intelligence) agent running on a Nyarch Linux system. "
                "You have access to a shell tool that allows you to execute commands, and LanceDB database tools to store and search information. "
                "The system has 'social' and 'recon' OSINT packages installed (e.g., tools like sherlock, holehe, theHarvester, nmap, etc.). "
                "When given a target or request, formulate a plan to gather information using the available CLI tools. "
                "Do not guess information; execute commands to find out. Store useful findings in the database so you don't forget them. "
                "You are interacting directly with the user in a chat interface. Explain your findings clearly."
            )
        }
    ]
    
    if initial_prompt:
        messages.append({"role": "user", "content": initial_prompt})
        console.print(f"[bold green]User:[/bold green] {initial_prompt}")
    
    console.print(f"[bold blue]OSINT Agent started. Using model: {MODEL}[/bold blue]")
    console.print("[dim]Type your message and press Enter. Type 'exit' or 'quit' to stop.[/dim]\n")

    while True:
        if not messages or messages[-1]["role"] not in ["user", "tool"]:
            try:
                user_input = console.input("[bold green]User:[/bold green] ")
                if user_input.strip().lower() in ['exit', 'quit']:
                    console.print("[bold blue]Exiting OSINT Agent...[/bold blue]")
                    break
                messages.append({"role": "user", "content": user_input})
            except (KeyboardInterrupt, EOFError):
                console.print("\n[bold blue]Exiting OSINT Agent...[/bold blue]")
                break

        try:
            # Query the model
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            
            # If the model wants to call a tool
            if response_message.tool_calls:
                # Add the assistant's request to the messages (required by API)
                messages.append(response_message)
                
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "execute_shell_command":
                        function_args = json.loads(tool_call.function.arguments)
                        command = function_args.get("command")
                        
                        # Execute the tool
                        function_response = execute_shell_command(command)
                        
                        # Add the tool result to the messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": function_response
                        })
                    
                    elif tool_call.function.name == "store_in_database":
                        function_args = json.loads(tool_call.function.arguments)
                        content = function_args.get("content")
                        source = function_args.get("source")
                        
                        function_response = store_in_database(content, source)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": function_response
                        })
                        
                    elif tool_call.function.name == "search_database":
                        function_args = json.loads(tool_call.function.arguments)
                        query = function_args.get("query")
                        
                        function_response = search_database(query)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": function_response
                        })
                # The loop will continue, passing the tool results back to the model
                continue
            
            # If the model replies with text
            elif response_message.content:
                console.print("\n[bold magenta]OSINT Agent:[/bold magenta]")
                console.print(Markdown(response_message.content))
                console.print() # Empty line for spacing
                messages.append({"role": "assistant", "content": response_message.content})
                
        except Exception as e:
            console.print(f"[bold red]Error communicating with Ollama:[/bold red] {str(e)}")
            console.print("[dim]Please ensure Ollama is running (`systemctl start ollama`) and the model is pulled (`ollama pull gpt-oss`).[/dim]")
            messages.pop() # Remove the last user message so they can try again

if __name__ == "__main__":
    initial_request = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    run_agent_loop(initial_request)
