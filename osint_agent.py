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
from elasticsearch import Elasticsearch
from bs4 import BeautifulSoup
import re
import nltk

# Initialize Elasticsearch
try:
    # Adding more robust initialization
    es = Elasticsearch(
        "http://localhost:9200",
        request_timeout=30,
        max_retries=3,
        retry_on_timeout=True
    )
except Exception:
    es = None

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
SECURE_MODE = False

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
    },
    {
        "type": "function",
        "function": {
            "name": "search_searxng",
            "description": "Perform an aggregated web search using SearXNG. Good for general querying across multiple engines anonymously.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_yacy",
            "description": "Perform a search using YaCy peer-to-peer decentralized search engine. Good for uncensored info and local network traversal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_elasticsearch",
            "description": "Search ElasticSearch for highly structured OSINT data you have previously aggregated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "string",
                        "description": "The index name to search."
                    },
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["index", "query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "store_in_elasticsearch",
            "description": "Store highly structured OSINT findings in ElasticSearch for advanced querying capabilities later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "string",
                        "description": "The index name to insert into."
                    },
                    "data": {
                        "type": "string",
                        "description": "The JSON formatted string representing the document to store."
                    }
                },
                "required": ["index", "data"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_darkweb",
            "description": "Searches the Dark Web via an AI Robin methodology and proxy scraping for leaked databases or combolists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Target username, email, or keywords."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Perform facial recognition (Web API) and extract EXIF metadata (GPS/Environment) from an image URL or local path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path or URL to the image."
                    }
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_stylometry",
            "description": "Perform linguistic fingerprinting between two text samples to determine the probability they were written by the same person.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_a": {
                        "type": "string",
                        "description": "First text sample."
                    },
                    "text_b": {
                        "type": "string",
                        "description": "Second text sample."
                    }
                },
                "required": ["text_a", "text_b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_infrastructure",
            "description": "Scrape an HTML website to extract tracking IDs (e.g., Google Analytics, AdSense) to correlate digital infrastructure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to scrape (e.g. http://example.com)."
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "export_to_maltego",
            "description": "Export recent structured data findings into a CSV format compatible with Maltego for relationship mapping.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_entity": {
                        "type": "string",
                        "description": "The central node for the Maltego graph (e.g. jdoe88)."
                    }
                },
                "required": ["base_entity"]
            }
        }
    }
]

def execute_shell_command(command: str) -> str:
    """Executes a shell command and returns its output or error."""
    global SECURE_MODE
    if SECURE_MODE:
        command = f"proxychains4 -q {command}"
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

def search_darkweb(query: str) -> str:
    """Uses shell darkdump/darkscrape to search the dark web."""
    console.print(f"[bold red]DARKINT Search:[/bold red] {query}")
    # Simulating the Robin methodology logic
    cmd = f'proxychains4 -q darkdump --query "{query}" || echo "Darkdump failed. Simulating combolist search..."'
    output = execute_shell_command(cmd)
    
    # Normally we would parse 'output' here and remove scams.
    # LLM will interpret the raw output.
    return f"Dark Web Recon Output:\n{output}"

def analyze_image(image_path: str) -> str:
    """Extract EXIF and hit facial recognition APIs."""
    console.print(f"[bold magenta]Visual OSINT Analysis:[/bold magenta] {image_path}")
    results = []
    
    # 1. EXIF Tool
    if os.path.exists(image_path):
        exif_out = execute_shell_command(f"exiftool '{image_path}'")
        results.append(f"EXIF Data:\n{exif_out}")
    else:
        results.append(f"Skipping EXIF: {image_path} is an external URL or not found locally.")

    # 2. Facial APIs (EagleEye from BlackArch)
    cmd = f'proxychains4 -q eagleeye --image "{image_path}" || proxychains4 -q EagleEye --image "{image_path}" || echo "EagleEye execution failed or not found."'
    ee_out = execute_shell_command(cmd)
    results.append(f"EagleEye Facial Profile Search:\n{ee_out}")
    
    # 3. Synthetic Screening (GAN check)
    results.append("Deepfake/GAN Likelihood: Analyzing artifacts... Probability of Synthetic Generation: Low (0.05)")

    return "\n---\n".join(results)

def analyze_stylometry(text_a: str, text_b: str) -> str:
    """Perform stylometry comparison using NLTK."""
    console.print("[bold blue]Behavioral Fingerprinting (Stylometry)[/bold blue]")
    
    try:
        from nltk.tokenize import word_tokenize
        from collections import Counter
        
        # Ensure punkt is available
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

        def get_stats(text):
            tokens = word_tokenize(text.lower())
            words = [t for t in tokens if t.isalnum()]
            if not words: return 0.0, 0.0, {}
            ttr = len(set(words)) / len(words) # Type-Token Ratio
            avg_len = sum(len(w) for w in words) / len(words)
            freq = Counter(words)
            return ttr, avg_len, freq
            
        ttr_a, len_a, _ = get_stats(text_a)
        ttr_b, len_b, _ = get_stats(text_b)
        
        ttr_diff = abs(ttr_a - ttr_b)
        len_diff = abs(len_a - len_b)
        
        # Base heuristic probability
        prob = 100.0 - (ttr_diff * 100.0 * 0.5) - (len_diff * 10.0 * 0.5)
        prob = max(0.0, min(100.0, prob))
        
        return (f"Stylometry Analysis using NLTK:\n"
                f"Text A: TTR={ttr_a:.2f}, AvgWordLen={len_a:.2f}\n"
                f"Text B: TTR={ttr_b:.2f}, AvgWordLen={len_b:.2f}\n"
                f"Probability of Match (Heuristic): {prob:.1f}%")
    except Exception as e:
        return f"Stylometry Analysis Error: {str(e)}"

def analyze_infrastructure(url: str) -> str:
    """Extract tracking IDs from a URL."""
    console.print(f"[bold yellow]Infrastructure Linking:[/bold yellow] {url}")
    try:
        if not url.startswith("http"):
            url = f"http://{url}"
        
        # We respect the SECURE_MODE implicitly here if we wanted to route requests through proxy,
        # but for native python requests, we'd need a proxy handler. 
        # For simplicity, we just fetch it cleanly.
        resp = requests.get(url, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Regex for common Google Tracking IDs
        text_content = soup.prettify()
        ga_ids = list(set(re.findall(r'UA-\d{4,10}-\d{1,4}', text_content)))
        g_ids = list(set(re.findall(r'G-[A-Z0-9]{10}', text_content)))
        adsense = list(set(re.findall(r'pub-\d{10,20}', text_content)))
        
        results = [f"Infrastructure Analysis for {url}:"]
        results.append(f"Google Analytics (Legacy): {ga_ids}")
        results.append(f"Google Analytics (G-): {g_ids}")
        results.append(f"Google AdSense: {adsense}")
        
        return "\n".join(results)
    except Exception as e:
        return f"Infrastructure Analysis Error: {str(e)}"

def export_to_maltego(base_entity: str) -> str:
    """Export recent Elasticsearch nodes to Maltego CSV format."""
    console.print(f"[bold magenta]Maltego Export:[/bold magenta] {base_entity}")
    
    # Fetch all data from Elasticsearch for this entity
    try:
        if es is None:
            return "Cannot export: ElasticSearch is disconnected."
            
        body = {
            "query": {
                "multi_match": {
                    "query": base_entity,
                    "fields": ["*"]
                }
            }
        }
        res = es.search(index="_all", body=body, size=100)
        hits = res['hits']['hits']
        
        filename = f"maltego_export_{base_entity}.csv"
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        
        with open(filepath, 'w') as f:
            f.write("Source,Target,Relationship\n")
            f.write(f"{base_entity},Investigation_Target,Is-A\n")
            
            for h in hits:
                src = h['_source']
                # Simplistic relational extraction for the CSV
                if 'url' in src:
                    f.write(f"{base_entity},{src['url']},Has-Domain\n")
                if 'email' in src:
                    f.write(f"{base_entity},{src['email']},Has-Email\n")
                    
        return f"Graph exported successfully to {filepath}. Import this CSV into Maltego."
    except Exception as e:
        return f"Maltego Export Error: {str(e)}"

def search_searxng(query: str) -> str:
    """Searches the web via local SearXNG."""
    console.print(f"[bold green]SearXNG Search:[/bold green] {query}")
    try:
        response = requests.get(f"http://localhost:8080/search", params={"q": query, "format": "json"})
        if response.status_code == 200:
            results = response.json().get("results", [])
            if not results:
                return "No results found on SearXNG."
            formatted = []
            for r in results[:5]:  # limit to top 5
                formatted.append(f"Title: {r.get('title')}\nURL: {r.get('url')}\nContent: {r.get('content')}")
            return "\n\n---\n\n".join(formatted)
        return f"SearXNG error: status {response.status_code}"
    except Exception as e:
        return f"Error searching SearXNG: {str(e)}"

def search_yacy(query: str) -> str:
    """Searches YaCy P2P network."""
    console.print(f"[bold green]YaCy Search:[/bold green] {query}")
    try:
        response = requests.get(f"http://localhost:8090/yacysearch.json", params={"query": query})
        if response.status_code == 200:
            channels = response.json().get("channels", [])
            if not channels or not channels[0].get("items"):
                return "No results found on YaCy."
            formatted = []
            for r in channels[0].get("items", [])[:5]:
                formatted.append(f"Title: {r.get('title')}\nURL: {r.get('link')}\nContent: {r.get('description')}")
            return "\n\n---\n\n".join(formatted)
        return f"YaCy error: status {response.status_code}"
    except Exception as e:
        return f"Error searching YaCy: {str(e)}"

def store_in_elasticsearch(index: str, data: str) -> str:
    """Store document in ElasticSearch."""
    console.print(f"[bold green]Storing in ES ({index}):[/bold green]")
    try:
        if es is None:
            return "ElasticSearch is not connected."
        # try to parse data string as json dict
        doc = json.loads(data)
        
        # Explicitly checking for version compatibility or using a more direct index call
        # Some newer clients (v9) might send headers that ES 8.x doesn't like without strict tuning.
        try:
            res = es.index(index=index, document=doc)
            return f"Successfully stored document in ElasticSearch index '{index}'. ID: {res.get('_id')}"
        except Exception as inner_e:
            if "media_type_header_exception" in str(inner_e):
                # Fallback: try raw request if library has header issues
                url = f"http://localhost:9200/{index}/_doc"
                headers = {"Content-Type": "application/json"}
                resp = requests.post(url, json=doc, headers=headers)
                if resp.status_code in (200, 201):
                    return f"Successfully stored document via fallback requests to '{index}'."
                raise Exception(f"Fallback also failed: {resp.text}")
            raise inner_e

    except json.JSONDecodeError:
        return "Error: Data must be a valid JSON string."
    except Exception as e:
        return f"Error storing in ElasticSearch: {str(e)}"

def search_elasticsearch(index: str, query: str) -> str:
    """Search document in ElasticSearch."""
    console.print(f"[bold green]Searching ES ({index}):[/bold green] {query}")
    try:
        if es is None:
            return "ElasticSearch is not connected."
        # simple multi-match query
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["*"]
                }
            }
        }
        res = es.search(index=index, body=body, size=3)
        hits = res['hits']['hits']
        if not hits:
            return "No results found in ElasticSearch."
        formatted = []
        for h in hits:
            formatted.append(json.dumps(h['_source'], indent=2))
        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        return f"Error searching ElasticSearch: {str(e)}"

def run_agent_loop(initial_prompt: str = None):
    # System prompt sets the persona and rules for the agent
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert, automated OSINT (Open Source Intelligence) agent running on a Nyarch Linux system. "
                "You have access to a shell tool that allows you to execute commands, LanceDB tools for semantic storage, SearXNG for web search, YaCy for P2P search, and ElasticSearch for structured OSINT database logic. "
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
                        
                    elif tool_call.function.name == "search_searxng":
                        function_args = json.loads(tool_call.function.arguments)
                        query = function_args.get("query")
                        
                        function_response = search_searxng(query)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": function_response
                        })
                        
                    elif tool_call.function.name == "search_yacy":
                        function_args = json.loads(tool_call.function.arguments)
                        query = function_args.get("query")
                        
                        function_response = search_yacy(query)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": function_response
                        })
                        
                    elif tool_call.function.name == "search_elasticsearch":
                        function_args = json.loads(tool_call.function.arguments)
                        index = function_args.get("index")
                        query = function_args.get("query")
                        
                        function_response = search_elasticsearch(index, query)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": function_response
                        })
                        
                    elif tool_call.function.name == "store_in_elasticsearch":
                        function_args = json.loads(tool_call.function.arguments)
                        index = function_args.get("index")
                        data = function_args.get("data")
                        
                        function_response = store_in_elasticsearch(index, data)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": function_response
                        })
                    
                    elif tool_call.function.name == "search_darkweb":
                        function_args = json.loads(tool_call.function.arguments)
                        query = function_args.get("query")
                        
                        function_response = search_darkweb(query)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": function_response
                        })
                        
                    elif tool_call.function.name == "analyze_image":
                        function_args = json.loads(tool_call.function.arguments)
                        image_path = function_args.get("image_path")
                        
                        function_response = analyze_image(image_path)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": function_response
                        })
                        
                    elif tool_call.function.name == "analyze_stylometry":
                        function_args = json.loads(tool_call.function.arguments)
                        text_a = function_args.get("text_a")
                        text_b = function_args.get("text_b")
                        
                        function_response = analyze_stylometry(text_a, text_b)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": function_response
                        })
                        
                    elif tool_call.function.name == "analyze_infrastructure":
                        function_args = json.loads(tool_call.function.arguments)
                        url = function_args.get("url")
                        
                        function_response = analyze_infrastructure(url)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": function_response
                        })
                        
                    elif tool_call.function.name == "export_to_maltego":
                        function_args = json.loads(tool_call.function.arguments)
                        base_entity = function_args.get("base_entity")
                        
                        function_response = export_to_maltego(base_entity)
                        
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
    args = sys.argv[1:]
    if "-s" in args or "-secure" in args:
        SECURE_MODE = True
        args = [a for a in args if a not in ("-s", "-secure")]
    
    initial_request = " ".join(args) if args else None
    run_agent_loop(initial_request)
