import os, sys
import subprocess
import shlex

import asyncio, threading
import queue

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, PathCompleter, WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.document import Document

from azure.identity import AzureCliCredential

from azure.identity.aio import (
    AzureDeveloperCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    )
from openai_messages_token_helper import build_messages, get_token_limit

class AsyncChromaDB:
    from chromadb.config import Settings

    def __init__(self, persist_directory="/home/mihao/chroma_db", collection_name="commands"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    async def init(self):
        def sync_init():
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_or_create_collection(self.collection_name)
            return client, collection
        self._client, self._collection = await asyncio.to_thread(sync_init)

    async def add(self, documents: list[str], ids, embeddings):
        def sync_add():
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")

            ids = [str(hash(doc)) for doc in documents]
            embeddings = model.encode(documents).tolist()  # List of lists

            self._collection.add(
                documents=documents,
                ids=ids,
                embeddings=embeddings
            )
        await asyncio.to_thread(sync_add)

    async def query(self, query_texts, n_results=1, include=None):
        def sync_query():
            return self._collection.query(
                query_texts=query_texts,
                n_results=n_results,
                include=[
                    "documents", 
                ]
            )
        return await asyncio.run(sync_query)
        #return await asyncio.to_thread(sync_query)
  
    async def run_search_llm_prompt(self, prompt_text):
        if not prompt_text.startswith(":"):
            await self.query([prompt_text], n_results=3)
        
        prompt_text = prompt_text[1:]

        async def get_openai_client():
            azure_credential = AzureDeveloperCliCredential(process_timeout=60)
            token_provider = get_bearer_token_provider(azure_credential, "https://cognitiveservices.azure.com/.default")

            AZURE_OPENAI_API_VERSION = '2024-08-01-preview'
            endpoint = 'https://mihao-m5wrou16-northcentralus.services.ai.azure.com'

            openai_client = AsyncAzureOpenAI(                                                                                                                       api_version=AZURE_OPENAI_API_VERSION,
                        azure_endpoint=endpoint,
                        azure_ad_token_provider=token_provider,
                    )
            return openai_client
        if not hasattr(main, "_openai_client"):
            main._openai_client = await get_openai_client()
                        
        openai_client = main._openai_client

        model = 'gpt-4o'
        query_messages = build_messages(
                    model=model,
                    system_prompt='You are a helpful assistant.',
                    new_user_content=prompt_text,
                )
        
        result = 'No response from LLM!'
        try :
            response: ChatCompletion = await openai_client.chat.completions.create(
                                messages=query_messages,  # type: ignore
                                # Azure OpenAI takes the deployment name as the model name
                                model= model,
                                temperature=0.9,
                                max_tokens=2468,  # Setting too low risks malformed JSON, setting too high may affect performance
                            )
            result = response.choices[0].message.content
        except Exception as e:
            print(f"Error in generating Kusto query: {e}", file=sys.stderr)

        return result


def list_executables():
    executables = set()
    """
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if os.path.isdir(p):
            for fn in os.listdir(p):
                fp = os.path.join(p, fn)
                if os.access(fp, os.X_OK) and not os.path.isdir(fp):
                    executables.add(fn)
    """
    return sorted(executables)

print('Initializing...')
CMD_COMPLETER = WordCompleter(list_executables(), ignore_case=True)

PATH_COMPLETER = PathCompleter(expanduser=True)

class ShellCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        tokens = text.split()
        if not tokens:
            # First token: command completion
            for c in CMD_COMPLETER.words:
                if c.startswith(text):
                    yield Completion(c, start_position=-len(text))
        elif len(tokens) == 1:
            # Still completing the command
            start = tokens[0]
            for c in CMD_COMPLETER.words:
                if c.startswith(start):
                    yield Completion(c, start_position=-len(start))
            # Also allow path completion if user starts typing . or /
            if start.startswith(('.', '/')):
                yield from PATH_COMPLETER.get_completions(document, complete_event)
        else:
            # Completing argument: path completion
            # Find the last token (argument being completed)
            arg = tokens[-1]
            # Create a new Document for the argument
            arg_doc = Document(text=arg, cursor_position=len(arg))
            yield from PATH_COMPLETER.get_completions(arg_doc, complete_event)

# Built-in handlers
def handle_builtin(tokens):
    cmd = tokens[0]
    if cmd == "cd":
        target = tokens[1] if len(tokens) > 1 else os.environ.get("HOME", "/")
        try:
            os.chdir(os.path.expanduser(target))
        except Exception as e:
            print(f"cd: {e}")
        return True
    if cmd == "export":
        for part in tokens[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                os.environ[k] = v
            else:
                print(f"export: malformed: {part}")
        return True
    return False

print('get auto completer init done...')

# Main REPL
def main():
    session = PromptSession(completer=ShellCompleter())
    print('started session up...')

    bindings = KeyBindings()
    print('done with key bindings...')
    

    @bindings.add("c-c")
    def _(event):
        event.app.exit()  # Ctrl-C quits

    while True:
        try:
            cwd = os.getcwd()
            line = session.prompt(f"(aishell) {cwd} $ ", key_bindings=bindings)
            if not line.strip():
                continue
            if line.strip() in ("exit", "quit"):
                break

            # LLM invocation syntax: e.g. starts with ":"
            if line.startswith(":") or line.startswith("!"):
                # Run the async LLM prompt and print results
                async def run_llm():
                    results = await main._chroma_db.run_search_llm_prompt(line[1:].strip())
                    print(results)
                asyncio.run(run_llm())
                continue

            # split into tokens for built-ins
            tokens = shlex.split(line)
            if handle_builtin(tokens):
                continue
            
            result = subprocess.run(line, shell=True, executable="/bin/bash")
            if result.returncode == 0:            
                # Run the vector DB storage in a separate thread running async
                def store_cmd_thread(cmd):
                    async def store_cmd_async():
                        # Initialize AsyncChromaDB if not already done
                        if not hasattr(main, "_chroma_db"):
                            main._chroma_db = AsyncChromaDB()
                            await main._chroma_db.init()
                        # Add the command to the vector DB
                        await main._chroma_db.add([cmd], None, None)
                    asyncio.run(store_cmd_async())

                # Use a single background thread and a queue to reuse the thread

                if not hasattr(main, "_cmd_queue"):
                    main._cmd_queue = queue.Queue()

                    def worker():
                        while True:
                            cmd = main._cmd_queue.get()
                            if cmd is None:
                                break
                            async def store_cmd_async():
                                if not hasattr(main, "_chroma_db"):
                                    main._chroma_db = AsyncChromaDB()
                                    await main._chroma_db.init()
                                await main._chroma_db.add([cmd], None, None)
                            asyncio.run(store_cmd_async())
                            main._cmd_queue.task_done()

                    main._worker_thread = threading.Thread(target=worker, daemon=True)
                    main._worker_thread.start()

                main._cmd_queue.put(line)

        except (EOFError, KeyboardInterrupt):
            break

if __name__ == "__main__":
    main()
