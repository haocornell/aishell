import os
import subprocess
import shlex
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, PathCompleter, WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from sentence_transformers import SentenceTransformer


import chromadb
from chromadb.config import Settings

# Point to a local directory to persist data
client = chromadb.PersistentClient(path="~/chroma_db")

# Create or load a collection
collection = client.get_or_create_collection("persistent_collection")

def store_commit_cmd(cmd):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = [cmd]
    ids = [str(hash(cmd))]
    embeddings = model.encode(docs).tolist()  # List of lists
    collection.add(
        documents=docs,
        ids=ids,
        embeddings=embeddings
    )


# query vector db
def query_cmd(query):
    results = collection.query(
        query_texts=[query],
        n_results=1,
        include=[
            "ids",        # optional: the Chroma-assigned IDs
            "documents",  # optional: your stored text/blobs
            "metadatas",  # optional: any metadata dicts
            "distances",  # optional: float distances for each hit
            "embeddings"  # <— ask Chroma to return the stored vectors!
     ]
    )
    return results

# 1) Gather executables in PATH for command completion
def list_executables():
    executables = set()
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if os.path.isdir(p):
            for fn in os.listdir(p):
                fp = os.path.join(p, fn)
                if os.access(fp, os.X_OK) and not os.path.isdir(fp):
                    executables.add(fn)
    return sorted(executables)

CMD_COMPLETER = WordCompleter(list_executables(), ignore_case=True)
PATH_COMPLETER = PathCompleter(expanduser=True)

# 2) Custom completer that switches on first token vs rest
class ShellCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        tokens = text.split()
        if not tokens or document.text_before_cursor.endswith(" "):
            # next token: path completion
            yield from PATH_COMPLETER.get_completions(document, complete_event)
        else:
            # first token or partial: command completion
            if len(tokens) == 1:
                for c in CMD_COMPLETER.words:
                    if c.startswith(tokens[0]):
                        yield Completion(c, start_position=-len(tokens[0]))
            else:
                yield from PATH_COMPLETER.get_completions(document, complete_event)

# 3) Built-in handlers
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

# 4) Invoke LLM for special prompt
def run_llm_prompt(prompt_text):
    # stub: replace with your LLM client call
    response = query_cmd(prompt_text)
    # e.g. response = openai.ChatCompletion.create(...)
    print(response)
    return

# 5) Main REPL
def main():
    session = PromptSession(completer=ShellCompleter())
    bindings = KeyBindings()

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
            if line.startswith(":"):
                run_llm_prompt(line[1:].strip())
                continue

            # split into tokens for built-ins
            tokens = shlex.split(line)
            if handle_builtin(tokens):
                continue

            # else pass through to real shell
            store_commit_cmd(line)  # Store command in vector DB
            subprocess.run(line, shell=True, executable="/bin/bash")

        except (EOFError, KeyboardInterrupt):
            break

if __name__ == "__main__":
    main()
