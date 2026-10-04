# Agent examples

Each example on this page is a complete [custom agent](launch-agents.md) you
can copy: a chat page in your browser, a notebook server, and Claude Code
working on tasks in the background. [Your first agent](launch-agents.md#your-first-agent)
shows the simplest one.

- [A chat app in the browser](#a-chat-app-in-the-browser)
- [A notebook server](#a-notebook-server)
- [Coding agents in the background](#coding-agents-in-the-background)

## A chat app in the browser

These steps run a Streamlit chat page in the container, which you open in
your Mac browser:

1. Make a project and add the libraries:

   ```sh
   cd ~/src
   uv init --package --python 3.13 chat-desk
   cd chat-desk
   uv add streamlit openai
   ```

2. Write `app.py` in the project folder:

   ```python
   import os

   import streamlit as st
   from openai import OpenAI

   client = OpenAI()
   model = os.environ["GMLX_MODEL"]

   st.title("Chat desk")
   if "messages" not in st.session_state:
       st.session_state.messages = []
   for message in st.session_state.messages:
       st.chat_message(message["role"]).write(message["content"])
   if prompt := st.chat_input("Ask the local model"):
       st.session_state.messages.append({"role": "user", "content": prompt})
       st.chat_message("user").write(prompt)
       stream = client.chat.completions.create(
           model=model, messages=st.session_state.messages, stream=True)
       reply = st.chat_message("assistant").write_stream(stream)
       st.session_state.messages.append({"role": "assistant", "content": reply})
   ```

3. Add the agent to your gmlx config file:

   ```yaml
   launch:
     agents:
       chat-desk:
         runtime: python
         command: [streamlit, run, app.py, --server.address, "127.0.0.1",
                   --server.port, "8501", --server.headless, "true"]
         web_port: 8501
   ```

4. Launch it from the project folder:

   ```sh
   gmlx launch chat-desk --model qwen3.8-27b-ud-q6@instruct --detach
   ```

`launch` opens the page once Streamlit answers. End the session with
`gmlx launch chat-desk --stop`. `--server.headless true` stops Streamlit
from asking for an email address on its first start. Use an `@instruct`
model, because the page shows nothing while a thinking model thinks.

## A notebook server

These steps run JupyterLab in the container, with notebooks that call the
model:

1. Make a project with JupyterLab and the OpenAI library:

   ```sh
   cd ~/src
   uv init --package --python 3.13 lab
   cd lab
   uv add jupyterlab openai
   ```

2. Make a token for the page with `openssl rand -hex 16`, and add the agent
   to your gmlx config file with it:

   ```yaml
   launch:
     agents:
       lab:
         runtime: python
         command: [jupyter, lab, --ip, "127.0.0.1", --port, "8888",
                   --no-browser, --allow-root]
         web_port: 8888
         env: [JUPYTER_TOKEN=<your token>]
   ```

3. Launch it from the project folder with `gmlx launch lab --detach`, open
   the address that `launch` prints, and sign in with the token.

4. In a notebook, call the model:

   ```python
   import os

   from openai import OpenAI

   reply = OpenAI().chat.completions.create(
       model=os.environ["GMLX_MODEL"],
       messages=[{"role": "user", "content": "Name three uses of a local model."}])
   print(reply.choices[0].message.content)
   ```

5. End the session with `gmlx launch lab --stop` from the project folder.

Notebooks are saved in the project folder. Keep the token secret: any
program on the Mac can open the page, and a notebook runs any code in the
container. Jupyter needs `--allow-root` because the container runs as root.

## Coding agents in the background

These steps give Claude Code one task at a time in the background, each in
its own git worktree. You read and merge the branches it commits.

1. Make the folder `~/containers/claude-bg` with a one-line Containerfile:

   ```dockerfile
   FROM gmlx.invalid/launch-claude-code:base
   ```

2. Add the agent to your gmlx config file:

   ```yaml
   launch:
     agents:
       fixer:
         build: ~/containers/claude-bg
         api: anthropic
         command: [claude, -p, --dangerously-skip-permissions]
         env: [IS_SANDBOX=1, DISABLE_AUTOUPDATER=1]
   ```

3. Make a worktree for the task, and start the agent in it with the task
   after `--`:

   ```sh
   cd ~/src/app
   git worktree add ../app-mul -b add-mul
   cd ../app-mul
   gmlx launch fixer --detach -- "Add a mul function to calc.py, then commit it."
   ```

4. Follow the sessions with `gmlx launch --list`, which names each one's
   output file.

5. When the session has ended, read the branch, merge it, and clean up:

   ```sh
   cd ~/src/app
   git diff main add-mul
   git merge add-mul
   (cd ../app-mul && gmlx launch fixer --remove-home)
   git worktree remove ../app-mul
   ```

`-p` runs the task after `--`, and the session ends when Claude Code
finishes. `launch` sets `IS_SANDBOX=1` and `DISABLE_AUTOUPDATER=1` only for
the built-in `claude-code` client, so the agent sets both in `env`. Each
worktree is a separate project, so several tasks can run at once. The agent commits with your git name, because the session shares the
repository's [git folder](container-access.md#git-in-the-container).

Run `--remove-home` before you remove the worktree, because `launch` finds
the private home by the folder. Read each branch before you merge it, since
it can change
[files that the Mac runs](container-security.md#shares-that-lead-back-to-the-mac).
For a repository you do not trust, follow
[A session for code you do not trust](container-security.md#a-session-for-code-you-do-not-trust).
