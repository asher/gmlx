# Container recipes

This page shows complete setups for clients in
[container mode](launch-container.md). They extend pi and dsh with
packages, give a client tool servers and a headless browser, and run
services such as Postgres and web search beside the client. Each recipe
uses the methods of [Custom container images](container-images.md), and
each one needs container mode turned on for its client, with `--container`
or [`enabled`](config.md#launchcontainerenabled).

For programs of your own, see the recipes at the end of
[Custom agents](launch-agents.md).

- [pi packages](#pi-packages)
- [dsh plugins](#dsh-plugins)
- [Headless browsers](#headless-browsers)
- [Tool servers](#tool-servers)
- [Postgres](#postgres)
- [Web search for Open WebUI](#web-search-for-open-webui)

## pi packages

pi loads extensions, skills, prompt templates and themes from pi packages,
which are npm packages, git repositories or local folders. A package in the
image works in every project and needs no network while the session runs.
A package that `pi install` adds belongs to one project.

To put a package in the image:

1. Write a Containerfile that installs the package into a folder of its
   own:

   ```dockerfile
   FROM gmlx.invalid/launch-pi:base
   RUN npm install --prefix /opt/pi-packages --legacy-peer-deps pi-btw@0.7.1 \
    && npm cache clean --force
   ```

2. Name the folder with [`build`](config.md#launchcontainerclientsbuild),
   and load the package with pi's `-e` option in
   [`command`](config.md#launchcontainerclientscommand):

   ```yaml
   launch:
     container:
       clients:
         pi:
           build: ~/containers/pi
           command: [pi, -e, /opt/pi-packages/node_modules/pi-btw]
   ```

3. Launch pi from the project folder with `gmlx launch pi --container`.

The example installs [pi-btw](https://github.com/dbachelder/pi-btw), which
adds a `/btw` command for a side conversation. `--legacy-peer-deps` keeps
npm from adding a second copy of pi, since pi gives its packages its own
copy. pi loads the folder that `-e` names with its extensions, skills,
prompt templates and themes, and writes nothing to its settings.

Add one `-e` and its folder for each package, and write an exact version
for the reason that [A newer client](container-images.md#a-newer-client)
gives.

`pi install` from the shell puts the package in the private home and adds
it to the `packages` list in `~/.pi/agent/settings.json`. `launch` keeps that
list when it writes its own settings into the file. The install needs the
network, and each project gets its own copy:

```sh
cd ~/src/my-project
gmlx launch pi --shell -- -c "pi install npm:pi-btw@0.7.1"
```

When a package in its settings is missing, pi installs it, and it checks
a package without a version for updates. Both steps fail under
[`network: none`](config.md#launchcontainernetwork), so install with the
default network first and give each package a version.
`PI_OFFLINE=1` in [`env`](config.md#launchcontainerenv) stops pi from
trying.

To use the extensions, skills and prompt templates that you keep on the
Mac, copy their folders with
[`seed`](config.md#launchcontainerclientsseed):

```yaml
launch:
  container:
    clients:
      pi:
        seed: [~/.pi/agent/extensions, ~/.pi/agent/skills, ~/.pi/agent/prompts]
```

Seed these folders, not all of `~/.pi/agent`. That folder also holds
`auth.json` with the keys of your providers, and `launch` warns when a seed
copies it.

## dsh plugins

dsh installs plugins with pnpm into a profile. The profile that launch
starts is `gmlx`, at `~/.dsh/profiles/gmlx` in the private home, so each
project has its own plugins. The image that gmlx builds has no pnpm, so add
it first:

1. Write a Containerfile that installs pnpm:

   ```dockerfile
   FROM gmlx.invalid/launch-dsh:base
   RUN npm install -g pnpm@11.28.2 && npm cache clean --force
   ```

2. Name its folder with [`build`](config.md#launchcontainerclientsbuild):

   ```yaml
   launch:
     container:
       clients:
         dsh:
           build: ~/containers/dsh
   ```

3. Launch dsh once in the project with `gmlx launch dsh --container`.

4. Install the plugin from the shell, with the default network:

   ```sh
   cd ~/src/my-project
   gmlx launch dsh --shell -- -c "dsh plugin --profile gmlx add dsh-context@0.62.3"
   ```

The example installs
[dsh-context](https://github.com/bowenliang123/dsh-context), which adds a
context dashboard to the web app. dsh loads the plugin at its next start,
and the Plugins page of the web app installs plugins the same way, with
pnpm in the image too. The launch in step 3 makes the `gmlx` profile from
the `web` template, while a `dsh plugin` command before that launch makes a
profile without the web app.

The plugins, the pnpm store and its cache stay in the private home, so a
plugin keeps working under
[`network: none`](config.md#launchcontainernetwork) after the install. pnpm
blocks the build step of a plugin from a git repository until you allow it
in `~/.dsh/profiles/gmlx/pnpm-workspace.yaml`, as the message from
`dsh plugin` says. dsh also reads skills from `~/.dsh/skills` and
`~/.agents/skills`, which [`seed`](config.md#launchcontainerclientsseed)
can fill from the Mac.

## Headless browsers

For screenshots of plain pages, add Chromium and fonts to the image:

```yaml
launch:
  container:
    clients:
      claude-code:
        packages: [chromium, fonts-liberation, fonts-noto-color-emoji]
```

The client then takes a screenshot with:

```sh
chromium --headless --no-sandbox --disable-dev-shm-usage \
  --screenshot=/tmp/page.png https://example.com
```

Each part of that command has a reason:

- The client runs as root, and Chromium refuses to start as root without
  `--no-sandbox`.
- The container has no display, so Chromium runs headless only.
- `/dev/shm` is small in the container, and `--disable-dev-shm-usage`
  keeps Chromium from running out of it.
- A slim image has almost no fonts, so pages render text as empty boxes
  without `fonts-liberation` and `fonts-noto-color-emoji`.

Chromium prints D-Bus errors in a container, because no D-Bus service
runs there. They do not affect the screenshot.

Puppeteer downloads its own Chrome by default, which may have no Linux
arm64 build. To use the installed Chromium, add
`PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium` to
[`env`](config.md#launchcontainerenv), or set it with `ENV` in your own
Containerfile.

## Tool servers

An [MCP](glossary.md#mcp) tool server that runs in the container gives the
client tools that reach only what the session reaches. The tools of a
served assistant run on the Mac instead, as
[What the client reaches on the server](container-security.md#what-the-client-reaches-on-the-server)
explains, so give a coding agent its tools this way.

A tool server and the entry that names it can live in one of three places:

- The private home. The client's own command adds the entry from the
  shell, and it stays for that project, as with `claude mcp add --scope user`,
  `pi mcp add` or `hermes mcp add`. The server's program must be in the
  image, or the network must let `npx` or `uvx` download it when the client
  starts it.
- The image. Neither the client nor the Mac can change the server or its
  entry there, and the server starts with no download, also under
  [`network: none`](config.md#launchcontainernetwork). The example below
  works this way.
- A project file in the share, such as `.mcp.json` or `opencode.json`. The
  client can change it, and the clients on the Mac read it, as
  [Shares that lead back to the Mac](container-security.md#shares-that-lead-back-to-the-mac)
  explains. Use it only for servers that you run on the Mac too.

For example, this gives pi a memory server that keeps what pi stores in a
file of the private home, so the memory belongs to the project. The single
quotes keep `$HOME` for the shell in the container:

```sh
cd ~/src/my-project
gmlx launch pi --shell -- -c 'pi mcp add memory \
  --env MEMORY_FILE_PATH=$HOME/.pi/memory.jsonl \
  -- npx -y @modelcontextprotocol/server-memory@2026.8.31'
```

To give Claude Code a browser that it drives through the Playwright tool
server, with both in the image:

1. Write a Containerfile that adds Chromium, its fonts and the server, and
   copies the entry into the image:

   ```dockerfile
   FROM gmlx.invalid/launch-claude-code:base
   RUN apt-get update \
    && apt-get install -y --no-install-recommends chromium fonts-liberation fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*
   RUN npm install -g @playwright/mcp@0.0.83 && npm cache clean --force
   COPY mcp.json /etc/mcp/claude.json
   ```

2. Beside it, write `mcp.json`:

   ```json
   {
     "mcpServers": {
       "playwright": {
         "type": "stdio",
         "command": "playwright-mcp",
         "args": ["--browser", "chromium", "--executable-path", "/usr/bin/chromium",
                  "--headless", "--no-sandbox", "--isolated",
                  "--output-dir", "/tmp/playwright-mcp"]
       }
     }
   }
   ```

3. Name the folder, and load the entry with Claude Code's `--mcp-config`
   option:

   ```yaml
   launch:
     container:
       clients:
         claude-code:
           build: ~/containers/claude-playwright
           command: [claude, --mcp-config, /etc/mcp/claude.json]
   ```

4. Launch Claude Code from the project folder with
   `gmlx launch claude-code --container`. Its `/mcp` command lists
   `playwright`.

Playwright starts Chrome by default, which has no build for Linux on arm64,
so the entry names the Chromium of Debian. `--no-sandbox` has the reason
that [Headless browsers](#headless-browsers) gives. `--isolated` keeps the
browser profile in memory.

`--output-dir` takes the snapshots and the screenshots that the server
names itself, and a screenshot that the model names goes to the working
folder, which is the shared project folder. Claude Code loads a file that
`--mcp-config` names without asking, and the file stays out of the share.

For opencode, start the Containerfile from
`gmlx.invalid/launch-opencode:base`, leave out the `COPY` line, and give the
entry in [`env`](config.md#launchcontainerenv):

```yaml
launch:
  container:
    clients:
      opencode:
        build: ~/containers/opencode-playwright
        env:
          - 'OPENCODE_CONFIG_CONTENT={"mcp":{"playwright":{"type":"local","command":["playwright-mcp","--browser","chromium","--executable-path","/usr/bin/chromium","--headless","--no-sandbox","--isolated","--output-dir","/tmp/playwright-mcp"]}}}'
```

## Postgres

Postgres can run in the container with its data on a
[volume](launch-container.md#volumes), or on the Mac with a
[forwarded port](launch-container.md#forwarded-ports). A share does not
work for its data. Postgres refuses a data folder that it does not own, and
every file of a [share](launch-container.md#shares) belongs to root in the
container.

To run it in the container:

1. Write a Containerfile that installs Postgres and copies a start script:

   ```dockerfile
   FROM gmlx.invalid/launch-claude-code:base
   RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql \
    && rm -rf /var/lib/apt/lists/*
   COPY start-pg /usr/local/bin/start-pg
   RUN chmod 755 /usr/local/bin/start-pg
   ```

2. Beside it, write the start script `start-pg`. It starts Postgres under
   its own user, then runs the client:

   ```sh
   #!/bin/sh
   set -e
   data=/var/lib/postgresql/data
   bin=$(echo /usr/lib/postgresql/*/bin)
   mkdir -p "$data" /run/postgresql
   chown postgres:postgres "$data" /run/postgresql
   # A session never stops Postgres cleanly, so the pid file is stale.
   rm -f "$data/postmaster.pid"
   if [ ! -s "$data/PG_VERSION" ]; then
     runuser -u postgres -- "$bin/initdb" -D "$data"
   fi
   runuser -u postgres -- "$bin/pg_ctl" -D "$data" -l /tmp/postgres.log start
   exec "$@"
   ```

3. Name the folder, the start script and a volume:

   ```yaml
   launch:
     container:
       clients:
         claude-code:
           build: ~/containers/claude-pg
           command: [/usr/local/bin/start-pg, claude]
           volumes: [claude-pg:/var/lib/postgresql:8G]
   ```

4. Launch Claude Code from the project folder with
   `gmlx launch claude-code --container`.

The data lives in a subfolder of the volume, as
[Volumes](launch-container.md#volumes) advises. It survives from one
session to the next, and the client connects with `psql -U postgres`.

To use Postgres on the Mac instead, add `forward: [5432]`, after you give
it a password or a limited role as
[Forwarded ports](launch-container.md#forwarded-ports) says.

## Web search for Open WebUI

Open WebUI can search the web through SearXNG, a metasearch engine that
runs in the same container. The model then searches the web without an
account at a search provider.

1. In a folder such as `~/containers/open-webui-searxng`, write a
   Containerfile that installs SearXNG in a virtual environment of its own:

   ```dockerfile
   FROM gmlx.invalid/launch-open-webui:base
   ARG SEARXNG_COMMIT=19ffbcd30686e4008392e93de164200510dfb9d8
   RUN set -eux; \
       mkdir -p /opt/searxng/src; \
       cd /opt/searxng/src; \
       git init -q; \
       git fetch -q --depth 1 https://github.com/searxng/searxng.git "$SEARXNG_COMMIT"; \
       git -c advice.detachedHead=false checkout -q FETCH_HEAD; \
       /usr/bin/python3 -m venv /opt/searxng/venv; \
       /opt/searxng/venv/bin/pip install --no-cache-dir \
           -r requirements.txt -r requirements-server.txt
   COPY settings.yml /etc/searxng/settings.yml
   COPY start-webui /usr/local/bin/start-webui
   RUN chmod 755 /usr/local/bin/start-webui
   ```

2. Beside it, write the SearXNG settings `settings.yml`:

   ```yaml
   use_default_settings: true

   search:
     formats:
       - html
       - json

   server:
     limiter: false
     public_instance: false
     image_proxy: false
   ```

3. Beside them, write the start script `start-webui`:

   ```sh
   #!/bin/sh
   set -e
   secret=$(/opt/searxng/venv/bin/python -c 'import secrets; print(secrets.token_hex(32))')
   (
     cd /opt/searxng/src
     SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml SEARXNG_SECRET="$secret" \
       exec /opt/searxng/venv/bin/granian --interface wsgi \
         --host 127.0.0.1 --port 8888 searx.webapp:app
   ) >/tmp/searxng.log 2>&1 &
   exec open-webui serve --host "$HOST" --port "$PORT" "$@"
   ```

4. Name the folder, the start script and the search settings for the
   client:

   ```yaml
   launch:
     container:
       clients:
         open-webui:
           build: ~/containers/open-webui-searxng
           command: [/usr/local/bin/start-webui]
           network: default
           env:
             - ENABLE_WEB_SEARCH=true
             - WEB_SEARCH_ENGINE=searxng
             - SEARXNG_QUERY_URL=http://127.0.0.1:8888/search
             - BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=true
             - BYPASS_WEB_SEARCH_WEB_LOADER=true
   ```

5. Launch Open WebUI with `gmlx launch open-webui --container`, and turn on
   Web Search in a chat from the Integrations menu next to `+`.

The Containerfile installs SearXNG at the commit that `SEARXNG_COMMIT`
names. It calls `/usr/bin/python3`, the Python of Debian, because a bare
`python3` in this image is the one in Open WebUI's own environment. The
settings turn on the JSON results that Open WebUI reads, and they keep the
SearXNG limiter off, because the limiter needs a Valkey database.

The start script starts SearXNG on port 8888 of the container, then Open
WebUI on the address that `launch` gives in `HOST` and `PORT`. SearXNG
refuses to start with the secret key it ships with, so the script gives it
a new key at each start. The script changes folder only in the subshell
that starts SearXNG. Open WebUI keeps its own key in the folder
it starts in, and a new folder would end every sign-in at the next session.

SearXNG asks other search engines, so the container needs
`network: default`. Port 8888 must differ from the port of the gmlx server
and from each [`forward`](config.md#launchcontainerforward) port.

By default, the model calls the search itself as a tool. A model set to the
legacy way of calling tools gets the search results in its prompt instead,
and the two `BYPASS` lines set how.

The first `BYPASS` line gives the model the results without an embedding
step. Without it, Open WebUI embeds the results, which needs the embeddings
service that [Setting up the services](rag.md#setting-up-the-services)
turns on.

With the second line, the model gets the snippets of the results instead of
whole pages, which keeps its prompt short. Remove it when the model has the
context to read whole pages.

Open WebUI reads these variables only at the first start of its data
folder, and keeps the values in its database after that. When Open WebUI
already ran in a container, set the same values once in Admin Panel >
Settings > Web Search instead.

SearXNG writes its log to `/tmp/searxng.log` in the container. To read it,
or to test a search, open a shell in the running session with
`gmlx launch open-webui --shell` and run:

```sh
curl -fsS 'http://127.0.0.1:8888/search?q=test&format=json' | head -c 300
```

Some search engines answer automated queries with a CAPTCHA, so the results
depend on the engines that SearXNG asks. A new SearXNG commit arrives only
when you change `SEARXNG_COMMIT`, which builds the image again.
