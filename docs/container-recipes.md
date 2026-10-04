# Container recipes

Complete setups for clients in [container mode](launch-container.md): pi
packages, dsh plugins, a headless browser, tool servers, Postgres and web
search for Open WebUI. Each one builds on
[Custom container images](container-images.md).

- [pi packages](#pi-packages)
- [dsh plugins](#dsh-plugins)
- [Headless browsers](#headless-browsers)
- [Tool servers](#tool-servers)
- [Postgres](#postgres)
- [Web search for Open WebUI](#web-search-for-open-webui)

## pi packages

A pi package adds extensions, skills, prompt templates or themes. Put it in
the image to have it in every project, with no network needed:

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
adds a `/btw` command for a side conversation. Add one `-e` for each
package, and pin an exact version. `--legacy-peer-deps` stops npm from
installing a second copy of pi.

To add a package to one project only, install it from the shell. It goes
into the private home:

```sh
cd ~/src/my-project
gmlx launch pi --shell -- -c "pi install npm:pi-btw@0.7.1"
```

pi checks for updates to packages without a version, which fails under
[`network: none`](config.md#launchcontainernetwork). Pin versions, or set
`PI_OFFLINE=1` in [`env`](config.md#launchcontainerenv).

To bring the extensions, skills and prompts you keep on the Mac, copy their
folders with [`seed`](config.md#launchcontainerclientsseed):

```yaml
launch:
  container:
    clients:
      pi:
        seed: [~/.pi/agent/extensions, ~/.pi/agent/skills, ~/.pi/agent/prompts]
```

Seed these folders, not all of `~/.pi/agent`, which also holds your
provider keys in `auth.json`.

## dsh plugins

dsh installs plugins with pnpm, which the image gmlx builds does not have.
Plugins go into the `gmlx` profile in the private home, so each project has
its own:

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

3. Launch dsh once in the project with `gmlx launch dsh --container`, so
   the `gmlx` profile exists before you add plugins to it.

4. Install the plugin from the shell, with the default network:

   ```sh
   cd ~/src/my-project
   gmlx launch dsh --shell -- -c "dsh plugin --profile gmlx add dsh-context@0.62.3"
   ```

The example installs
[dsh-context](https://github.com/bowenliang123/dsh-context), a context
dashboard for the web app. dsh loads it at its next start. The Plugins page
of the web app works too, once pnpm is in the image. Installed plugins keep
working under [`network: none`](config.md#launchcontainernetwork).

## Headless browsers

For screenshots of plain pages, add Chromium and fonts to the image:

```yaml
launch:
  container:
    clients:
      claude-code:
        packages: [chromium, fonts-liberation, fonts-noto-color-emoji]
```

The client can then take a screenshot:

```sh
chromium --headless --no-sandbox --disable-dev-shm-usage \
  --screenshot=/tmp/page.png https://example.com
```

`--no-sandbox` is needed because the client runs as root, and
`--disable-dev-shm-usage` because the container's shared memory is small.
Without the font packages, pages show text as empty boxes. Chromium prints
D-Bus errors in a container, which you can ignore.

For Puppeteer, add `PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium` to
[`env`](config.md#launchcontainerenv), because the Chrome it downloads has
no Linux arm64 build.

## Tool servers

An [MCP](glossary.md#mcp) tool server in the container gives the client
tools that reach only what the session reaches. This is the safe way to
give a coding agent tools, because the tools of a
[served assistant](container-security.md#what-the-client-reaches-on-the-server)
run on the Mac.

The quickest way is the client's own MCP command, run from the shell. The
entry stays in the private home, for that project. This gives pi a memory
server:

```sh
cd ~/src/my-project
gmlx launch pi --shell -- -c 'pi mcp add memory \
  --env MEMORY_FILE_PATH=$HOME/.pi/memory.jsonl \
  -- npx -y @modelcontextprotocol/server-memory@2026.8.31'
```

To have a tool server work in every project and with `network: none`, put
it in the image. This gives Claude Code a browser it drives through the
Playwright tool server:

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

The entry names Debian's Chromium, because Playwright's Chrome has no Linux
arm64 build. The entry lives in the image, out of the client's reach.

Avoid project files such as `.mcp.json` in the share for tool servers. The
client can change them, and
[clients on the Mac read them](container-security.md#shares-that-lead-back-to-the-mac).

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
[volume](container-access.md#volumes). It cannot keep its data in a share,
because every file there belongs to root. To run it in the container:

1. Write a Containerfile that installs Postgres and copies a start script:

   ```dockerfile
   FROM gmlx.invalid/launch-claude-code:base
   RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql \
    && rm -rf /var/lib/apt/lists/*
   COPY start-pg /usr/local/bin/start-pg
   RUN chmod 755 /usr/local/bin/start-pg
   ```

2. Beside it, write the start script `start-pg`. It starts Postgres as the
   `postgres` user, then runs the client:

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

The data survives from one session to the next, and the client connects
with `psql -U postgres`. To use Postgres on the Mac instead, give it a
password and add `forward: [5432]`, as
[Forwarded ports](container-access.md#forwarded-ports) describes.

## Web search for Open WebUI

Open WebUI can search the web through SearXNG, a metasearch engine that
runs in the same container, with no account at a search provider.

1. In a new folder `~/containers/open-webui-searxng`, write a
   Containerfile that installs SearXNG in a separate virtual environment:

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

   Open WebUI reads these only on its first start. If it already ran in a
   container, set the same values in Admin Panel > Settings > Web Search.

5. Launch Open WebUI with `gmlx launch open-webui --container`, and turn on
   Web Search in a chat from the Integrations menu next to `+`.

SearXNG asks other search engines, so the container needs
`network: default`. Some engines answer automated queries with a CAPTCHA,
so results vary. To pick up a newer SearXNG, change `SEARXNG_COMMIT`.

The two `BYPASS` lines matter only for a model set to the legacy way of
calling tools. They skip the embedding step and give the model snippets
instead of whole pages. Remove the second one if the model has the context
to read whole pages.

To check SearXNG, open a shell in the running session with
`gmlx launch open-webui --shell` and run:

```sh
curl -fsS 'http://127.0.0.1:8888/search?q=test&format=json' | head -c 300
```

Its log is `/tmp/searxng.log` in the container.
