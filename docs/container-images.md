# Custom container images

The image gmlx builds for a client holds the client, Node.js, git, ripgrep,
curl and ssh. When a project needs more, such as `python3`, `make` or a
database, this page shows how to add it, from one line of config to an image
of your own. [Container recipes](container-recipes.md) has complete setups.

- [Extra packages](#extra-packages)
- [Your own Containerfile](#your-own-containerfile)
- [A newer client](#a-newer-client)
- [A ready-made image](#a-ready-made-image)
- [What an image needs](#what-an-image-needs)
- [Starting services with the client](#starting-services-with-the-client)

## Extra packages

Most projects need only a few Debian packages. List them in
[`packages`](config.md#launchcontainerclientspackages):

```yaml
launch:
  container:
    clients:
      claude-code:
        packages: [make, python3, postgresql-client]
```

The next launch rebuilds the image with them, and later launches reuse it.

## Your own Containerfile

For anything more, write a Containerfile. gmlx tags each client's image as
`gmlx.invalid/launch-<client>:base`. The `.invalid` domain can never exist,
so no registry can serve an image with that name. Start from it to build on
the image gmlx made:

```dockerfile
FROM gmlx.invalid/launch-claude-code:base
RUN apt-get update \
 && apt-get install -y --no-install-recommends make python3 \
 && rm -rf /var/lib/apt/lists/*
```

Put the Containerfile in a folder of its own, and name that folder with
[`build`](config.md#launchcontainerclientsbuild):

```yaml
launch:
  container:
    clients:
      claude-code:
        build: ~/containers/claude-code
```

`launch` rebuilds the image when the Containerfile or another file in the
folder changes, and when a gmlx upgrade changes the base. List large folders
such as `node_modules` in a `.dockerignore` to keep them out of the build.
`--rebuild` builds again from scratch, which also brings Debian updates.

Write the `FROM` line exactly as shown, because `launch` reads it to find the
base. Any client's `:base` works, and so does
`gmlx.invalid/launch-runtime-python:base`, the base for
[custom agents](launch-agents.md#your-own-image) with `runtime: python`.

Keep the build folder outside every folder a session shares read-write. Its
code runs at the next build with internet access, so `launch` refuses a
build folder that a client could change.

## A newer client

The image gmlx builds pins each client at one version, and a gmlx release
brings newer ones. To run a newer client sooner, install it over the base:

```dockerfile
FROM gmlx.invalid/launch-claude-code:base
RUN npm install -g @anthropic-ai/claude-code@<version>
```

Write an exact version, never `latest`. A new version number in the file
triggers a rebuild, while `latest` does not. The install line for each
client is:

| Client | Install line |
|--------|--------------|
| `claude-code` | `npm install -g @anthropic-ai/claude-code@<version>` |
| `opencode` | `npm install -g opencode-ai@<version>` |
| `pi` | `npm install -g @earendil-works/pi-coding-agent@<version>` |
| `dsh` | `npm install -g @deepseek-ai/dsh@<version>` |
| `hermes` | `/opt/venv/bin/pip install --no-cache-dir "hermes-agent[mcp]==<version>"` |
| `elia` | `/opt/venv/bin/pip install --no-cache-dir elia-chat==<version>` |
| `open-webui` | `/opt/venv/bin/pip install --no-cache-dir open-webui==<version>` |
| `omp`, `goose`, `aichat` | A release download for Linux on arm64 into `/usr/local/bin`, as in the Containerfile that gmlx ships. |

For `omp`, `goose` and `aichat`, copy the download and checksum lines from
the shipped Containerfile, `gmlx/container/files/Containerfile` in the
installed package. Remove your line again once a gmlx upgrade catches up.

## A ready-made image

To run an image from a registry or your local image store, name it with
[`image`](config.md#launchcontainerclientsimage):

```yaml
launch:
  container:
    clients:
      opencode:
        image: docker.io/me/opencode-box:1
```

`launch` pulls it once and keeps running that exact image until
`--rebuild` pulls it again. `--image REF` runs another image for one launch.

## What an image needs

An image must be for Linux on arm64 and contain the command that runs.
`launch` checks this before the first session and stops with a message when
the command is missing.
[A command is not in the image](troubleshooting.md#a-command-is-not-in-the-image)
covers the fixes.

### The command that runs

The container runs the client's command, followed by the arguments after
`--`. [`command`](config.md#launchcontainerclientscommand) changes it:

- A list replaces the client's command. `launch` still writes the client's
  configuration.
- `image` runs the image's own ENTRYPOINT and CMD, and the arguments after
  `--` replace CMD.

Some clients get extra arguments from `launch`, such as `goose session`. To
keep them in a `command` list, copy them from the
[dry run](launch-container.md#the-dry-run).

## Starting services with the client

A start script can start a database or another service before the client:

1. In a new folder, write `start.sh`:

   ```sh
   #!/bin/sh
   set -e
   # Start services here.
   exec "$@"
   ```

2. Beside it, write a Containerfile that copies the script into the image:

   ```dockerfile
   FROM gmlx.invalid/launch-claude-code:base
   COPY start.sh /usr/local/bin/start.sh
   RUN chmod 755 /usr/local/bin/start.sh
   ```

3. Name the folder and put the script before the client's command:

   ```yaml
   launch:
     container:
       clients:
         claude-code:
           build: ~/containers/claude-code
           command: [/usr/local/bin/start.sh, claude]
   ```

The script starts its services and then runs `claude`. A service that will
not run as root can start with `runuser -u <user> --`, as the
[Postgres](container-recipes.md#postgres) recipe shows.
