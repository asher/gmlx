# Custom container images

This page covers changing the image that a
[container mode](launch-container.md) session runs, from a few extra
packages to a custom image and services that start with the client.
Read it when a project needs tools that the shipped image lacks, and see
[Container recipes](container-recipes.md) for complete setups.

The keys it uses, with their rules and defaults, are in the
[configuration reference](config.md#launch). They take effect only in a
container launch, as
[Turning on container mode](launch-container.md#turning-on-container-mode)
describes.

- [Extra packages](#extra-packages)
- [Your own Containerfile](#your-own-containerfile)
- [A newer client](#a-newer-client)
- [A ready-made image](#a-ready-made-image)
- [What an image needs](#what-an-image-needs)
- [Starting services with the client](#starting-services-with-the-client)

## Extra packages

[`packages`](config.md#launchcontainerclientspackages) adds Debian packages
to the image that gmlx builds for a client:

```yaml
launch:
  container:
    clients:
      claude-code:
        packages: [make, python3, postgresql-client]
```

The next launch builds the image again with the packages, and later
launches reuse it. With a custom Containerfile, the packages apply only
through the client's `:base`, as the reference entry for
[`packages`](config.md#launchcontainerclientspackages) says.

## Your own Containerfile

For more than packages, write a Containerfile that starts from the image
gmlx builds for the client and name it with
[`build`](config.md#launchcontainerclientsbuild). The `:base` tag always
names the current gmlx image of that client:

```dockerfile
FROM gmlx.invalid/launch-claude-code:base
RUN apt-get update \
 && apt-get install -y --no-install-recommends make python3 \
 && rm -rf /var/lib/apt/lists/*
```

```yaml
launch:
  container:
    clients:
      claude-code:
        build: ~/containers/claude-code
```

`build` names a folder with a file named `Containerfile` or `Dockerfile`,
and that folder is the build context. It can also name the Containerfile
itself, and then the Containerfile's folder is the build context. A
Containerfile must be a regular file under 16 KiB, which `container build`
requires.

Write the `gmlx.invalid/launch-<client>:base` reference literally, since
`launch` finds it by reading the file. Any client's `:base` works, and so
does `gmlx.invalid/launch-runtime-python:base`, the image that runs
[custom agents](launch-agents.md#your-own-image) with `runtime: python`.
`launch` refuses any other `gmlx.invalid` reference, because those tags are
deleted when a newer build replaces them. An agent's `build` follows the
rules of this section.

Keep the build folder out of every folder a session shares read-write. The
client could change it there, and its change would run at the next build
with internet access. For this reason, `launch` refuses a read-write share
that overlaps the build folder of any client or agent, or that contains a
link on the path to it. It also refuses to build from a folder or
Containerfile that overlaps a read-write share of the session, a folder an
earlier launch shared read-write, or the private homes.

`launch` builds your image again when the Containerfile or a file in the
build context changes, and when a gmlx upgrade changes the base. The line
`rebuilding because <files> changed` names up to three of the changed
files.

List folders like `node_modules` in a `.dockerignore` in the context to
keep them out of both the build and that check. A
`<Containerfile>.dockerignore` beside the Containerfile replaces
`.dockerignore` when it exists. `launch` leaves the `.git` folder at the
root of the context out of that check, but the build still receives it, so
list `.git` in the ignore file to keep it out of the image.

When `launch` cannot use the ignore file, it prints the reason, and every
change in the context then rebuilds the image.

`--rebuild` builds your image again without its cache. When the
Containerfile names no `:base`, it also pulls the registry images the
Containerfile starts from again. When it names a `:base`, `launch` first
rebuilds that base without its cache, and it does not pull the other
registry images your Containerfile names.

Apple's image builder is a separate virtual machine that uses memory while
it runs. `launch` stops a builder that its build started once no other
build uses it, or at the next launch when a signal ends the build. For any
other idle builder, `launch` and `gmlx doctor` print the
`container builder stop` command.

A build never gets your SSH agent. `launch` refuses to build while the
builder forwards the agent, and
[troubleshooting](troubleshooting.md#launch-refuses-to-build-while-the-builder-forwards-your-ssh-agent)
gives the fix.

## A newer client

The image that gmlx builds pins each client at one version, so a newer
version of the client arrives with a gmlx release. To run one sooner,
install it over the client's `:base` in a custom Containerfile, and name
its folder with [`build`](config.md#launchcontainerclientsbuild):

```dockerfile
FROM gmlx.invalid/launch-claude-code:base
RUN npm install -g @anthropic-ai/claude-code@<version>
```

Write an exact version, never `latest`. `launch` rebuilds the image when
the Containerfile changes, so a new version number rebuilds it, while
`latest` moves only when something else rebuilds the image. The install
line for each client is:

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

The shipped Containerfile gives the download address and the checksum form
for `omp`, `goose` and `aichat`. The copy that matches your gmlx is
`gmlx/container/files/Containerfile` in the installed package, and the
[newest one](https://github.com/asher/gmlx/blob/main/gmlx/container/files/Containerfile)
is on GitHub. When a gmlx upgrade moves the client past your version, remove
the line again.

## A ready-made image

[`image`](config.md#launchcontainerclientsimage) runs an image from a
registry or your local image store instead:

```yaml
launch:
  container:
    clients:
      opencode:
        image: docker.io/me/opencode-box:1
```

`launch` pulls a missing image once and then runs it by its content digest,
so a later push to the same tag changes nothing until `--rebuild` pulls it
again. `--image REF` runs another image for one launch. The image must have
what [What an image needs](#what-an-image-needs) lists.

## What an image needs

Any image works when it is for Linux on arm64 and contains the command that
runs, and `launch` refuses an image for another architecture. `launch`
checks an image from `image` or `build` once for each command. A command
that is missing or cannot run stops the launch before the session starts,
and under `--shell` the check only warns.

A program built for another system passes the check and fails when the
session starts.
[Troubleshooting](troubleshooting.md#a-command-is-not-in-the-image) covers
both cases.

### The command that runs

The container runs the client's command, followed by the arguments after
`--`. [`command`](config.md#launchcontainerclientscommand) changes it:

- A list replaces the client's command with its arguments. `launch` still
  writes the client's configuration, and the arguments after `--` follow
  the list.
- The word `image` runs the ENTRYPOINT and CMD of the image, where an
  ENTRYPOINT of `[""]` counts as none. The arguments after `--` replace
  CMD, and the container starts in the image's working folder when it sets
  one.

goose runs as `goose session`, and `launch` adds arguments to the commands
of `elia`, `dsh` and `open-webui` at each launch. For these
clients, copy the arguments into a `command` list from the dry run's line
`the command: setting replaces the client's own command`.

## Starting services with the client

A start script in the image can start a database, a search engine or
another service before the client. The script comes first in
[`command`](config.md#launchcontainerclientscommand) and runs the client
when its services run:

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

3. Name the folder and the command:

   ```yaml
   launch:
     container:
       clients:
         claude-code:
           build: ~/containers/claude-code
           command: [/usr/local/bin/start.sh, claude]
   ```

The script starts its services, then runs the rest of its arguments, here
`claude`. A service that refuses to run as root starts as another user with
`runuser -u <user> --`, which the
[Postgres](container-recipes.md#postgres) recipe uses.
