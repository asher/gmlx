# Custom container images

This page covers changing what runs in a
[container mode](launch-container.md) session. It goes from a few extra
packages to an image of your own and services that start with the client,
and [Container recipes](container-recipes.md) applies these methods to
common tasks.

The keys it uses, with their rules and defaults, are in the
[configuration reference](config.md#launch). They take effect only in a
container launch, as
[Turning on container mode](launch-container.md#turning-on-container-mode)
describes.

- [What persists](#what-persists)
- [Extra packages](#extra-packages)
- [Your own Containerfile](#your-own-containerfile)
- [A newer client](#a-newer-client)
- [A ready-made image](#a-ready-made-image)
- [Starting services with the client](#starting-services-with-the-client)

## What persists

The shares, the volumes and the images persist between sessions, and so
does the [private home](glossary.md#private-home) of each project, with all
that the client or a shell writes under it. Everything else the container
writes is discarded when the session ends.

A package that `apt-get install` or `npm install -g` adds from `--shell`
lands in the image's own folders, so it is gone at the next session. To
keep such a tool, put it in the image with one of the methods on this page.

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
launches reuse it. With your own Containerfile, the packages apply only
through the client's own `:base`, as
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

`build` names a folder that holds a file named `Containerfile` or
`Dockerfile`, and that folder is the build context. It can also name the
Containerfile itself, and then the folder that holds the file is the build
context. A Containerfile must be a regular file under 16 KiB, which
`container build` requires.

Write the `gmlx.invalid/launch-<client>:base` reference literally, since
launch finds it by reading the file. Any client's `:base` works, and so
does `gmlx.invalid/launch-runtime-python:base`, the image that runs
[custom agents](launch-agents.md#your-own-image) with `runtime: python`.
Launch refuses any other `gmlx.invalid` reference, because those tags are
deleted when a newer build replaces them. An agent's `build` takes a
Containerfile by the rules of this section.

Keep the build folder out of every folder a session shares read-write.
The client could change it there, and its change would run at the next
build with internet access.

A read-write share that holds or lies in the build folder of any client or
agent, or that holds a link on the way to it, is refused. Launch also
refuses to build from a folder or Containerfile that overlaps a read-write
share of the session, a folder an earlier launch shared read-write, or the
private homes.

Launch builds your image again when the Containerfile or a file in the
build context changes, and when a gmlx upgrade changes the base. The line
`rebuilding because <files> changed` names up to three of the changed
files.

A `.dockerignore` in the context keeps folders such as `node_modules` out
of both the build and that check. A `<Containerfile>.dockerignore` beside
the Containerfile takes its place when it exists. Launch leaves the `.git`
folder at the root of the context out of that check, but the build still
receives it, so list `.git` in the ignore file to keep it out of the image.

Launch reads the ignore file when it is an ordinary file of at most 1 MiB
with at most 200 patterns that it can match the way `container build`
does. Otherwise it prints a line that names the reason, and every change in
the context rebuilds the image.

`--rebuild` builds your image again without its cache. When the
Containerfile names no `:base`, it also pulls the registry images the
Containerfile starts from again. When it names a `:base`, launch first
rebuilds that base without its cache, and it does not pull the other
registry images your Containerfile names.

Apple's image builder is a virtual machine of its own, and it holds memory
while it runs. Launch stops a builder that its own build started, once no
other build uses it. A builder that launch did not start, and that keeps
running with no build, gets one line with the `container builder stop`
command, and `gmlx doctor` reports it too.

When signals end a launch before it stops its builder, launch prints a
line that says the builder may still run, with the `container builder stop`
command. The next launch stops that builder, and `gmlx doctor` reports it
until then. When the stop fails, launch prints the `container builder stop`
command, or the restart of the container service when the service does not
answer.

A build never gets your SSH agent. Launch refuses to build while the
builder forwards the agent, as
[Launch refuses to build while the builder forwards your SSH agent](troubleshooting.md#launch-refuses-to-build-while-the-builder-forwards-your-ssh-agent)
explains.

## A newer client

The image that gmlx builds pins each client at one version, so a newer
version of the client arrives with a gmlx release. To run one sooner,
install it over the client's `:base` in your own Containerfile, and name
its folder with [`build`](config.md#launchcontainerclientsbuild):

```dockerfile
FROM gmlx.invalid/launch-claude-code:base
RUN npm install -g @anthropic-ai/claude-code@<version>
```

Write an exact version, never `latest`. Launch rebuilds the image when the
Containerfile changes, so a new version number rebuilds it, while `latest`
moves only when something else rebuilds the image. The install line for
each client is:

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

Launch pulls a missing image once and then runs it by its content digest,
so a later push to the same tag changes nothing until `--rebuild` pulls it
again. The image must meet the requirements in
[The image](launch-container.md#the-image), with the client's own command
or the one [`command`](config.md#launchcontainerclientscommand) names.
`--image REF` runs another image for one launch.

## Starting services with the client

To start a service before the client, add a start script to the image and
name it first in [`command`](config.md#launchcontainerclientscommand):

1. In a folder of its own, write `start.sh`. It starts its services, then
   runs the rest of its arguments:

   ```sh
   #!/bin/sh
   set -e
   # Start services here.
   exec "$@"
   ```

2. Beside it, write a Containerfile that copies the script into the image
   and gives it its execute bit:

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

A `command` list replaces the client's own command, with its arguments.
goose runs as `goose session`, and the handlers of `elia`, `dsh` and
`open-webui` add arguments at each launch, so for them copy the arguments
from the dry run's line `the command: setting replaces the client's own
command`.

A service that refuses to run as root starts under its own user with
`runuser -u <user> --`, as the [Postgres](container-recipes.md#postgres)
recipe shows.
