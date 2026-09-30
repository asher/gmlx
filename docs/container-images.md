# Custom container images

This page covers changing what runs in the container of a
[container mode](launch-container.md) session, from a few extra packages to
an image of your own and services that start with the client. The keys it
uses, with their rules and defaults, are in the
[configuration reference](config.md#launch).

- [What persists](#what-persists)
- [Extra packages](#extra-packages)
- [Your own Containerfile](#your-own-containerfile)
- [A newer client](#a-newer-client)
- [A ready-made image](#a-ready-made-image)
- [Starting services with the client](#starting-services-with-the-client)
- [Headless browsers](#headless-browsers)
- [Postgres](#postgres)

## What persists

The shares, the private home, the volumes and the images persist between
sessions. Everything else the container writes is discarded when the
session ends, including packages you install from `--shell`. To keep a
tool, put it in the image with one of the methods on this page.

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
launch finds it by reading the file. Any client's `:base` works, and launch
refuses any other `gmlx.invalid` reference, because those tags are deleted
when a newer build replaces them.

Keep the build folder out of every folder a session shares read-write.
The client could change it there, and its change would run at the next
build with internet access. Launch refuses a read-write share that holds or
lies in any client's build folder. It refuses to build from a folder or
Containerfile that overlaps a read-write share of the session, a folder an
earlier launch shared read-write, or the private homes of the clients.

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
other build uses it. A builder that keeps running with no build gets one
line with the `container builder stop` command, and `gmlx doctor` reports
it too.

A build never gets your SSH agent. Launch refuses to build while the
builder forwards the agent, as
[the troubleshooting entry](troubleshooting.md#launch-refuses-to-build-while-the-builder-forwards-your-ssh-agent)
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
| `pi` | `npm install -g @mariozechner/pi-coding-agent@<version>` |
| `dsh` | `npm install -g @deepseek-ai/dsh@<version>` |
| `hermes` | `/opt/venv/bin/pip install --no-cache-dir hermes-agent==<version>` |
| `elia` | `/opt/venv/bin/pip install --no-cache-dir elia-chat==<version>` |
| `open-webui` | `/opt/venv/bin/pip install --no-cache-dir open-webui==<version>` |
| `omp`, `goose`, `aichat` | A release download for Linux on arm64 into `/usr/local/bin`, as in the Containerfile that gmlx ships. |

The [shipped Containerfile](https://github.com/asher/gmlx/blob/main/gmlx/container/files/Containerfile)
gives the download address and the checksum form for `omp`, `goose` and
`aichat`. When a gmlx upgrade moves the client past your version, remove
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
name it first in [`command`](config.md#launchcontainerclientscommand). The
script starts its services, then runs the rest of its arguments. Put it in
a folder as `start.sh`:

```sh
#!/bin/sh
set -e
# Start services here.
exec "$@"
```

Beside it, a Containerfile copies the script into the image and gives it
its execute bit:

```dockerfile
FROM gmlx.invalid/launch-claude-code:base
COPY start.sh /usr/local/bin/start.sh
RUN chmod 755 /usr/local/bin/start.sh
```

Then name the folder and the command:

```yaml
launch:
  container:
    clients:
      claude-code:
        build: ~/containers/claude-code
        command: [/usr/local/bin/start.sh, claude]
```

A `command` list replaces the client's own command. The handlers of `elia`,
`dsh` and `open-webui` add arguments to it at each launch, so for them copy
the arguments from the dry run's line `the command: setting replaces the
client's own command`. A service that refuses to run as root, such as
Postgres, starts under its own user with `runuser -u <user> --`.

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

## Postgres

Postgres can run in the container with its data on a
[volume](launch-container.md#volumes), or on the Mac with a
[forwarded port](launch-container.md#forwarded-ports). A share does not
work for its data, because Postgres refuses a data folder it does not own
and a share keeps no owners in the container.

To run it in the container, put this Containerfile and a `start-pg` script
in one folder:

```dockerfile
FROM gmlx.invalid/launch-claude-code:base
RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql \
 && rm -rf /var/lib/apt/lists/*
COPY start-pg /usr/local/bin/start-pg
RUN chmod 755 /usr/local/bin/start-pg
```

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

Then name the folder, the start script and a volume:

```yaml
launch:
  container:
    clients:
      claude-code:
        build: ~/containers/claude-pg
        command: [/usr/local/bin/start-pg, claude]
        volumes: [claude-pg:/var/lib/postgresql:8G]
```

The data lives in a subfolder of the volume, as
[Volumes](launch-container.md#volumes) advises. It survives from one
session to the next, and the client connects with `psql -U postgres`.

To use Postgres on the Mac instead, add `forward: [5432]`, after you give
it a password or a limited role as
[Forwarded ports](launch-container.md#forwarded-ports) says.
