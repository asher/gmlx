# Custom container images

This page covers changing what the container of a
[container mode](launch-container.md) session contains: extra packages,
your own Containerfile, a ready-made image, and services that start with
the client. The keys it uses are in the
[configuration reference](config.md#launch).

- [What persists](#what-persists)
- [Extra packages](#extra-packages)
- [Your own Containerfile](#your-own-containerfile)
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
launches reuse it. `packages` applies only to the image gmlx builds, so it
cannot be combined with `image`.

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

`build` names the Containerfile, or a folder that holds a `Containerfile`
or `Dockerfile` and is also the build context. The path must be absolute
or start with `~`. Write the `gmlx.invalid/launch-<client>:base` reference
literally, since launch finds it by reading the file. Any client's `:base`
works, and launch refuses any other `gmlx.invalid` reference, because those
tags are deleted when a newer build replaces them.

Launch builds your image again when the Containerfile or a file in the
build context changes, and when a gmlx upgrade changes the base. A
`.dockerignore` in the context keeps folders such as `node_modules` out of
both the build and that check. A `<Containerfile>.dockerignore` beside the
Containerfile takes its place when it exists. A Containerfile must stay
under 16 KiB, which `container build` requires.

`--rebuild` rebuilds the base with fresh downloads, then builds your image
without its cache. It does not pull other registry images that your
Containerfile names again. A `packages` list reaches a `build` image only
when its Containerfile starts from that client's own `:base`, and launch
refuses the list otherwise.

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
again. The image must be for Linux on arm64 and contain the command that
runs, which is the client's own command unless
[`command`](config.md#launchcontainerclientscommand) says otherwise.
`--image REF` runs another image for one launch.

## Starting services with the client

To start a service before the client, give the image a start script and
name it first in [`command`](config.md#launchcontainerclientscommand). The
script starts its services, then runs the rest of its arguments:

```sh
#!/bin/sh
set -e
# Start services here.
exec "$@"
```

```yaml
launch:
  container:
    clients:
      claude-code:
        command: [/usr/local/bin/start.sh, claude]
```

A `command` list replaces the client's own command. The handlers of `elia`,
`dsh` and `open-webui` add arguments to it at each launch, so for them copy
the command that `--config-only` prints under the replaced command. A
service that refuses to run as root, such as Postgres, starts under its own
user with `runuser -u <user> --`.

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
- `/dev/shm` holds only 64 MB in the container, and
  `--disable-dev-shm-usage` keeps Chromium from running out of it.
- A slim image has almost no fonts, so pages render text as empty boxes
  without `fonts-liberation` and `fonts-noto-color-emoji`.

Chromium prints D-Bus errors in a container, because no D-Bus service
runs there. They do not affect the screenshot.

Puppeteer downloads its own Chrome by default, which may have no Linux
arm64 build. Set `PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium` in the
image to use the installed Chromium.

## Postgres

Postgres can run in the container with its data on a
[volume](launch-container.md#volumes), or on the Mac with a
[forwarded port](launch-container.md#forwarded-ports). A share does not
work for its data, because every file in a share appears to belong to root
and Postgres refuses a data folder it does not own.

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

The data lives in a subfolder of the volume, because the volume's root
holds `lost+found`. It survives from one session to the next, and the
client connects with `psql -U postgres`.

To use Postgres on the Mac instead, add `forward: [5432]`. Homebrew's
Postgres accepts local connections without a password, which would give
the client every database, so first create a role with a password and only
the rights the task needs.
