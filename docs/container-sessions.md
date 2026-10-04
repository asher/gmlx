# Container sessions

When you launch a client with `--container`, gmlx starts a session: a small
Linux virtual machine that runs the client for the folder you are in. This
page covers what you do after the first launch: open more terminals, run apps
in the background, stop sessions and find their logs.

- [One session per project](#one-session-per-project)
- [Open a second terminal](#open-a-second-terminal)
- [End a session](#end-a-session)
- [Run an app in the background](#run-an-app-in-the-background)
- [See and stop sessions](#see-and-stop-sessions)
- [Open a shell in the container](#open-a-shell-in-the-container)
- [Logs](#logs)

## One session per project

Each folder you launch from is a separate project, with its own settings,
history and logins. `claude --continue` in `~/src/app` never picks up a
conversation from `~/src/site`.

```sh
cd ~/src/app  && gmlx launch claude-code --container   # a session for ~/src/app
cd ~/src/site && gmlx launch claude-code --container   # another one for ~/src/site
```

Sessions for different projects or different clients run side by side. Each
one is a virtual machine that uses memory, so end the ones you no longer
need. Open WebUI and elia do not work on a folder, so each of them has one
session for all folders.

[The private home](launch-container.md#the-private-home) explains where a
project's settings and history are kept.

## Open a second terminal

Run the same launch in another terminal, from the project folder or any
folder inside it:

```sh
gmlx launch claude-code --container
```

```
[launch] joining the running claude-code session for ~/src/app
```

The second terminal runs another copy of the client in the same container,
with the same files and the same private home. Arguments after `--` go to the
new copy, so `gmlx launch claude-code --container -- --resume` opens a
conversation from the project's history.

Flags that change a session, such as `--image`, `--rebuild` or `--network`,
cannot join. `launch` says which flag to leave out. To use them, end the
session and launch again.

A launch from a parent folder, or of another client, starts a separate
session. When two sessions then share the same files, `launch` warns you.
Two virtual machines that write the same files at the same time can damage
them.

## End a session

Quit the client, or close its terminal. The session ends when its last copy
exits, and gmlx removes the container. The project's private home stays for
the next launch.

If you quit the client in the first terminal while other copies still run,
that terminal waits for them. Press Ctrl-C twice there to end the whole
session.

Ctrl-Z does not suspend a client in a container. Quit the client when you
need the terminal back.

## Run an app in the background

Open WebUI, the dsh web profiles and custom agents can run without a
terminal:

```sh
gmlx launch open-webui --detach
```

```
[launch] open-webui runs in the background at http://[::1]:<port>/.
[launch] its output goes to ~/.cache/gmlx/launch/output-open-webui-default.log. gmlx launch --list shows the running sessions, and gmlx launch open-webui --stop in this folder ends this one.
```

Launch the app again to get its address. Claude Code and the other terminal
clients cannot run detached. To give a coding agent a task in the
background, define it as a custom agent, as
[Coding agents in the background](launch-agents.md#coding-agents-in-the-background)
shows.

## See and stop sessions

`gmlx launch --list` shows every session, and the command that ends each one:

```
TARGET       PROJECT            STATE    LAUNCH      ADDRESS               STARTED
claude-code  ~/src/app          running  foreground  -                     14:02
open-webui   (default project)  running  detached    http://[::1]:<port>/  13:40
[launch] the open-webui session writes its output to ~/.cache/gmlx/launch/output-open-webui-default.log
[launch] to end the claude-code session for ~/src/app, run gmlx launch claude-code --stop --no-mount-cwd --mount . in ~/src/app
[launch] to end the open-webui session, run gmlx launch open-webui --stop
```

`gmlx launch <name> --list` shows the sessions of one client, and
`gmlx status` lists them too.

To end a session, run `--stop` from the project folder:

```sh
gmlx launch claude-code --stop
```

This works for detached sessions and for sessions in another terminal. The
project keeps its private home. If a launch crashed and left its container
running, `--list` shows that container with the `container stop` command
that ends it.

## Open a shell in the container

`--shell` opens a shell with the same files, private home and server
connection as the client:

```sh
gmlx launch claude-code --shell                     # a shell instead of the client
gmlx launch claude-code --shell -- -c "npm test"    # run one command
```

While a session runs in the project, `--shell` opens the shell inside it, so
you can look at what the agent is doing. For a browser app, `--shell` starts
the session with a shell instead of the app, and prints the command that
starts the app.

Anything you install outside the private home is gone when the session ends.
To keep a tool, add it to the image, as
[Container images](container-images.md) shows.

## Logs

When a session fails or ends without a message, read its logs:

- `~/.cache/gmlx/launch/last-<client>-<project>.log` on the Mac holds what
  `launch` did.
- `.gmlx-entry.log` in the private home holds errors from inside the
  container.

`<project>` is the project folder's name followed by a short hash. A
detached session writes everything to the `output-<client>-<project>.log`
file in the same folder, which `--detach` and `--list` print.
