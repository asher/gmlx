# Container sessions

A container session runs one client for one project in an Apple container,
and this page covers how launches join, end, list and log these sessions.
Read it after your first launch in [Container mode](launch-container.md),
when you open a second terminal in a project, keep an app running in the
background or end a session that does not stop.

- [Projects and sessions](#projects-and-sessions)
- [Joining a running session](#joining-a-running-session)
- [When a session ends](#when-a-session-ends)
- [Stopping a session and logs](#stopping-a-session-and-logs)
- [The shell](#the-shell)
- [Sessions in the background](#sessions-in-the-background)
- [Listing and ending sessions](#listing-and-ending-sessions)

## Projects and sessions

A session runs one client for one project in a separate virtual machine.
The project is the current folder that the session shares, and `launch`
names it after that folder, with 16 hex digits of a hash of its real path,
such as `my-project-1a2b3c4d5e6f7a8b`. When a `--mount` or a `mounts`
entry shares the current folder, the project is the folder of that share,
the longest when several hold the current folder.

Launches whose shares do not hold the current folder belong to the
`default` project. Examples are elia, and a launch with `--no-mount-cwd`
and no mount that holds the current folder. Open WebUI always belongs to
the `default` project, because it keeps one store of chats. Every other
client, dsh in the browser included, gets a project for each folder.

Sessions of different projects or different clients run side by side, and
each one uses memory, which [Limits](container-security.md#limits)
counts. Each project keeps its
[private home](launch-container.md#the-private-home), so `--continue`,
history and the tools a client installs in its home stay with the project.

## Joining a running session

Another launch of the same client in the project joins the running session,
instead of starting a second virtual machine. A launch from a folder inside
the session's project folder or one of its read-write shares joins too,
and the session with the longest such folder takes the launch. A launch in
the `default` project joins the running session of the client's `default`
project.

The joining launch prints `joining the running <client> session for <folder>`
and runs another copy of the client in the same container, in the current
folder, with the arguments that it gets after `--`. From a folder that the
session does not share, as in the `default` project, the copy starts in the
session's working folder, and `launch` names the folders that the session
shares.

The copies share the private home inside one virtual machine, where file
locks work, as two copies of a client share a home on the Mac. File locks
do not reach from one virtual machine to another, and two sessions in one
home could corrupt the client's databases and settings. So two virtual
machines never share a private home, and a client gets a separate home in
each project.

These launches start a separate session instead of joining:

- A launch of another client in the project.
- A launch from a folder that holds a running session's project, such as
  its parent folder.
- A launch from a folder that a session shares read-only, outside its
  project folder, since a copy there could not change the files.

Two virtual machines then share the same files, and file locks do not reach
from one to the other, so `launch` warns and names the other session, even
one that is still starting. Launch the same client from the project folder
instead, or wait until the other session has ended.

A joining launch ignores the flags that chose the session's server and
model, such as `--model` or `--port`, with a note. A `--mount` joins when
the session already has that share, with the same folder, path and mode, so
the command that started a session joins it again.

`launch` refuses the other flags that shape a new session, which are
`--image`, `--rebuild`, `--reseed`, `--seed-instructions`, `--network`,
`--config-only` and `--provider-id`. It also refuses a dsh profile other
than the running one. The refusal says to leave out the flag to join, and
for `--mount` it lists the session's shares in the form that joins. To use
one of these flags, end the session and launch again.

While a session is still starting or is ending, a launch in its project
stops and says when to try again. A session is starting while its launch
starts the container service, prepares the image and boots the virtual
machine, and it is ending once its container stops. Another command that
holds the project, such as a `--remove-home` that waits for its answer or a
`--config-only` run, stops a launch the same way.

When `container ls` fails, `launch` cannot tell whether a session starts
or ends, so it stops. Its message says to try again once `container ls`
works, or to restart the container service, which also stops that session.

## When a session ends

The session lasts until its last copy exits. Closing the window of a joined
copy, or stopping its launch, ends only that copy. The copy gets SIGHUP, as
a client on the Mac does, and it is killed when it has not exited 10
seconds later.

The terminal that started the session stays with it. When the first copy
exits while others still run, that terminal says the session stays open
while they run, and it waits. A Ctrl-C there asks for a second one, which
ends the session.

When the session ends, or `launch` stops its container, the other copies
get SIGHUP, as from a closed terminal. A copy that exits within 5 seconds
prints
`the session ended in another terminal, so this copy of <command> stopped`.
Here `<command>` is the program that the copy runs, for example `claude` or
`bash`. A copy still running then stops with the container. Another Ctrl-C
in the first terminal stops them at once.

## Stopping a session and logs

Ctrl-C reaches the client as it does on the Mac. Closing the window of the
launch that started a session stops the session. A browser app or a custom
agent that should keep running without a window starts with
[`--detach`](#sessions-in-the-background).

`launch` writes the session log to
`~/.cache/gmlx/launch/last-<client>-<project>.log`, and the private home
keeps `.gmlx-entry.log` for errors inside the container. Read both when a
session fails or ends without a message.

The launch that started the session exits with the first copy's exit
code. [Exit codes](cli.md#exit-codes) lists the codes that `launch` and the
container return instead, a signal's code among them.

After the last copy of the client exits, `launch` removes the container and
its session files, and prints the `container delete --force` command for a
container that is still there. The next launch of that client in that
project cleans up a session whose launch was killed. Any other launch names
the leftover container with its `container stop` command, since that
virtual machine uses memory until it stops.

Ctrl-Z cannot suspend a client in the container. The client goes on
running, and the first Ctrl-Z says so. While the first terminal waits for
joined copies, Ctrl-Z there prints how to end the session, and the wait
goes on. Quit the client instead when you need the terminal.

The dsh web profiles run with no terminal in the container, so Ctrl-Z
suspends `launch` itself. The page then cannot reach dsh, and dsh cannot
reach the server, until you run `fg`.

When the launch that started the session is stopped, it stops the
container, and the client gets up to 10 seconds to exit. A second stop
kills the container, at the latest when those 10 seconds end. A stop that
arrives while the virtual machine starts waits until the container runs,
for up to a minute, and then acts. A third stop, for a container service
that no longer answers, kills `container run` and puts the terminal
settings back as they were.

While `launch` prepares the image, a Ctrl-C, SIGTERM or SIGHUP ends the
launch once its clean-up is done. `launch` ignores a second one, so that
its clean-up, which can include the stop of the image builder, finishes. A
second Ctrl-C also says to press Ctrl-C again to stop at once, and a third
one ends the clean-up too.

## The shell

`--shell` opens a shell in the same image, with the same shares, private
home and server connection as the client, instead of starting the client.
It runs `bash`, or `sh` when the image has no `bash`, and the arguments
after `--` go to the shell:

```sh
gmlx launch claude-code --shell
gmlx launch claude-code --shell -- -c "npm test"
```

While a session of that client runs for the project, `--shell`
[joins it](#joining-a-running-session) with a shell instead, to look at
what the agent is doing. The shell starts in the current folder when a
share of the session holds it, and in the session's working folder
otherwise. Like any joined copy, the shell keeps the session open until it
exits.

For a browser app, `--shell` starts the session with a shell and no app.
`launch` prints the address of the app and the command that starts it, such
as `dsh ... --port 3100` with the project's port, so run that command in
the shell.

With `command: image`, the printed command first changes to the image's
working folder, such as `cd /app/backend && bash start.sh` for the official
Open WebUI image. For a runtime agent, the printed command starts with
`uv run`, as [Sessions and data](launch-agents.md#sessions-and-data)
describes.

The first start of dsh makes its profile from the `web` template, and a
line says to leave out `--from-default-profile web` after that. dsh then
prints its address with `127.0.0.1`, so open that address with `[::1]` in
its place. While the shell runs, a second launch prints the start command
again, and `gmlx launch <client> --shell` opens another shell in the
session.

What you install from the shell outside the private home is gone when the
session ends, as
[What the client sees](launch-container.md#what-the-client-sees) explains.
An image with no shell at all makes `--shell` stop with a message, so add a
shell to a custom image to use it.

## Sessions in the background

`--detach` starts the session of Open WebUI, a dsh web profile or a custom
agent in the background and returns once the session runs. Until then,
`launch` shows the session's output. It then prints the app's address, the
file that takes the rest of the output, and the commands that list and end
the session:

```sh
gmlx launch open-webui --detach
gmlx launch --list
gmlx launch open-webui --stop
```

A detached session has no terminal, so `launch` refuses `--detach` with
`--shell` and for a client that needs a terminal, such as pi or
claude-code. To give Claude Code a task in the background, define it as a
custom agent, as
[Coding agents in the background](launch-agents.md#coding-agents-in-the-background)
shows. An agent without a browser interface gets empty input, and a program
that reads its input gets end of file at once.

All output of the session, from `launch` and from the client, goes to
`~/.cache/gmlx/launch/output-<client>-<project>.log`. The next detached
launch of the project empties that file. When the client's output in it
passes 64 MiB, `launch` empties the file and starts it with a line that
counts how often it did so, and the newest output stays. The image build
and other output before the session starts do not count toward that limit.

A second `--detach` of a browser app in the project prints the address of
the running app, as any second launch does. For an agent without a browser
interface, a second `--detach` is refused, since a copy that joins the
session needs a terminal, so launch without `--detach` to join it. While
the first session still starts or ends, the rule in
[Joining a running session](#joining-a-running-session) applies.

After the container starts, `--detach` waits up to 2 minutes for it to run.
For a browser app, it waits up to 5 minutes and 30 seconds for the app to
answer, so that the 5-minute wait in
[Browser apps](launch-container.md#browser-apps) ends first. When the time
ends, or at a Ctrl-C, `launch` stops waiting and the session goes on.
`launch` then prints the path of the output file, and
`gmlx launch --list` shows whether the session runs.

When the session ends during the wait, `launch` says so and exits with the
session's exit code. A launch that fails before its session runs prints its
message in your terminal. [Exit codes](cli.md#exit-codes) lists the code of
each case.

When Apple container has no Linux kernel, `--detach` asks the kernel
question in your terminal and shows the download there, and then starts the
session in the background.

## Listing and ending sessions

`gmlx launch --list` prints a table of the sessions that start, run or end,
and `gmlx launch <name> --list` limits it to one client or agent. Each row
shows the project folder, the state, whether `--detach` started the
session, the browser app's address and when its launch started. The
address leaves out its query, which holds dsh's login token, and a second
launch of dsh in the project opens the whole address.

Below the table, `--list` names the output file of each detached session
and the command that ends each session.
[`gmlx status`](cli.md#gmlx-status) prints a line for each session too.

`--stop` ends the session of the current project, whether `--detach`
started it or not. `--mount-cwd`, `--no-mount-cwd` and `--mount` choose
the project as they do for a launch, so the stop command that `--detach`
prints repeats the ones you passed. From a folder inside a session's
project folder, `--stop` ends the session that a launch from there would
join. The project keeps its private home, volumes and port.

To end a session, `--stop` sends SIGTERM to the launch that runs it, which
stops the container as closing its window does, and waits up to a minute
for that launch to exit. When the launch of a session is gone, `--stop`
stops and deletes the container itself. With no session in the project,
`--stop` names the command that ends each other session of the client.

The container of a session is named `gmlx-<client>-` followed by 6 hex
digits, which `container ls` shows. A container left over from a launch
that is gone gets its `container stop` command in the `--list` output.
`gmlx launch --list` shows every such container, also one of an agent whose
home is gone and one of an image check, whose target shows as
`image check`. The session of an agent that is no longer in
`launch.agents` gets its `container stop` command too, because
`gmlx launch` refuses that name for anything but `--list`.

While a session runs, `launch` keeps a session file for it in the project's
folder under `~/.local/share/gmlx/launch/<client>/projects`. Another launch
can hold the project before it has written that file, as in the moment
after `--detach` starts its launch in the background. `--stop` then asks
you to try again.

A session that has not ended after a minute, or a container that does not
stop, gets the command that ends it. That command is `container stop`, or
`kill -KILL` with the launch's process ID while the session has no
container yet.

`--list`, `gmlx status` and `--stop` wait up to 5 seconds for the container
service. When it does not answer, `--list` and `gmlx status` show a session
whose state `launch` cannot tell as `unknown`, list no leftover container
and print the error. `gmlx status` prints that error only when it lists a
session or the service gave no answer in time, since a stopped service
runs no container. `--stop` still ends a session that has a session file,
through its launch. Without one, and while the service has not stopped,
`--stop` cannot tell whether a container is left over, so it ends nothing
and prints the error.

`--stop` reads the launch settings. So when they do not load, `--list`
prints their error first. Each session then gets its `--stop` command, to
run once the settings load, and its `container stop` command when it has a
container. [Exit codes](cli.md#exit-codes) lists the code of each outcome
of `--stop`.
