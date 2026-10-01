# Container security

This page describes what a client in [container mode](launch-container.md)
can still reach on the Mac and on the gmlx server, and the limits a session
runs under. Read it before you share a folder read-write or turn on an
option that gives the client more access.

- [Shares that lead back to the Mac](#shares-that-lead-back-to-the-mac)
- [Access you turn on](#access-you-turn-on)
- [What the client reaches on the server](#what-the-client-reaches-on-the-server)
- [Limits](#limits)

## Shares that lead back to the Mac

The container limits what the client can reach. It does not limit what the
client does in the folders you share, and a read-write share leads back to
the Mac in these ways:

- Files the client writes in a share run on the Mac when you use them.
  Examples are `.git/hooks`, `.git/config`, `.envrc` and the scripts in
  `package.json`, so read what the client changed there before you run the
  project on the Mac.
- A `gmlx.yaml` the client writes in a share takes effect only when you
  pass it with `--config`. It can then change where the server listens,
  turn off its key or add a tool server command that the server runs on the
  Mac, so read it before you use it.
- When the server's config file, a model folder it scans or a model file it
  lists is in a read-write share, the client can change what the server
  loads, and launch prints a warning. Move that file or folder out of the
  share, or share it read-only with `--mount PATH:ro`. When the running
  server has no config file, or an older gmlx started it, launch cannot
  check it and prints a line that names the fix.
- A client's [`build`](config.md#launchcontainerclientsbuild) folder runs
  its code at the next build, with internet access even under
  `network: none`, so launch keeps it out of every read-write share, as
  [Your own Containerfile](container-images.md#your-own-containerfile)
  describes.
- Launch warns when `PYTHONPATH` has an empty or relative entry, because a
  `gmlx` package that the client writes in a share would then run in a
  `gmlx` command you start. Remove that entry. An empty entry is what
  `export PYTHONPATH="$PYTHONPATH:/x"` leaves when the variable was unset.

Launch checks every shared folder again right before the container starts,
and it stops when one has changed, such as a folder that another session's
client replaced with a link. A change after that check still reaches the
container, so share only folders that no other session can write.

## Access you turn on

Only the variables in [`env`](config.md#launchcontainerenv) and the ones the
client's configuration needs reach the container.

[`ssh_agent`](config.md#launchcontainerssh_agent) lets the client use the
SSH agent on the Mac while the session runs. The client can sign with every
key loaded in that agent, and it can also remove keys from the agent. Load
only the keys the task needs.

A deploy key in the [private home](launch-container.md#ssh-in-the-container)
reaches only its own repository, so use one when one repository is enough.
The client can still copy that key and use it after the session.

Each [forwarded port](launch-container.md#forwarded-ports) gives the client
a Mac service with the rights of a local user. With
[clipboard images](launch-container.md#clipboard-images) on, the client can
read the clipboard image at any time during the session.

The container reaches the internet and your local network unless you set
[`network: none`](config.md#launchcontainernetwork). On the default
network, a Mac service that listens on all addresses is reachable from the
container, and so is any device on your network. The connection to the
server needs no sudo, changes no network setting and raises no firewall
prompt, and the server sees `Host: 127.0.0.1:<port>` on every request.

## What the client reaches on the server

The client reaches only the inference routes of the gmlx server, and only
the served assistants you list for it. It connects through a
[session socket](glossary.md#session-socket), which the server opens for
that session alone, not through the server's port. The socket needs no
key, so the client's configuration holds the placeholder key
`gmlx-container-session` and never the server's key.

Those routes are the model list, chat, text completions, responses,
messages, embeddings, rerank, speech, transcription, images and `systemone`,
plus `/health`. Every other route answers 404, so the client cannot unload
or keep models, reload the server's configuration or open another socket.

Served assistants stay hidden from the client unless its
[`assistants`](config.md#launchcontainerclientsassistants) key lists them.
A request that names any other assistant gets the answer for an unknown
model, and the model list leaves it out. Launch prints one line for each
assistant the client can use, with the tool servers it calls:

```text
[launch] open-webui can use assistant home, whose tools run on the Mac: web, files
```

An assistant's tools run on the Mac with your rights, outside the
container, and the messages the client sends decide which tools it calls.
In a chat app such as Open WebUI, you write those messages, so the risk is
modest. A coding agent also sends text from the files, command output and
web pages it reads, and any of them can carry instructions for the tools.
Give a coding agent no assistants.

A request through the socket takes an image, audio or video only as inline
data. The client therefore cannot make the server read a Mac file, even one
in the server's [media folder](api.md#media-in-requests), or fetch a URL,
even with [`server.media_urls`](config.md#servermedia_urls) on. The
socket's body limits are smaller than those of the TCP port, as
[Limits and back-pressure](api.md#limits-and-back-pressure) lists.

With [`server.stt`](config.md#serverstt) set, the transcription and
translation routes write the client's upload to a temporary file and run
ffmpeg on it. ffmpeg on the Mac therefore parses bytes the client chose, so
keep it up to date, or leave `server.stt` unset on a server that container
clients use.

A [browser app](launch-container.md#browser-apps) page runs in your browser
on the Mac. While the session is open, the server refuses the requests that
a local page on the web port sends to its TCP port, so the page reaches the
server only through the socket. Another gmlx server on the Mac answers that
page as it answers any local page, so set a
[`server.api_key`](config.md#serverapi_key) on any other server you run.

These limits apply to a plain http server on this Mac, which is a server
whose host resolves only to loopback addresses or the Mac's own. With
`--base-url` naming another host or an https URL, launch opens no socket and
prints a line saying so. The client then gets the key you pass with
`--api-key`, and it can do all that key allows on that server.

A local server that offers no session sockets refuses container mode, since
launch cannot limit it. When that server is gmlx, the message says to run
`gmlx restart`, so that it runs the installed version.

The socket ends with the session. When the server restarts during a
session, launch asks it for a new socket, with the same assistants, at the
client's next request. When the server gives none, such as after a restart
with another API key, launch prints the reason after the client exits.

## Limits

Each file the container reads in a share holds one file handle on the Mac
until the container stops. A client that reads a very large tree, such as a
home folder full of projects, can reach the Mac's limit for one process,
and two such sessions can reach the limit of the whole Mac. The two limits
differ from Mac to Mac, and `sysctl kern.maxfilesperproc kern.maxfiles`
prints them. Share narrow folders.

[`gmlx doctor`](cli.md#gmlx-doctor) reports the open file count and both
limits, and it warns when more than half of the Mac's limit is open while a
session runs. Stop the session to release the handles.

The container's memory counts against the model server's memory until the
container stops, even when the client inside frees it.
[`memory`](config.md#launchcontainermemory) sets its size, and launch warns
once for each size above a quarter of the Mac's memory. When other launch
containers already run, launch prints the memory that all of them and the
new one will hold, against the Mac's.
