//! Clipboard stand-ins for `clipboard: images`.
//!
//! Started under the name `xclip`, `xsel` or `wl-paste`, the entry answers
//! the two requests clients make to paste an image: the list of clipboard
//! types, and the data of an image type. It reads both from the Mac through
//! the session's clipboard socket. Every other request, such as reading
//! text or writing the clipboard, exits 1 with a one-line message.

use std::ffi::{OsStr, OsString};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;

/// The session's clipboard socket in the guest.
pub const CLIP_SOCK: &str = "/var/host-services/gmlx-clip.sock";
/// Names another socket, for tests on the Mac.
pub const CLIP_SOCK_ENV: &str = "GMLX_CLIP_SOCK";
/// The folder of the stand-in links, first on `PATH` under `--clipboard`.
pub const CLIP_BIN: &str = "/opt/gmlx/bin";

#[derive(Debug, PartialEq)]
pub enum Request {
    /// The clipboard's types, one per line.
    Types,
    /// The data of one image type.
    Image(String),
}

/// The stand-in `name` is, or None when the entry runs as itself.
pub fn tool_name(arg0: &OsStr) -> Option<&'static str> {
    let base = Path::new(arg0).file_name()?;
    ["xclip", "xsel", "wl-paste"].into_iter().find(|t| OsStr::new(t) == base)
}

fn write_refused(tool: &str) -> String {
    format!("{tool}: the container cannot write the Mac clipboard.")
}

fn text_refused(tool: &str) -> String {
    format!("{tool}: only images pass from the Mac clipboard into the container, not text.")
}

fn is_image(target: &str) -> bool {
    target.starts_with("image/")
}

fn parse_xclip(args: &[&str]) -> Result<Request, String> {
    let mut target: Option<&str> = None;
    let mut out = false;
    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "-o" | "-out" => out = true,
            "-i" | "-in" | "-f" | "-filter" => return Err(write_refused("xclip")),
            "-selection" | "-sel" | "-se" => {
                let value = args.get(i + 1).copied().unwrap_or("");
                if !matches!(value, "clipboard" | "c" | "clip") {
                    return Err(format!("xclip: only the clipboard selection comes from the \
                                        Mac, not {value}."));
                }
                i += 1;
            }
            "-t" | "-target" => {
                target = args.get(i + 1).copied();
                i += 1;
            }
            "-d" | "-display" | "-l" | "-loops" => i += 1,
            "-silent" | "-quiet" | "-noutf8" | "-r" | "-rmlastnl" | "-verbose" => {}
            other if other.starts_with('-') => {
                return Err(format!("xclip: {other} is not supported here."));
            }
            _ => return Err(write_refused("xclip")),        // a file to copy from
        }
        i += 1;
    }
    if !out {
        return Err(write_refused("xclip"));
    }
    match target {
        Some("TARGETS") => Ok(Request::Types),
        Some(t) if is_image(t) => Ok(Request::Image(t.to_string())),
        _ => Err(text_refused("xclip")),
    }
}

fn parse_wl_paste(args: &[&str]) -> Result<Request, String> {
    let mut target: Option<String> = None;
    let mut list = false;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i];
        match arg {
            "-l" | "--list-types" => list = true,
            "-n" | "--no-newline" => {}
            "-t" | "--type" => {
                target = args.get(i + 1).map(|s| s.to_string());
                i += 1;
            }
            "-p" | "--primary" => {
                return Err("wl-paste: only the clipboard selection comes from the Mac, not \
                            the primary selection.".into());
            }
            "-w" | "--watch" => return Err("wl-paste: --watch is not supported here.".into()),
            _ if arg.starts_with("--type=") => target = Some(arg["--type=".len()..].to_string()),
            _ => return Err(format!("wl-paste: {arg} is not supported here.")),
        }
        i += 1;
    }
    if list {
        return Ok(Request::Types);
    }
    match target {
        Some(t) if is_image(&t) => Ok(Request::Image(t)),
        _ => Err(text_refused("wl-paste")),
    }
}

/// The request a stand-in's arguments make, or the message it refuses with.
pub fn parse(tool: &str, args: &[OsString]) -> Result<Request, String> {
    let strs: Vec<&str> = args.iter().map(|a| a.to_str().unwrap_or("")).collect();
    match tool {
        "xclip" => parse_xclip(&strs),
        "wl-paste" => parse_wl_paste(&strs),
        _ => {
            if strs.iter().any(|a| matches!(*a, "-i" | "--input" | "-a" | "--append")) {
                Err(write_refused(tool))
            } else {
                Err(text_refused(tool))
            }
        }
    }
}

/// Sends the request and copies the answer to `out`. The Mac answers
/// `OK <length>` and that many bytes, or `ERR <message>`.
pub fn ask(sock: &Path, request: &Request, out: &mut dyn Write) -> Result<(), String> {
    let mut stream = UnixStream::connect(sock).map_err(|e| match e.kind() {
        io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused => {
            "the Mac clipboard is not available in this session. \
             Turn on clipboard: images in the launch config to paste images."
                .to_string()
        }
        _ => format!("cannot reach the Mac clipboard ({e})"),
    })?;
    let line = match request {
        Request::Types => "TYPES\n".to_string(),
        Request::Image(t) => format!("IMAGE {t}\n"),
    };
    stream.write_all(line.as_bytes()).map_err(|e| format!("cannot ask the Mac clipboard ({e})"))?;
    let mut reader = BufReader::new(stream);
    let mut status = String::new();
    reader.by_ref().take(1024).read_line(&mut status)
        .map_err(|e| format!("no answer from the Mac clipboard ({e})"))?;
    let status = status.trim_end();
    if let Some(message) = status.strip_prefix("ERR ") {
        return Err(message.to_string());
    }
    let length: u64 = status.strip_prefix("OK ").and_then(|n| n.parse().ok())
        .ok_or_else(|| "the Mac clipboard sent an answer this stand-in cannot read".to_string())?;
    let copied = io::copy(&mut reader.take(length), out)
        .map_err(|e| format!("the image did not arrive ({e})"))?;
    if copied != length {
        return Err("the image arrived cut short".into());
    }
    Ok(())
}

/// Runs the stand-in and returns its exit code.
pub fn run(tool: &str, args: &[OsString]) -> i32 {
    let request = match parse(tool, args) {
        Ok(r) => r,
        Err(message) => {
            eprintln!("{message}");
            return 1;
        }
    };
    let sock = std::env::var_os(CLIP_SOCK_ENV).unwrap_or_else(|| CLIP_SOCK.into());
    let stdout = io::stdout();
    let mut out = stdout.lock();
    match ask(Path::new(&sock), &request, &mut out).and_then(|()| out.flush().map_err(|e| e.to_string())) {
        Ok(()) => 0,
        Err(message) => {
            eprintln!("{tool}: {message}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;
    use std::thread;

    fn os(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    #[test]
    fn dispatch_by_program_name() {
        assert_eq!(tool_name(OsStr::new("/opt/gmlx/bin/xclip")), Some("xclip"));
        assert_eq!(tool_name(OsStr::new("wl-paste")), Some("wl-paste"));
        assert_eq!(tool_name(OsStr::new("xsel")), Some("xsel"));
        assert_eq!(tool_name(OsStr::new("/opt/gmlx/gmlx-entry")), None);
    }

    #[test]
    fn image_requests_the_clients_make() {
        let types = Request::Types;
        let png = Request::Image("image/png".into());
        assert_eq!(parse("xclip", &os(&["-selection", "clipboard", "-t", "TARGETS", "-o"])), Ok(types));
        assert_eq!(parse("xclip", &os(&["-selection", "clipboard", "-t", "image/png", "-o"])),
                   Ok(Request::Image("image/png".into())));
        assert_eq!(parse("wl-paste", &os(&["--list-types"])), Ok(Request::Types));
        assert_eq!(parse("wl-paste", &os(&["-l"])), Ok(Request::Types));
        assert_eq!(parse("wl-paste", &os(&["--type", "image/png", "--no-newline"])), Ok(png));
        assert_eq!(parse("wl-paste", &os(&["-t", "image/png"])),
                   Ok(Request::Image("image/png".into())));
        assert_eq!(parse("wl-paste", &os(&["--type=image/png"])),
                   Ok(Request::Image("image/png".into())));
    }

    #[test]
    fn refuses_text_writes_and_other_selections() {
        for (tool, args, word) in [
            ("xclip", vec!["-selection", "clipboard", "-o"], "not text"),
            ("xclip", vec!["-selection", "clipboard", "-t", "image/png", "-i", "f.png"], "cannot write"),
            ("xclip", vec!["-selection", "primary", "-o"], "only the clipboard"),
            ("xclip", vec!["-selection", "clipboard"], "cannot write"),
            ("wl-paste", vec![], "not text"),
            ("wl-paste", vec!["--type", "text/plain"], "not text"),
            ("wl-paste", vec!["--primary"], "only the clipboard"),
            ("xsel", vec!["--clipboard", "--output"], "not text"),
            ("xsel", vec!["--clipboard", "--input"], "cannot write"),
        ] {
            let err = parse(tool, &os(&args)).unwrap_err();
            assert!(err.contains(word), "{tool} {args:?}: {err}");
        }
    }

    fn serve_once(answer: &'static [u8]) -> (std::path::PathBuf, thread::JoinHandle<String>) {
        let dir = std::env::temp_dir().join(format!("ge-clip-{}-{}", std::process::id(),
                                                    answer.len()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("clip.sock");
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        let handle = thread::spawn(move || {
            let (conn, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(conn.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            (&conn).write_all(answer).unwrap();
            line
        });
        (path, handle)
    }

    #[test]
    fn asks_the_mac_and_copies_the_bytes() {
        let (path, handle) = serve_once(b"OK 4\n\x89PNG");
        let mut out = Vec::new();
        ask(&path, &Request::Image("image/png".into()), &mut out).unwrap();
        assert_eq!(out, b"\x89PNG");
        assert_eq!(handle.join().unwrap(), "IMAGE image/png\n");
    }

    #[test]
    fn passes_the_mac_message_on() {
        let (path, _handle) = serve_once(b"ERR there is no image on the Mac clipboard\n");
        let err = ask(&path, &Request::Types, &mut Vec::new()).unwrap_err();
        assert_eq!(err, "there is no image on the Mac clipboard");
    }

    #[test]
    fn missing_socket_names_the_config_key() {
        let err = ask(Path::new("/nonexistent/clip.sock"), &Request::Types, &mut Vec::new())
            .unwrap_err();
        assert!(err.contains("clipboard: images"), "{err}");
    }
}
