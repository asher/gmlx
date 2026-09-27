//! Guest entry for `gmlx launch --container`.
//!
//! `gmlx-entry [--tcp PORT=SOCK]... [--unix SOCK=PORT]... [--clipboard] [--shell] -- CMD ARGS`
//! binds the relay listeners, starts the relay as a detached process, and
//! replaces itself with the client. `gmlx-entry --check CMD` only resolves CMD.
//! Started as `xclip`, `xsel` or `wl-paste`, the binary is a clipboard
//! stand-in instead. The binary is static and needs nothing from the image
//! but the command it runs.

mod clipboard;
mod relay;

use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{exit, Command};

/// A relay listener failed to bind.
pub const EXIT_LISTEN: i32 = 125;
/// The command was found but could not be run.
pub const EXIT_CANNOT_RUN: i32 = 126;
/// The command is not in the image.
pub const EXIT_NOT_FOUND: i32 = 127;
/// The arguments are malformed.
pub const EXIT_USAGE: i32 = 2;

/// The search path when the image sets no `PATH`.
pub const DEFAULT_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";

const USAGE: &str = "usage: gmlx-entry [--tcp PORT=SOCK]... [--unix SOCK=PORT]... \
                     [--clipboard] [--shell] -- CMD [ARGS]...\n       gmlx-entry --check CMD";

#[derive(Debug, PartialEq)]
pub enum Mode {
    Run(RunSpec),
    Check(OsString),
}

#[derive(Debug, PartialEq, Default)]
pub struct RunSpec {
    /// Guest TCP ports on 127.0.0.1, each relayed to a Unix socket.
    pub tcp: Vec<(u16, PathBuf)>,
    /// Guest Unix sockets, each relayed to a TCP port on 127.0.0.1.
    pub unix: Vec<(PathBuf, u16)>,
    /// Run `bash` or `sh` with the arguments after `--`.
    pub shell: bool,
    /// Put the clipboard stand-ins first on the client's `PATH`.
    pub clipboard: bool,
    /// The command and its arguments, or the shell's arguments.
    pub argv: Vec<OsString>,
}

fn parse_port(text: &OsStr) -> Result<u16, String> {
    text.to_str()
        .and_then(|t| t.parse::<u16>().ok())
        .filter(|p| *p > 0)
        .ok_or_else(|| format!("{} is not a port number", text.to_string_lossy()))
}

fn split_eq(value: &OsStr) -> Result<(&OsStr, &OsStr), String> {
    let bytes = value.as_bytes();
    match bytes.iter().position(|b| *b == b'=') {
        Some(i) => Ok((OsStr::from_bytes(&bytes[..i]), OsStr::from_bytes(&bytes[i + 1..]))),
        None => Err(format!("{} has no =", value.to_string_lossy())),
    }
}

/// Parses the arguments after the program name.
pub fn parse_args(args: &[OsString]) -> Result<Mode, String> {
    let mut spec = RunSpec::default();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_os_str();
        let value = || {
            args.get(i + 1)
                .map(|v| v.as_os_str())
                .ok_or(format!("{} needs a value", arg.to_string_lossy()))
        };
        match arg.to_str() {
            Some("--") => {
                spec.argv = args[i + 1..].to_vec();
                if spec.argv.is_empty() && !spec.shell {
                    return Err("no command after --".into());
                }
                return Ok(Mode::Run(spec));
            }
            Some("--check") => {
                let cmd = value()?;
                if args.len() != 2 {
                    return Err("--check takes only the command".into());
                }
                return Ok(Mode::Check(cmd.to_os_string()));
            }
            Some("--tcp") => {
                let (port, sock) = split_eq(value()?)?;
                spec.tcp.push((parse_port(port)?, PathBuf::from(sock)));
                i += 1;
            }
            Some("--unix") => {
                let (sock, port) = split_eq(value()?)?;
                spec.unix.push((PathBuf::from(sock), parse_port(port)?));
                i += 1;
            }
            Some("--shell") => spec.shell = true,
            Some("--clipboard") => spec.clipboard = true,
            _ => return Err(format!("unknown argument {}", arg.to_string_lossy())),
        }
        i += 1;
    }
    Err("missing -- before the command".into())
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else { return false };
    if !meta.is_file() || meta.permissions().mode() & 0o111 == 0 {
        return false;
    }
    let Ok(c_path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else { return false };
    // SAFETY: c_path is a valid NUL-terminated string for the call's duration.
    unsafe { libc::access(c_path.as_ptr(), libc::X_OK) == 0 }
}

/// Finds `cmd` as the image would run it: as given when it contains `/`,
/// otherwise in each `PATH` folder in turn, with an empty entry meaning the
/// current folder.
pub fn resolve(cmd: &OsStr, path_env: Option<&OsStr>) -> Option<PathBuf> {
    if cmd.is_empty() {
        return None;
    }
    if cmd.as_bytes().contains(&b'/') {
        let path = PathBuf::from(cmd);
        return is_executable_file(&path).then_some(path);
    }
    let search = path_env.unwrap_or(OsStr::new(DEFAULT_PATH));
    search.as_bytes().split(|b| *b == b':').find_map(|dir| {
        let dir = if dir.is_empty() { Path::new(".") } else { Path::new(OsStr::from_bytes(dir)) };
        let candidate = dir.join(cmd);
        is_executable_file(&candidate).then_some(candidate)
    })
}

fn not_found_message(cmd: &OsStr, path_env: Option<&OsStr>) -> String {
    let name = cmd.to_string_lossy();
    if cmd.as_bytes().contains(&b'/') {
        format!("gmlx-entry: {name} is not an executable file in this image. \
                 The image needs the command it runs.")
    } else {
        let search = path_env.map(|p| p.to_string_lossy()).unwrap_or(DEFAULT_PATH.into());
        format!("gmlx-entry: {name} is not on the image's PATH ({search}). Install it \
                 in the image, or set command: in the launch config.")
    }
}

/// The shell for `--shell`: `bash`, else `sh`.
pub fn resolve_shell(path_env: Option<&OsStr>) -> Option<PathBuf> {
    resolve(OsStr::new("bash"), path_env).or_else(|| resolve(OsStr::new("sh"), path_env))
}

fn fail(code: i32, message: &str) -> ! {
    eprintln!("{message}");
    exit(code)
}

/// The `PATH` the client gets under `--clipboard`: the stand-ins first,
/// then the image's own search path.
pub fn clipboard_path(path_env: Option<&OsStr>) -> OsString {
    let mut path = OsString::from(clipboard::CLIP_BIN);
    path.push(":");
    path.push(path_env.unwrap_or(OsStr::new(DEFAULT_PATH)));
    path
}

fn main() {
    let mut all = std::env::args_os();
    let arg0 = all.next().unwrap_or_default();
    let args: Vec<OsString> = all.collect();
    if let Some(tool) = clipboard::tool_name(&arg0) {
        exit(clipboard::run(tool, &args));
    }
    let mode = parse_args(&args)
        .unwrap_or_else(|e| fail(EXIT_USAGE, &format!("gmlx-entry: {e}\n{USAGE}")));
    let path_env = std::env::var_os("PATH");
    match mode {
        Mode::Check(cmd) => match resolve(&cmd, path_env.as_deref()) {
            Some(found) => println!("{}", found.display()),
            None => fail(EXIT_NOT_FOUND, &not_found_message(&cmd, path_env.as_deref())),
        },
        Mode::Run(spec) => run(spec, path_env),
    }
}

fn run(spec: RunSpec, path_env: Option<OsString>) -> ! {
    let (program, name, rest) = if spec.shell {
        let shell = resolve_shell(path_env.as_deref()).unwrap_or_else(|| {
            fail(EXIT_NOT_FOUND,
                 "gmlx-entry: the image has no shell (bash or sh), so --shell cannot open one.")
        });
        let name = shell.file_name().map(OsStr::to_os_string).unwrap_or_default();
        (shell, name, spec.argv)
    } else {
        let cmd = spec.argv[0].clone();
        let found = resolve(&cmd, path_env.as_deref())
            .unwrap_or_else(|| fail(EXIT_NOT_FOUND, &not_found_message(&cmd, path_env.as_deref())));
        (found, cmd, spec.argv[1..].to_vec())
    };

    if !spec.tcp.is_empty() || !spec.unix.is_empty() {
        let listeners = relay::bind_all(&spec.tcp, &spec.unix)
            .unwrap_or_else(|e| fail(EXIT_LISTEN, &format!("gmlx-entry: {e}")));
        relay::start_detached(listeners).unwrap_or_else(|e| {
            fail(EXIT_LISTEN, &format!("gmlx-entry: cannot start the relay: {e}"))
        });
    }

    // exec keeps only an ignored disposition, and Rust ignores SIGPIPE at
    // startup, so give the client the default back.
    // SAFETY: setting a signal disposition has no memory-safety preconditions.
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_DFL) };
    let mut command = Command::new(&program);
    command.arg0(&name).args(&rest);
    if spec.clipboard {
        command.env("PATH", clipboard_path(path_env.as_deref()));
    }
    let err = command.exec();
    let code = if err.kind() == std::io::ErrorKind::NotFound { EXIT_NOT_FOUND } else { EXIT_CANNOT_RUN };
    fail(code, &format!("gmlx-entry: cannot run {}: {err}", program.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn os(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    #[test]
    fn parses_listeners_and_command() {
        let mode = parse_args(&os(&["--tcp", "8080=/s/api.sock", "--unix", "/s/web.sock=3000",
                                    "--", "claude", "--continue"])).unwrap();
        assert_eq!(mode, Mode::Run(RunSpec {
            tcp: vec![(8080, PathBuf::from("/s/api.sock"))],
            unix: vec![(PathBuf::from("/s/web.sock"), 3000)],
            shell: false,
            clipboard: false,
            argv: os(&["claude", "--continue"]),
        }));
    }

    #[test]
    fn shell_takes_optional_args() {
        assert_eq!(parse_args(&os(&["--shell", "--"])).unwrap(),
                   Mode::Run(RunSpec { shell: true, ..Default::default() }));
        let Mode::Run(spec) = parse_args(&os(&["--shell", "--", "-c", "npm test"])).unwrap()
        else { panic!("not a run") };
        assert_eq!(spec.argv, os(&["-c", "npm test"]));
    }

    #[test]
    fn clipboard_flag_and_path() {
        let Mode::Run(spec) = parse_args(&os(&["--clipboard", "--", "claude"])).unwrap()
        else { panic!("not a run") };
        assert!(spec.clipboard);
        assert_eq!(clipboard_path(Some(OsStr::new("/usr/bin"))), "/opt/gmlx/bin:/usr/bin");
        assert_eq!(clipboard_path(None), format!("/opt/gmlx/bin:{DEFAULT_PATH}").as_str());
    }

    #[test]
    fn check_mode() {
        assert_eq!(parse_args(&os(&["--check", "claude"])).unwrap(),
                   Mode::Check(OsString::from("claude")));
        assert!(parse_args(&os(&["--check", "a", "b"])).is_err());
    }

    #[test]
    fn refuses_bad_arguments() {
        let cases: [&[&str]; 8] = [
            &["--"], &["claude"], &["--tcp", "x=/s", "--", "c"], &["--tcp", "0=/s", "--", "c"],
            &["--tcp", "80", "--", "c"], &["--unix", "/s=99999", "--", "c"],
            &["--bogus", "--", "c"], &["--tcp"],
        ];
        for bad in cases {
            assert!(parse_args(&os(bad)).is_err(), "{bad:?} parsed");
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("gmlx-entry-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make(path: &Path, mode: u32) {
        fs::write(path, "#!/bin/sh\n").unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
    }

    #[test]
    fn resolves_on_path_in_order() {
        let dir = scratch("path");
        let (a, b) = (dir.join("a"), dir.join("b"));
        fs::create_dir_all(&a).unwrap();
        fs::create_dir_all(&b).unwrap();
        make(&a.join("tool"), 0o644); // not executable: skipped
        make(&b.join("tool"), 0o755);
        let path = OsString::from(format!("{}:{}", a.display(), b.display()));
        assert_eq!(resolve(OsStr::new("tool"), Some(&path)), Some(b.join("tool")));
        assert_eq!(resolve(OsStr::new("missing"), Some(&path)), None);
        fs::create_dir_all(a.join("dir")).unwrap();
        assert_eq!(resolve(OsStr::new("dir"), Some(&path)), None); // folders never match
    }

    #[test]
    fn slash_commands_are_used_as_given() {
        let dir = scratch("slash");
        make(&dir.join("run"), 0o755);
        let given = dir.join("run");
        assert_eq!(resolve(given.as_os_str(), Some(OsStr::new(""))), Some(given.clone()));
        assert_eq!(resolve(dir.join("nope").as_os_str(), None), None);
    }

    #[test]
    fn default_path_when_unset() {
        assert_eq!(resolve(OsStr::new("sh"), None), Some(PathBuf::from("/bin/sh")));
    }

    #[test]
    fn shell_prefers_bash_then_sh() {
        let dir = scratch("shell");
        make(&dir.join("sh"), 0o755);
        let path = dir.clone().into_os_string();
        assert_eq!(resolve_shell(Some(&path)), Some(dir.join("sh")));
        make(&dir.join("bash"), 0o755);
        assert_eq!(resolve_shell(Some(&path)), Some(dir.join("bash")));
        let empty = scratch("noshell").into_os_string();
        assert_eq!(resolve_shell(Some(&empty)), None);
    }
}
