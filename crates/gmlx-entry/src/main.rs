//! Guest entry for `gmlx launch --container`.
//!
//! `gmlx-entry [--tcp PORT=SOCK]... [--unix SOCK=PORT]... [--clipboard] [--shell] -- CMD ARGS`
//! binds the relay listeners, starts the relay as a detached process, links
//! root's `.ssh` to the private home's, runs the client and keeps the
//! session open until every joined copy has exited.
//! `gmlx-entry [--clipboard] --join [--copy-id ID] [--shell] -- CMD ARGS` runs
//! one more copy of a client in the running session, and
//! `gmlx-entry --hangup ID` sends that copy the SIGHUP a closed terminal
//! would send. `gmlx-entry --check CMD` only resolves CMD. Started as `xclip`, `xsel` or `wl-paste`, the binary is a
//! clipboard stand-in instead. The binary is static and needs nothing from
//! the image but the command it runs.

mod clipboard;
mod relay;
mod session;
mod ssh;

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
/// A copy cannot join the session, because it is ending.
pub const EXIT_ENDING: i32 = 75;

/// The search path when the image sets no `PATH`.
pub const DEFAULT_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";

const USAGE: &str = "usage: gmlx-entry [--tcp PORT=SOCK]... [--unix SOCK=PORT]... \
                     [--clipboard] [--shell] -- CMD [ARGS]...\n       \
                     gmlx-entry [--clipboard] --join [--copy-id ID] [--shell] -- CMD [ARGS]...\n       \
                     gmlx-entry --hangup ID\n       \
                     gmlx-entry --check CMD";

#[derive(Debug, PartialEq)]
pub enum Mode {
    Run(RunSpec),
    Check(OsString),
    /// Sends SIGHUP to the joined copy of this ID.
    Hangup(String),
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
    /// Run one more copy in the running session instead of starting it.
    pub join: bool,
    /// The ID that `--hangup` names this joined copy by.
    pub copy_id: Option<String>,
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

fn copy_id(text: &OsStr) -> Result<String, String> {
    text.to_str()
        .filter(|t| session::valid_copy_id(t))
        .map(str::to_owned)
        .ok_or_else(|| format!("{} is not a copy ID, which is 1 to 64 lowercase hex digits",
                               text.to_string_lossy()))
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
                if spec.join && !(spec.tcp.is_empty() && spec.unix.is_empty()) {
                    return Err("--join takes no listeners, since the session's relay \
                                already runs".into());
                }
                if spec.copy_id.is_some() && !spec.join {
                    return Err("--copy-id names a joined copy, so it needs --join".into());
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
            Some("--hangup") => {
                let id = copy_id(value()?)?;
                if args.len() != 2 {
                    return Err("--hangup takes only the copy ID".into());
                }
                return Ok(Mode::Hangup(id));
            }
            Some("--copy-id") => {
                spec.copy_id = Some(copy_id(value()?)?);
                i += 1;
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
            Some("--join") => spec.join = true,
            _ => return Err(format!("unknown argument {}", arg.to_string_lossy())),
        }
        i += 1;
    }
    Err("missing -- before the command".into())
}

/// What a lookup of a command found.
#[derive(Debug, PartialEq)]
pub enum Resolved {
    /// An executable file.
    Found(PathBuf),
    /// A regular file without the execute bit, and no executable match.
    NotExecutable(PathBuf),
    Missing,
}

/// Whether `path` is a regular file, and whether it can be executed.
fn file_state(path: &Path) -> Option<bool> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() {
        return None;
    }
    if meta.permissions().mode() & 0o111 == 0 {
        return Some(false);
    }
    let Ok(c_path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return Some(false);
    };
    // SAFETY: c_path is a valid NUL-terminated string for the call's duration.
    Some(unsafe { libc::access(c_path.as_ptr(), libc::X_OK) == 0 })
}

/// Finds `cmd` as execvp would: as given when it contains `/`, otherwise in
/// each `PATH` folder in turn, with an empty entry meaning the current
/// folder. A file without the execute bit is remembered and the search goes
/// on, so a later executable match still wins.
pub fn resolve_full(cmd: &OsStr, path_env: Option<&OsStr>) -> Resolved {
    if cmd.is_empty() {
        return Resolved::Missing;
    }
    if cmd.as_bytes().contains(&b'/') {
        let path = PathBuf::from(cmd);
        return match file_state(&path) {
            Some(true) => Resolved::Found(path),
            Some(false) => Resolved::NotExecutable(path),
            None => Resolved::Missing,
        };
    }
    let search = path_env.unwrap_or(OsStr::new(DEFAULT_PATH));
    let mut first_denied = None;
    for dir in search.as_bytes().split(|b| *b == b':') {
        let dir = if dir.is_empty() { Path::new(".") } else { Path::new(OsStr::from_bytes(dir)) };
        let candidate = dir.join(cmd);
        match file_state(&candidate) {
            Some(true) => return Resolved::Found(candidate),
            Some(false) if first_denied.is_none() => first_denied = Some(candidate),
            _ => {}
        }
    }
    first_denied.map_or(Resolved::Missing, Resolved::NotExecutable)
}

/// The executable match for `cmd`, if there is one.
pub fn resolve(cmd: &OsStr, path_env: Option<&OsStr>) -> Option<PathBuf> {
    match resolve_full(cmd, path_env) {
        Resolved::Found(path) => Some(path),
        _ => None,
    }
}

fn no_execute_bit_message(path: &Path) -> String {
    format!("[launch] {} has no execute bit. Run chmod 755 on it in the Containerfile.",
            shown(path.as_os_str()))
}

/// Exits 126 or 127 with the message for a lookup that found no executable.
fn fail_unresolved(resolved: Resolved, cmd: &OsStr, path_env: Option<&OsStr>) -> ! {
    match resolved {
        Resolved::NotExecutable(path) => fail(EXIT_CANNOT_RUN, &no_execute_bit_message(&path)),
        _ => fail(EXIT_NOT_FOUND, &not_found_message(cmd, path_env)),
    }
}

/// How many bytes of a file the Linux kernel reads for its `#!` line.
const SHEBANG_MAX: usize = 256;

/// The `env` options that take the next word as their value.
const ENV_VALUE_OPTIONS: [&str; 6] = ["-u", "--unset", "-C", "--chdir", "-P", "-a"];

/// Why the `#!` line of a found file stops it from running.
#[derive(Debug, PartialEq)]
pub enum Shebang {
    /// The interpreter, or the command `env` runs, is not in the image.
    Missing(OsString),
    /// `env`, the interpreter, receives words with blanks between them
    /// without `-S`, so it looks for one command with that whole name.
    OneName { env: OsString, name: OsString },
    /// The line ends in a carriage return, from Windows line endings.
    CarriageReturn,
}

/// What stops the `#!` line of `file` from running, read the way the Linux
/// kernel reads it. An absolute or relative interpreter must be an
/// executable file. For `env`, the command it runs is looked up on `PATH`
/// as env would. A file with no `#!` line, one that cannot be read, and a
/// line whose interpreter name the kernel cuts off give None, and exec
/// reports those.
pub fn shebang_problem(file: &Path, path_env: Option<&OsStr>) -> Option<Shebang> {
    use std::io::Read;
    // The kernel reads into a zeroed buffer, so a short file ends in NULs.
    let mut head = [0u8; SHEBANG_MAX];
    let mut f = std::fs::File::open(file).ok()?;
    let mut len = 0;
    while len < head.len() {
        match f.read(&mut head[len..]) {
            Ok(0) => break,
            Ok(n) => len += n,
            Err(_) => return None,
        }
    }
    let blank = |b: &u8| *b == b' ' || *b == b'\t';
    // The kernel looks for the newline only before the first NUL.
    let text = &head[..head.iter().position(|b| *b == 0).unwrap_or(SHEBANG_MAX)];
    let line = match text.iter().position(|b| *b == b'\n') {
        Some(end) => &head[..end],
        None => {
            // With no newline, Linux uses the first 255 bytes. It refuses
            // the file only when no blank or NUL follows the interpreter
            // name there, since the name may be cut off. Otherwise it runs
            // the interpreter with the argument cut short.
            let cut = &head[..SHEBANG_MAX - 1];
            let body = cut.strip_prefix(b"#!")?;
            let start = body.iter().position(|b| !blank(b))?;
            if !body[start..].iter().any(|b| blank(b) || *b == 0) {
                return None;
            }
            cut
        }
    };
    let line = line.strip_prefix(b"#!")?;
    // Linux reads the line as a C string, so a NUL byte ends it.
    let line = &line[..line.iter().position(|b| *b == 0).unwrap_or(line.len())];
    let trimmed = trim(line, blank);
    if trimmed.last() == Some(&b'\r') {
        return Some(Shebang::CarriageReturn);
    }
    // The kernel gives the interpreter the rest of the line as one
    // argument, with the blanks at its ends removed.
    let split = trimmed.iter().position(blank).unwrap_or(trimmed.len());
    let interpreter = OsStr::from_bytes(&trimmed[..split]);
    if interpreter.is_empty() {
        return None;
    }
    if file_state(Path::new(interpreter)) != Some(true) {
        return Some(Shebang::Missing(interpreter.to_os_string()));
    }
    if Path::new(interpreter).file_name() != Some(OsStr::new("env")) {
        return None;
    }
    let arg = trim(&trimmed[split..], blank);
    let words = if let Some(rest) = split_string_option(arg) {
        env_split(rest)
    } else if arg.is_empty() || arg.starts_with(b"-") || arg.contains(&b'=') {
        // Other options in one argument, or a setting with no command:
        // env decides, so there is nothing to look up here.
        return None;
    } else {
        // Without -S, env runs the whole argument as one command name.
        let cmd = OsStr::from_bytes(arg);
        return match resolve_full(cmd, path_env) {
            Resolved::Found(_) => None,
            _ if arg.iter().any(blank) => Some(Shebang::OneName {
                env: interpreter.to_os_string(), name: cmd.to_os_string() }),
            _ => Some(Shebang::Missing(cmd.to_os_string())),
        };
    };
    // The command is the first word that is not an option, an option's
    // value or a NAME=VALUE setting.
    let mut rest = words.iter();
    while let Some(word) = rest.next() {
        let text = OsStr::from_bytes(word).to_str().unwrap_or("");
        if ENV_VALUE_OPTIONS.contains(&text) {
            rest.next();
        } else if word.starts_with(b"-") || word.contains(&b'=') {
            continue;
        } else {
            let cmd = OsStr::from_bytes(word);
            return match resolve_full(cmd, path_env) {
                Resolved::Found(_) => None,
                _ => Some(Shebang::Missing(cmd.to_os_string())),
            };
        }
    }
    None
}

fn trim(bytes: &[u8], blank: impl Fn(&u8) -> bool) -> &[u8] {
    let start = bytes.iter().position(|b| !blank(b)).unwrap_or(bytes.len());
    let end = bytes.iter().rposition(|b| !blank(b)).map_or(start, |i| i + 1);
    &bytes[start..end]
}

/// The text after `-S` or `--split-string` when the argument starts with
/// one of them.
fn split_string_option(arg: &[u8]) -> Option<&[u8]> {
    for option in [&b"--split-string="[..], b"--split-string", b"-S"] {
        if let Some(rest) = arg.strip_prefix(option) {
            return Some(rest);
        }
    }
    None
}

/// Splits `env -S` text into words at blanks, removing single and double
/// quotes around parts of a word as env does for simple cases.
fn env_split(text: &[u8]) -> Vec<Vec<u8>> {
    let mut words = Vec::new();
    let mut word: Option<Vec<u8>> = None;
    let mut quote: Option<u8> = None;
    for &b in text {
        match quote {
            Some(q) if b == q => quote = None,
            Some(_) => word.get_or_insert_with(Vec::new).push(b),
            None if b == b'\'' || b == b'"' => {
                quote = Some(b);
                word.get_or_insert_with(Vec::new);
            }
            None if b == b' ' || b == b'\t' => words.extend(word.take()),
            None => word.get_or_insert_with(Vec::new).push(b),
        }
    }
    words.extend(word);
    words
}

/// A name as it may appear in a message, with control characters escaped
/// so they never reach the terminal.
fn shown(name: &OsStr) -> String {
    name.to_string_lossy()
        .chars()
        .map(|c| if c.is_control() { c.escape_default().to_string() } else { c.to_string() })
        .collect()
}

fn shebang_message(file: &Path, problem: &Shebang) -> String {
    match problem {
        Shebang::Missing(interpreter) => format!(
            "[launch] {} names {} in its #! line, which is not in the image.",
            shown(file.as_os_str()), shown(interpreter)),
        Shebang::OneName { env, name } => format!(
            "[launch] {} has \"{}\" after env in its #! line, and env receives it as one \
             command name. Write #!{} -S {} to pass it as separate words.",
            shown(file.as_os_str()), shown(name), shown(env), shown(name)),
        Shebang::CarriageReturn => format!(
            "[launch] {} has a #! line that ends in a carriage return, from Windows line \
             endings. Convert the file to Unix line endings.", shown(file.as_os_str())),
    }
}

/// Exits 126 when the `#!` line of `file` stops it from running.
fn check_interpreter(file: &Path, path_env: Option<&OsStr>) {
    if let Some(problem) = shebang_problem(file, path_env) {
        fail(EXIT_CANNOT_RUN, &shebang_message(file, &problem));
    }
}

fn not_found_message(cmd: &OsStr, path_env: Option<&OsStr>) -> String {
    let name = shown(cmd);
    if cmd.as_bytes().contains(&b'/') {
        format!("[launch] {name} is not an executable file in this image. \
                 The image needs the command it runs.")
    } else {
        let search = shown(path_env.unwrap_or(OsStr::new(DEFAULT_PATH)));
        format!("[launch] {name} is not on the image's PATH ({search}). Install it in \
                 the image, or set launch.container.clients.<client>.command.")
    }
}

/// The shell for `--shell`: `bash`, else `sh`. When neither can run, the
/// first one found without the execute bit is reported.
pub fn resolve_shell(path_env: Option<&OsStr>) -> Resolved {
    let bash = resolve_full(OsStr::new("bash"), path_env);
    if matches!(bash, Resolved::Found(_)) {
        return bash;
    }
    let sh = resolve_full(OsStr::new("sh"), path_env);
    match (bash, sh) {
        (_, Resolved::Found(path)) => Resolved::Found(path),
        (Resolved::NotExecutable(path), _) | (_, Resolved::NotExecutable(path)) => {
            Resolved::NotExecutable(path)
        }
        _ => Resolved::Missing,
    }
}

fn fail(code: i32, message: &str) -> ! {
    eprintln!("{message}");
    exit(code)
}

/// The `PATH` the client gets under `--clipboard`: the stand-in folder
/// `bin` first, then the image's own search path.
pub fn clipboard_path(bin: &OsStr, path_env: Option<&OsStr>) -> OsString {
    let mut path = bin.to_os_string();
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
        .unwrap_or_else(|e| fail(EXIT_USAGE, &format!("[launch] {e}\n{USAGE}")));
    let path_env = std::env::var_os("PATH");
    match mode {
        Mode::Check(cmd) => match resolve_full(&cmd, path_env.as_deref()) {
            Resolved::Found(found) => {
                check_interpreter(&found, path_env.as_deref());
                println!("{}", shown(found.as_os_str()))
            }
            other => fail_unresolved(other, &cmd, path_env.as_deref()),
        },
        Mode::Hangup(id) => exit(if session::hangup(&session::dir(), &id) { 0 } else { 1 }),
        Mode::Run(spec) => run(spec, path_env),
    }
}

fn run(spec: RunSpec, path_env: Option<OsString>) -> ! {
    let (program, name, rest) = if spec.shell {
        let shell = match resolve_shell(path_env.as_deref()) {
            Resolved::Found(path) => path,
            Resolved::NotExecutable(path) => {
                fail(EXIT_CANNOT_RUN, &no_execute_bit_message(&path))
            }
            Resolved::Missing => fail(EXIT_NOT_FOUND,
                "[launch] the image has no shell (bash or sh), so --shell cannot open one."),
        };
        let name = shell.file_name().map(OsStr::to_os_string).unwrap_or_default();
        (shell, name, spec.argv)
    } else {
        let cmd = spec.argv[0].clone();
        let found = match resolve_full(&cmd, path_env.as_deref()) {
            Resolved::Found(path) => path,
            other => fail_unresolved(other, &cmd, path_env.as_deref()),
        };
        (found, cmd, spec.argv[1..].to_vec())
    };
    check_interpreter(&program, path_env.as_deref());

    let dir = session::dir();
    // A copy joins before anything starts, so a session that is ending
    // refuses it at once.
    let joined = spec.join.then(|| {
        session::join(&dir, spec.copy_id.as_deref()).unwrap_or_else(|why| refuse_join(why))
    });
    if !spec.tcp.is_empty() || !spec.unix.is_empty() {
        let listeners = relay::bind_all(&spec.tcp, &spec.unix)
            .unwrap_or_else(|e| fail(EXIT_LISTEN, &format!("[launch] {e}")));
        relay::start_detached(listeners).unwrap_or_else(|e| {
            fail(EXIT_LISTEN, &format!("[launch] cannot start the connections to the Mac ({e})."))
        });
    }
    // The relay forks before the lock and the terminal are opened, so it
    // holds neither.
    let lock = if spec.join {
        None
    } else {
        match session::create(&dir) {
            Ok(lock) => Some(lock),
            Err(e) => {
                eprintln!("[launch] cannot create the session folder {} ({e}), so no other \
                           copy can join this session.", shown(dir.as_os_str()));
                None
            }
        }
    };
    if !spec.join {
        ssh::link_home(&ssh::passwd(), std::env::var_os("HOME"));
    }

    let mut command = Command::new(&program);
    command.arg0(&name).args(&rest).env_remove(session::DIR_ENV).env_remove(ssh::PASSWD_ENV);
    if spec.clipboard {
        command.env("PATH", clipboard_path(&clipboard::clip_bin(), path_env.as_deref()));
    }
    let tty = session::Tty::probe();
    session::block_signals();
    let pid = session::spawn(command, &tty).unwrap_or_else(|err| {
        if let Some(copy) = &joined {
            copy.leave();
        }
        exec_failed(&program, err, path_env.as_deref())
    });
    // A stopping container stops the joined copies too, and a copy whose
    // terminal closed kills its client when the SIGHUP does not end it.
    let armed = std::cell::Cell::new(false);
    let outcome = session::wait_client(pid, tty.own_group(), || {
        if !spec.join {
            session::stop_copies(&dir);
        } else if let Some(id) = spec.copy_id.as_deref() {
            if !armed.get() && session::hung_up(&dir, id) {
                armed.set(true);
                session::kill_after(pid, session::HANGUP_GRACE);
            }
        }
    });
    if let Some(terminal) = tty.foreground() {
        terminal.take_back();
    }
    if let Some(copy) = &joined {
        copy.leave();
        if session::ended(&dir) {
            session::say(&format!("[launch] the session ended in another terminal, so this \
                                   copy of {} stopped.", shown(&name)));
        }
        session::finish(outcome.code)
    }
    let Some(lock) = lock else { session::finish(outcome.code) };
    if let Some(since) = outcome.stopping {
        session::wait_for_stopped(dir, lock, outcome.code, since)
    }
    if session::try_end(&dir, &lock) {
        session::finish(outcome.code)
    }
    session::wait_for_copies(dir, lock, &name, outcome.code)
}

fn refuse_join(why: session::Refused) -> ! {
    match why {
        session::Refused::Ending => fail(EXIT_ENDING,
            "[launch] the session is ending, so this copy cannot join it. Launch again once \
             it has stopped."),
        // The terminal of this copy closed before the copy started, so
        // nothing is left to print on.
        session::Refused::HungUp => exit(128 + libc::SIGHUP),
        session::Refused::NoSession => fail(EXIT_ENDING, &format!(
            "[launch] this copy cannot join the session, because its session folder {} is \
             gone. End the session and launch again.", shown(session::dir().as_os_str()))),
    }
}

/// Exits 126 with the reason the found `program` did not start.
fn exec_failed(program: &Path, err: std::io::Error, path_env: Option<&OsStr>) -> ! {
    if err.kind() == std::io::ErrorKind::NotFound {
        // The file itself was found, so the missing file is the program
        // that runs it: a #! interpreter or the ELF program loader.
        if let Some(problem) = shebang_problem(program, path_env) {
            fail(EXIT_CANNOT_RUN, &shebang_message(program, &problem));
        }
        fail(EXIT_CANNOT_RUN, &format!(
            "[launch] cannot run {}, because its #! interpreter or its program loader is not \
             in the image.", shown(program.as_os_str())));
    }
    fail(EXIT_CANNOT_RUN, &format!("[launch] cannot run {} ({err}).",
                                   shown(program.as_os_str())))
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
            join: false,
            copy_id: None,
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
        let bin = OsStr::new(clipboard::CLIP_BIN);
        assert_eq!(clipboard_path(bin, Some(OsStr::new("/usr/bin"))), "/opt/gmlx/bin:/usr/bin");
        assert_eq!(clipboard_path(bin, None), format!("/opt/gmlx/bin:{DEFAULT_PATH}").as_str());
    }

    #[test]
    fn join_takes_the_shell_and_refuses_listeners() {
        let Mode::Run(spec) = parse_args(&os(&["--clipboard", "--join", "--shell", "--"])).unwrap()
        else { panic!("not a run") };
        assert!(spec.join && spec.shell && spec.clipboard && spec.argv.is_empty());
        assert!(parse_args(&os(&["--join", "--tcp", "80=/s", "--", "c"])).is_err());
        assert!(parse_args(&os(&["--join", "--"])).is_err());
    }

    #[test]
    fn a_copy_id_goes_with_join_and_names_a_hangup() {
        let Mode::Run(spec) = parse_args(&os(&["--join", "--copy-id", "0f3a", "--", "c"]))
            .unwrap()
        else { panic!("not a run") };
        assert_eq!(spec.copy_id.as_deref(), Some("0f3a"));
        assert!(parse_args(&os(&["--copy-id", "0f3a", "--", "c"])).is_err());
        for bad in ["", "0F3A", "../x", "g1", &"a".repeat(65)] {
            assert!(parse_args(&os(&["--join", "--copy-id", bad, "--", "c"])).is_err(), "{bad}");
            assert!(parse_args(&os(&["--hangup", bad])).is_err(), "{bad}");
        }
        assert_eq!(parse_args(&os(&["--hangup", "0f3a"])).unwrap(),
                   Mode::Hangup("0f3a".into()));
        assert!(parse_args(&os(&["--hangup", "0f3a", "--", "c"])).is_err());
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
        make(&a.join("tool"), 0o644); // not executable: remembered, the search goes on
        make(&b.join("tool"), 0o755);
        let path = OsString::from(format!("{}:{}", a.display(), b.display()));
        assert_eq!(resolve(OsStr::new("tool"), Some(&path)), Some(b.join("tool")));
        make(&a.join("only"), 0o644);
        assert_eq!(resolve_full(OsStr::new("only"), Some(&path)),
                   Resolved::NotExecutable(a.join("only")));
        assert_eq!(resolve(OsStr::new("only"), Some(&path)), None);
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
        assert_eq!(resolve_full(dir.join("nope").as_os_str(), None), Resolved::Missing);
        make(&dir.join("plain"), 0o644);
        assert_eq!(resolve_full(dir.join("plain").as_os_str(), None),
                   Resolved::NotExecutable(dir.join("plain")));
    }

    fn script(path: &Path, text: &str) -> PathBuf {
        fs::write(path, text).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
        path.to_path_buf()
    }

    #[test]
    fn finds_a_missing_shebang_interpreter() {
        let dir = scratch("shebang");
        let bin = dir.join("bin");
        fs::create_dir_all(&bin).unwrap();
        script(&bin.join("tool"), "#!/bin/sh\n");
        let path = OsString::from(format!("{}:/usr/bin:/bin", bin.display()));
        let p = Some(path.as_os_str());
        let missing = |name: &str| Some(Shebang::Missing(OsString::from(name)));
        let absolute = script(&dir.join("a"), "#!/nope/python3 -u\nprint(1)\n");
        assert_eq!(shebang_problem(&absolute, p), missing("/nope/python3"));
        let present = script(&dir.join("b"), "#! /bin/sh\necho hi\n");
        assert_eq!(shebang_problem(&present, p), None);
        let env_missing = script(&dir.join("c"), "#!/usr/bin/env missingtool\n");
        assert_eq!(shebang_problem(&env_missing, p), missing("missingtool"));
        // Without -S, env gets "tool --flag" as one command name.
        let env_one_arg = script(&dir.join("d"), "#!/usr/bin/env tool --flag\n");
        assert_eq!(shebang_problem(&env_one_arg, p), Some(Shebang::OneName {
            env: OsString::from("/usr/bin/env"), name: OsString::from("tool --flag") }));
        let env_found = script(&dir.join("d2"), "#!/usr/bin/env tool\n");
        assert_eq!(shebang_problem(&env_found, p), None);
        let env_split = script(&dir.join("e"), "#!/usr/bin/env -S -u HOME X=1 missingtool -x\n");
        assert_eq!(shebang_problem(&env_split, p), missing("missingtool"));
        let env_split_found = script(&dir.join("f"), "#!/usr/bin/env -S tool -x\n");
        assert_eq!(shebang_problem(&env_split_found, p), None);
        let env_long = script(&dir.join("f2"), "#!/usr/bin/env --split-string=tool -x\n");
        assert_eq!(shebang_problem(&env_long, p), None);
        let plain = script(&dir.join("g"), "echo no shebang\n");
        assert_eq!(shebang_problem(&plain, p), None);
        let bare_env = script(&dir.join("h"), "#!/usr/bin/env\n");
        assert_eq!(shebang_problem(&bare_env, p), None);
        let env_option = script(&dir.join("i"), "#!/usr/bin/env -i tool\n");
        assert_eq!(shebang_problem(&env_option, p), None);    // env decides
    }

    #[test]
    fn env_split_removes_simple_quotes() {
        assert_eq!(env_split(b" 'my tool' \"-x\" a'b'c "),
                   vec![b"my tool".to_vec(), b"-x".to_vec(), b"abc".to_vec()]);
        assert_eq!(env_split(b"''"), vec![Vec::<u8>::new()]);
        let dir = scratch("quoted");
        let bin = dir.join("bin");
        fs::create_dir_all(&bin).unwrap();
        script(&bin.join("tool"), "#!/bin/sh\n");
        let path = OsString::from(format!("{}:/usr/bin:/bin", bin.display()));
        let quoted = script(&dir.join("q"), "#!/usr/bin/env -S 'tool' -x\n");
        assert_eq!(shebang_problem(&quoted, Some(&path)), None);
        let quoted_missing = script(&dir.join("r"), "#!/usr/bin/env -S \"no tool\" -x\n");
        assert_eq!(shebang_problem(&quoted_missing, Some(&path)),
                   Some(Shebang::Missing(OsString::from("no tool"))));
    }

    #[test]
    fn reads_the_line_as_linux_does() {
        let dir = scratch("linux-line");
        let p = Some(OsStr::new("/usr/bin:/bin"));
        let crlf = script(&dir.join("crlf"), "#!/bin/sh\r\necho hi\r\n");
        assert_eq!(shebang_problem(&crlf, p), Some(Shebang::CarriageReturn));
        let crlf_arg = script(&dir.join("crlf-arg"), "#!/bin/sh -e\r\n");
        assert_eq!(shebang_problem(&crlf_arg, p), Some(Shebang::CarriageReturn));
        // A NUL byte ends the line.
        let nul = script(&dir.join("nul"), "#!/nope/x\0/bin/sh\n");
        assert_eq!(shebang_problem(&nul, p), Some(Shebang::Missing(OsString::from("/nope/x"))));
        // With no newline in the first 256 bytes, the kernel runs the
        // interpreter with the argument cut short.
        let long = format!("#!/nope/x {}\n", "a".repeat(300));
        let long = script(&dir.join("long"), &long);
        assert_eq!(shebang_problem(&long, p), Some(Shebang::Missing(OsString::from("/nope/x"))));
        // The kernel refuses the file when the interpreter name may be cut
        // off, so exec reports it.
        let long_name = format!("#!/{}\n", "a".repeat(300));
        let long_name = script(&dir.join("long-name"), &long_name);
        assert_eq!(shebang_problem(&long_name, p), None);
        let fits = format!("#!/nope/x {}\n", "a".repeat(200));
        let fits = script(&dir.join("fits"), &fits);
        assert_eq!(shebang_problem(&fits, p), Some(Shebang::Missing(OsString::from("/nope/x"))));
        // A short file with no newline still has its #! line read.
        let no_newline = script(&dir.join("eof"), "#!/nope/y");
        assert_eq!(shebang_problem(&no_newline, p),
                   Some(Shebang::Missing(OsString::from("/nope/y"))));
    }

    #[test]
    fn messages_escape_control_characters() {
        let message = shebang_message(Path::new("/s"),
                                      &Shebang::Missing(OsString::from("tool\x1b[31m")));
        assert_eq!(message, "[launch] /s names tool\\u{1b}[31m in its #! line, which is not \
                             in the image.");
        let one_name = Shebang::OneName { env: OsString::from("/usr/bin/env"),
                                          name: OsString::from("sh -x") };
        assert_eq!(shebang_message(Path::new("/s"), &one_name),
                   "[launch] /s has \"sh -x\" after env in its #! line, and env receives it \
                    as one command name. Write #!/usr/bin/env -S sh -x to pass it as separate \
                    words.");
        assert!(shebang_message(Path::new("/s"), &Shebang::CarriageReturn)
            .contains("ends in a carriage return, from Windows line endings"));
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
        assert_eq!(resolve_shell(Some(&path)), Resolved::Found(dir.join("sh")));
        make(&dir.join("bash"), 0o644); // not executable: sh still wins
        assert_eq!(resolve_shell(Some(&path)), Resolved::Found(dir.join("sh")));
        make(&dir.join("bash"), 0o755);
        assert_eq!(resolve_shell(Some(&path)), Resolved::Found(dir.join("bash")));
        let empty = scratch("noshell").into_os_string();
        assert_eq!(resolve_shell(Some(&empty)), Resolved::Missing);
        let denied = scratch("deniedshell");
        make(&denied.join("bash"), 0o644);
        make(&denied.join("sh"), 0o644);
        assert_eq!(resolve_shell(Some(denied.as_os_str())),
                   Resolved::NotExecutable(denied.join("bash")));
    }

    #[test]
    fn messages_escape_terminal_controls() {
        let evil = "/x\u{1b}]52;c;ZXZpbA==\u{7}\u{1b}[2A\u{9b}";
        let messages = [
            not_found_message(OsStr::new("tool"), Some(OsStr::new(evil))),
            not_found_message(OsStr::new(evil), None),
            no_execute_bit_message(Path::new(evil)),
        ];
        for m in &messages {
            assert!(!m.chars().any(|c| c.is_control()), "{m:?}");
            assert!(m.contains("\\u{1b}]52;c;ZXZpbA==\\u{7}\\u{1b}[2A\\u{9b}"), "{m:?}");
        }
    }
}
