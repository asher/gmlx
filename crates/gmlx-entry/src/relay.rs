//! The guest side of the socket relays. Each accepted connection gets a
//! connection to its target and two copy threads, one per direction, and an
//! end of file in one direction becomes a half-close of the other side.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::os::fd::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::Duration;

/// The relay stops logging once its log reaches this size.
const LOG_CAP: usize = 1 << 20;
const LOG_NAME: &str = ".gmlx-entry.log";

pub enum Listener {
    /// 127.0.0.1 and, when it binds, ::1, relayed to a Unix socket.
    Tcp { port: u16, sockets: Vec<TcpListener>, target: PathBuf },
    /// A Unix socket relayed to a TCP port on the loopback address.
    Unix { path: PathBuf, socket: UnixListener, port: u16 },
}

/// Binds every listener before the relay starts, so the client can connect
/// as soon as it runs.
pub fn bind_all(tcp: &[(u16, PathBuf)], unix: &[(PathBuf, u16)]) -> Result<Vec<Listener>, String> {
    let mut out = Vec::new();
    for (port, target) in tcp {
        let v4 = TcpListener::bind(("127.0.0.1", *port)).map_err(|e| {
            format!("cannot listen on 127.0.0.1:{port} ({e}). Something in the image \
                     already uses that port.")
        })?;
        let mut sockets = vec![v4];
        if let Ok(v6) = TcpListener::bind(("::1", *port)) {
            sockets.push(v6);
        }
        out.push(Listener::Tcp { port: *port, sockets, target: target.clone() });
    }
    for (path, port) in unix {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {} ({e})", parent.display()))?;
        }
        let _ = std::fs::remove_file(path);
        let socket = UnixListener::bind(path)
            .map_err(|e| format!("cannot listen on {} ({e})", path.display()))?;
        out.push(Listener::Unix { path: path.clone(), socket, port: *port });
    }
    Ok(out)
}

struct Log {
    file: Mutex<Option<File>>,
    written: AtomicUsize,
}

static LOG: OnceLock<Log> = OnceLock::new();

fn log_line(message: &str) {
    let Some(log) = LOG.get() else { return };
    let line = format!("{message}\n");
    if log.written.fetch_add(line.len(), Ordering::Relaxed) + line.len() > LOG_CAP {
        return;
    }
    if let Ok(mut guard) = log.file.lock() {
        if let Some(file) = guard.as_mut() {
            let _ = file.write_all(line.as_bytes());
        }
    }
}

/// Opens the relay log in `$HOME`, truncated for this session. Without a
/// writable home the relay logs nothing.
fn open_log() -> Option<File> {
    let home = std::env::var_os("HOME")?;
    OpenOptions::new().create(true).write(true).truncate(true)
        .open(Path::new(&home).join(LOG_NAME)).ok()
}

/// Starts the relay in a detached grandchild and returns in the parent, which
/// then execs the client. The relay runs in its own session with SIGINT and
/// SIGHUP ignored, so a Ctrl-C meant for the client never cuts the server
/// connection, and its output goes to the log instead of the client's screen.
pub fn start_detached(listeners: Vec<Listener>) -> io::Result<()> {
    // SAFETY: no thread has started yet, so fork duplicates only this thread.
    let child = unsafe { libc::fork() };
    if child < 0 {
        return Err(io::Error::last_os_error());
    }
    if child > 0 {
        return wait_intermediate(child);
    }
    // SAFETY: plain system calls in the single-threaded child; _exit never
    // returns, so the intermediate child leaves without running destructors.
    // It exits 1 when the relay's own fork fails, so the parent reports it.
    unsafe {
        libc::setsid();
        let grandchild = libc::fork();
        if grandchild < 0 {
            libc::_exit(1);
        }
        if grandchild > 0 {
            libc::_exit(0);
        }
        libc::signal(libc::SIGINT, libc::SIG_IGN);
        libc::signal(libc::SIGHUP, libc::SIG_IGN);
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
    }
    let log = open_log();
    redirect_stdio(log.as_ref());
    let _ = LOG.set(Log { file: Mutex::new(log), written: AtomicUsize::new(0) });
    serve(listeners);
    // SAFETY: the relay ends only when every accept loop has failed.
    unsafe { libc::_exit(0) }
}

/// Waits for the intermediate child, which exits at once: 0 when the relay
/// started, 1 when its fork failed.
fn wait_intermediate(child: libc::pid_t) -> io::Result<()> {
    let mut status = 0;
    loop {
        // SAFETY: waits for our own child; status is a valid out-pointer.
        if unsafe { libc::waitpid(child, &mut status, 0) } >= 0 {
            break;
        }
        let err = io::Error::last_os_error();
        if err.kind() != io::ErrorKind::Interrupted {
            return Err(err);
        }
    }
    if libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0 {
        Ok(())
    } else {
        Err(io::Error::other("the relay process did not start"))
    }
}

fn redirect_stdio(log: Option<&File>) {
    if let Ok(null) = File::open("/dev/null") {
        // SAFETY: dup2 onto the standard descriptors of this process.
        unsafe { libc::dup2(null.as_raw_fd(), 0) };
    }
    let out = log.map(|f| f.as_raw_fd())
        .or_else(|| OpenOptions::new().write(true).open("/dev/null").ok().map(|f| {
            let fd = f.as_raw_fd();
            std::mem::forget(f);
            fd
        }));
    if let Some(fd) = out {
        // SAFETY: as above.
        unsafe {
            libc::dup2(fd, 1);
            libc::dup2(fd, 2);
        }
    }
}

/// Starts a thread, or returns false when the system cannot start one. The
/// release profile aborts on a panic, so a failed `thread::spawn` would end
/// every relayed connection.
fn spawn<F: FnOnce() + Send + 'static>(f: F) -> Option<thread::JoinHandle<()>> {
    match thread::Builder::new().spawn(f) {
        Ok(handle) => Some(handle),
        Err(e) => {
            log_line(&format!("cannot start a thread ({e})"));
            None
        }
    }
}

/// Runs every accept loop until all of them end.
pub fn serve(listeners: Vec<Listener>) {
    let mut loops = Vec::new();
    for listener in listeners {
        match listener {
            Listener::Tcp { port, sockets, target } => {
                for socket in sockets {
                    let target = target.clone();
                    loops.extend(spawn(move || accept_tcp(port, socket, target)));
                }
            }
            Listener::Unix { path, socket, port } => {
                loops.extend(spawn(move || accept_unix(path, socket, port)));
            }
        }
    }
    for handle in loops {
        let _ = handle.join();
    }
}

/// Handles an accept error. One that lasts, such as running out of file
/// descriptors, is logged once per run of failures and pauses the loop for
/// 100 ms, so the loop does not spin.
fn accept_failed(what: &str, e: &io::Error, logged: &mut bool) {
    let lasting = matches!(e.raw_os_error(),
        Some(libc::EMFILE | libc::ENFILE | libc::ENOBUFS | libc::ENOMEM));
    if !lasting {
        return;
    }
    if !*logged {
        log_line(&format!("{what}: cannot accept a connection ({e})"));
        *logged = true;
    }
    thread::sleep(Duration::from_millis(100));
}

fn accept_tcp(port: u16, socket: TcpListener, target: PathBuf) {
    let mut logged = false;
    for conn in socket.incoming() {
        let conn = match conn {
            Ok(conn) => conn,
            Err(e) => {
                accept_failed(&format!("port {port}"), &e, &mut logged);
                continue;
            }
        };
        logged = false;
        let target = target.clone();
        spawn(move || match UnixStream::connect(&target) {
            Ok(up) => join(Conn::Tcp(conn), Conn::Unix(up)),
            Err(e) => log_line(&format!("port {port}: cannot reach {} ({e})", target.display())),
        });
    }
}

fn accept_unix(path: PathBuf, socket: UnixListener, port: u16) {
    let mut logged = false;
    for conn in socket.incoming() {
        let conn = match conn {
            Ok(conn) => conn,
            Err(e) => {
                accept_failed(&path.display().to_string(), &e, &mut logged);
                continue;
            }
        };
        logged = false;
        let path = path.clone();
        spawn(move || {
            // The first error names 127.0.0.1, the address the message gives.
            let up = TcpStream::connect(("127.0.0.1", port))
                .or_else(|e| TcpStream::connect(("::1", port)).map_err(|_| e));
            match up {
                Ok(up) => join(Conn::Unix(conn), Conn::Tcp(up)),
                Err(e) => log_line(&format!(
                    "{}: nothing listens on 127.0.0.1:{port} yet ({e})", path.display())),
            }
        });
    }
}

enum Conn {
    Tcp(TcpStream),
    Unix(UnixStream),
}

impl Conn {
    fn try_clone(&self) -> io::Result<Conn> {
        Ok(match self {
            Conn::Tcp(s) => Conn::Tcp(s.try_clone()?),
            Conn::Unix(s) => Conn::Unix(s.try_clone()?),
        })
    }

    fn shutdown(&self, how: Shutdown) {
        let _ = match self {
            Conn::Tcp(s) => s.shutdown(how),
            Conn::Unix(s) => s.shutdown(how),
        };
    }
}

impl Read for Conn {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Conn::Tcp(s) => s.read(buf),
            Conn::Unix(s) => s.read(buf),
        }
    }
}

impl Write for Conn {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Conn::Tcp(s) => s.write(buf),
            Conn::Unix(s) => s.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Copies `from` to `to` until end of file, then half-closes `to`. An error
/// closes both sides, so the other direction ends too.
fn pump(mut from: Conn, mut to: Conn) {
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        match from.read(&mut buf) {
            Ok(0) => {
                to.shutdown(Shutdown::Write);
                return;
            }
            Ok(n) => {
                if to.write_all(&buf[..n]).is_err() {
                    break;
                }
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
    from.shutdown(Shutdown::Both);
    to.shutdown(Shutdown::Both);
}

/// Relays one accepted connection and its upstream in both directions.
fn join(down: Conn, up: Conn) {
    let (Ok(down2), Ok(up2)) = (down.try_clone(), up.try_clone()) else {
        log_line("cannot duplicate a connection");
        return;
    };
    let Some(back) = spawn(move || pump(up2, down2)) else {
        return;
    };
    pump(down, up);
    let _ = back.join();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufRead;

    fn exit_child(code: i32) -> libc::pid_t {
        // SAFETY: the child only calls _exit, which is async-signal-safe.
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            unsafe { libc::_exit(code) };
        }
        pid
    }

    #[test]
    fn a_failed_relay_fork_is_an_error() {
        let _forks = crate::session::test_forks().read().unwrap_or_else(|e| e.into_inner());
        assert!(wait_intermediate(exit_child(0)).is_ok());
        assert!(wait_intermediate(exit_child(1)).is_err());
    }

    fn scratch_sock(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gmlx-relay-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// A TCP listener already bound to a port the system picked, so tests
    /// running in parallel threads never race for the same port.
    fn tcp_listener(target: PathBuf) -> (u16, Vec<Listener>) {
        let socket = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = socket.local_addr().unwrap().port();
        (port, vec![Listener::Tcp { port, sockets: vec![socket], target }])
    }

    /// An echo server that answers each line and half-closes after EOF.
    fn echo_unix(path: &Path) {
        let _ = std::fs::remove_file(path);
        let server = UnixListener::bind(path).unwrap();
        thread::spawn(move || {
            for conn in server.incoming().flatten() {
                thread::spawn(move || {
                    let mut out = conn.try_clone().unwrap();
                    for line in io::BufReader::new(conn).lines().map_while(Result::ok) {
                        out.write_all(format!("echo {line}\n").as_bytes()).unwrap();
                    }
                    out.write_all(b"bye\n").unwrap();
                });
            }
        });
    }

    #[test]
    fn tcp_to_unix_with_half_close() {
        let target = scratch_sock("echo-a.sock");
        echo_unix(&target);
        let (port, listeners) = tcp_listener(target);
        thread::spawn(move || serve(listeners));
        let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
        c.write_all(b"one\ntwo\n").unwrap();
        c.shutdown(Shutdown::Write).unwrap(); // the server sees EOF and still answers
        let mut got = String::new();
        c.read_to_string(&mut got).unwrap();
        assert_eq!(got, "echo one\necho two\nbye\n");
    }

    #[test]
    fn unix_to_tcp_creates_the_parent_folder() {
        let server = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = server.local_addr().unwrap().port();
        thread::spawn(move || {
            for mut conn in server.incoming().flatten() {
                let mut buf = [0u8; 5];
                conn.read_exact(&mut buf).unwrap();
                conn.write_all(&buf).unwrap();
            }
        });
        let path = scratch_sock("nested/dir/web.sock");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        let listeners = bind_all(&[], &[(path.clone(), port)]).unwrap();
        thread::spawn(move || serve(listeners));
        let mut c = UnixStream::connect(&path).unwrap();
        c.write_all(b"hello").unwrap();
        let mut buf = [0u8; 5];
        c.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"hello");
    }

    #[test]
    fn busy_port_is_an_error() {
        let held = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = held.local_addr().unwrap().port();
        let err = bind_all(&[(port, PathBuf::from("/nope"))], &[]).err().unwrap();
        assert!(err.contains(&format!("127.0.0.1:{port}")), "{err}");
    }

    #[test]
    fn unreachable_target_closes_the_client() {
        let (port, listeners) = tcp_listener(scratch_sock("missing.sock"));
        thread::spawn(move || serve(listeners));
        let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let mut got = Vec::new();
        c.read_to_end(&mut got).unwrap();
        assert!(got.is_empty());
    }
}
