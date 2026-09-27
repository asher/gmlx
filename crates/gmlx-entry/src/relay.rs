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
        let mut status = 0;
        // SAFETY: waits for the intermediate child, which exits at once.
        unsafe { libc::waitpid(child, &mut status, 0) };
        return Ok(());
    }
    // SAFETY: plain system calls in the single-threaded child; _exit never
    // returns, so the intermediate child leaves without running destructors.
    unsafe {
        libc::setsid();
        if libc::fork() != 0 {
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

/// Runs every accept loop until all of them end.
pub fn serve(listeners: Vec<Listener>) {
    let mut loops = Vec::new();
    for listener in listeners {
        match listener {
            Listener::Tcp { port, sockets, target } => {
                for socket in sockets {
                    let target = target.clone();
                    loops.push(thread::spawn(move || accept_tcp(port, socket, target)));
                }
            }
            Listener::Unix { path, socket, port } => {
                loops.push(thread::spawn(move || accept_unix(path, socket, port)));
            }
        }
    }
    for handle in loops {
        let _ = handle.join();
    }
}

fn accept_tcp(port: u16, socket: TcpListener, target: PathBuf) {
    for conn in socket.incoming() {
        let Ok(conn) = conn else { continue };
        let target = target.clone();
        thread::spawn(move || match UnixStream::connect(&target) {
            Ok(up) => join(Conn::Tcp(conn), Conn::Unix(up)),
            Err(e) => log_line(&format!("port {port}: cannot reach {} ({e})", target.display())),
        });
    }
}

fn accept_unix(path: PathBuf, socket: UnixListener, port: u16) {
    for conn in socket.incoming() {
        let Ok(conn) = conn else { continue };
        let path = path.clone();
        thread::spawn(move || {
            let up = TcpStream::connect(("127.0.0.1", port))
                .or_else(|_| TcpStream::connect(("::1", port)));
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
    let back = thread::spawn(move || pump(up2, down2));
    pump(down, up);
    let _ = back.join();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufRead;

    fn scratch_sock(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gmlx-relay-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn free_port() -> u16 {
        TcpListener::bind(("127.0.0.1", 0)).unwrap().local_addr().unwrap().port()
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
        let port = free_port();
        let listeners = bind_all(&[(port, target)], &[]).unwrap();
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
        let port = free_port();
        let listeners = bind_all(&[(port, scratch_sock("missing.sock"))], &[]).unwrap();
        thread::spawn(move || serve(listeners));
        let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let mut got = Vec::new();
        c.read_to_end(&mut got).unwrap();
        assert!(got.is_empty());
    }
}
