//! The session a main entry keeps open for the copies that join it.
//!
//! The main entry runs the session's client, and each `--join` runs one more
//! copy of a client in the same container. A joined copy holds a shared
//! `flock` on the copies lock while it runs. When its own client exits, the
//! main entry takes that lock exclusively, so it waits until the last copy
//! has exited, and the container stops only then. Each copy also writes a
//! file named by its process ID, which holds the copy's ID. The count in
//! the main entry's message uses these files, and so does a hangup, which
//! finds a copy by its ID.
//!
//! The client runs in a process group of its own, which gets the terminal
//! when the entry has it, and the entry passes SIGTERM, SIGHUP, SIGINT and
//! SIGQUIT on to that group. When another group has the terminal in the
//! foreground, the client stays in the entry's group, as after an exec.
//!
//! The guest's init does no job control, so nothing else can resume a
//! stopped client. The entry resumes a client that Ctrl-Z or SIGSTOP stops,
//! and the job-control signals never stop the entry itself.

use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::mem::MaybeUninit;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The folder of the copies lock and the copy files. The guest runs as
/// root, and the folder is readable only by its owner. Launch refuses a
/// share at this path, so the folder is always the container's own.
pub const DIR: &str = "/run/gmlx-session";
/// Names another folder, for the tests that run the entry outside a guest.
/// The client never sees it.
pub const DIR_ENV: &str = "GMLX_ENTRY_SESSION_DIR";
const LOCK: &str = "copies.lock";
/// Written by the main entry when it ends, so a copy that takes the shared
/// lock after the main entry has exited still refuses to join.
const ENDED: &str = "ended";
const COPY_PREFIX: &str = "copy-";
/// Written by a hangup, so a copy that joins after its hangup stops at once.
const HANGUP_PREFIX: &str = "hangup-";

/// The signals the entry passes on to the client's process group.
pub const FORWARDED: [libc::c_int; 4] = [libc::SIGTERM, libc::SIGHUP, libc::SIGINT, libc::SIGQUIT];

/// The job-control signals. The entry waits for them too, so they never
/// stop it, and it passes none of them on.
const JOB_CONTROL: [libc::c_int; 3] = [libc::SIGTSTP, libc::SIGTTIN, libc::SIGTTOU];

/// How long the main entry waits for the joined copies after it ends the
/// session, counted from the signal that ended it. Launch gives
/// `container stop` 10 seconds, so the copies can exit before the container
/// is killed.
pub const COPY_GRACE: Duration = Duration::from_secs(5);

/// How long a joined copy whose terminal closed has after its SIGHUP before
/// the entry kills its client. A client whose output nobody reads any more
/// can block in a write and never act on the SIGHUP.
pub const HANGUP_GRACE: Duration = Duration::from_secs(10);

/// The session folder: [`DIR`], or the folder [`DIR_ENV`] names.
pub fn dir() -> PathBuf {
    std::env::var_os(DIR_ENV)
        .filter(|d| !d.is_empty())
        .map_or_else(|| PathBuf::from(DIR), PathBuf::from)
}

fn flock(file: &File, op: libc::c_int) -> io::Result<()> {
    loop {
        // SAFETY: the descriptor stays open for the duration of the call.
        if unsafe { libc::flock(file.as_raw_fd(), op) } == 0 {
            return Ok(());
        }
        let err = io::Error::last_os_error();
        if err.kind() != io::ErrorKind::Interrupted {
            return Err(err);
        }
    }
}

fn open_lock(dir: &Path, create: bool) -> io::Result<File> {
    // std opens every file close-on-exec, so no client inherits the lock.
    OpenOptions::new()
        .read(true)
        .write(create)
        .create(create)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(dir.join(LOCK))
}

/// Creates the session folder with mode 0700 and opens its copies lock.
/// A folder that is already there must be a folder of this user and not a
/// link, and what an earlier main entry left in it is removed. The parent
/// folder is made when the image has none.
pub fn create(dir: &Path) -> io::Result<File> {
    if let Some(parent) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::DirBuilder::new().recursive(true).mode(0o755).create(parent)?;
    }
    match fs::DirBuilder::new().mode(0o700).create(dir) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
            let meta = fs::symlink_metadata(dir)?;
            // SAFETY: geteuid has no preconditions.
            if !meta.is_dir() || meta.uid() != unsafe { libc::geteuid() } {
                return Err(io::Error::other(format!(
                    "{} is not a folder of this user", dir.display())));
            }
            fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
            for entry in fs::read_dir(dir)? {
                let name = entry?.file_name();
                let bytes = name.as_bytes();
                if name == ENDED || bytes.starts_with(COPY_PREFIX.as_bytes())
                    || bytes.starts_with(HANGUP_PREFIX.as_bytes())
                {
                    let _ = fs::remove_file(dir.join(&name));
                }
            }
        }
        Err(e) => return Err(e),
    }
    open_lock(dir, true)
}

/// Marks the session ended, so no copy joins it from now on.
pub fn mark_ended(dir: &Path) {
    let _ = OpenOptions::new()
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(dir.join(ENDED));
}

/// Takes the copies lock exclusively when no copy holds it, and then marks
/// the session ended. Returns false while a copy runs. A lock that fails
/// for another reason ends the session too, so the main entry never hangs.
pub fn try_end(dir: &Path, lock: &File) -> bool {
    match flock(lock, libc::LOCK_EX | libc::LOCK_NB) {
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => false,
        _ => {
            mark_ended(dir);
            true
        }
    }
}

/// Waits until the last copy has exited, and then marks the session ended.
pub fn wait_end(dir: &Path, lock: &File) {
    let _ = flock(lock, libc::LOCK_EX);
    mark_ended(dir);
}

/// Whether the main entry has marked the session ended.
pub fn ended(dir: &Path) -> bool {
    fs::symlink_metadata(dir.join(ENDED)).is_ok()
}

/// Marks the session ended, so that a copy can tell why it stopped, and
/// then sends SIGHUP to each copy, as a closed terminal would. Returns how
/// many copies there were.
pub fn stop_copies(dir: &Path) -> usize {
    mark_ended(dir);
    let left = copies(dir);
    for &pid in &left {
        // SAFETY: kill has no memory-safety preconditions.
        unsafe { libc::kill(pid, libc::SIGHUP) };
    }
    left.len()
}

fn alive(pid: libc::pid_t) -> bool {
    // SAFETY: signal 0 only checks that the process exists.
    let sent = unsafe { libc::kill(pid, 0) } == 0;
    sent || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// The process IDs of the joined copies that still run. The file of a copy
/// whose process is gone, such as one that was killed, is removed.
pub fn copies(dir: &Path) -> Vec<libc::pid_t> {
    let Ok(entries) = fs::read_dir(dir) else { return Vec::new() };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name.to_str()
            .and_then(|n| n.strip_prefix(COPY_PREFIX))
            .and_then(|p| p.parse::<libc::pid_t>().ok())
        else {
            continue;
        };
        if pid > 0 && alive(pid) {
            out.push(pid);
        } else {
            let _ = fs::remove_file(entry.path());
        }
    }
    out
}

/// Whether `id` can name a copy: 1 to 64 lowercase hex digits, so it is
/// safe in a file name.
pub fn valid_copy_id(id: &str) -> bool {
    (1..=64).contains(&id.len()) && id.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Sends SIGHUP to the joined copy named `id`, as a closed terminal would,
/// and returns whether one runs. The mark it writes first stops a copy that
/// joins after this call, since a copy checks for the mark after it writes
/// its file.
pub fn hangup(dir: &Path, id: &str) -> bool {
    let _ = OpenOptions::new()
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(dir.join(format!("{HANGUP_PREFIX}{id}")));
    let mut found = false;
    for pid in copies(dir) {
        let file = dir.join(format!("{COPY_PREFIX}{pid}"));
        let named = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&file)
            .and_then(|mut f| {
                let mut text = String::new();
                io::Read::read_to_string(&mut f, &mut text).map(|_| text)
            });
        if named.is_ok_and(|text| text == id) {
            // SAFETY: kill has no memory-safety preconditions.
            unsafe { libc::kill(pid, libc::SIGHUP) };
            found = true;
        }
    }
    found
}

/// Whether a hangup for the copy `id` has come.
pub fn hung_up(dir: &Path, id: &str) -> bool {
    fs::symlink_metadata(dir.join(format!("{HANGUP_PREFIX}{id}"))).is_ok()
}

/// Kills the process group `group` after `delay`, unless the entry exits
/// first.
pub fn kill_after(group: libc::pid_t, delay: Duration) {
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        forward(group, libc::SIGKILL)
    });
}

/// Why a copy cannot join the session.
#[derive(Debug, PartialEq)]
pub enum Refused {
    /// The main entry is ending the session.
    Ending,
    /// The container has no session folder, so its main entry takes no
    /// copies.
    NoSession,
    /// A hangup for this copy came before it joined.
    HungUp,
}

/// A joined copy: the shared lock, and the file that counts it.
pub struct Joined {
    _lock: File,
    pid_file: PathBuf,
}

impl Joined {
    /// Removes the copy's file. The lock goes when the process exits.
    pub fn leave(&self) {
        let _ = fs::remove_file(&self.pid_file);
    }
}

/// Joins the session in `dir` without waiting, as the copy named `id`.
/// The copy writes its file before it checks the end mark and the hangup
/// mark, the reverse of the order [`stop_copies`] and [`hangup`] use, so
/// each copy either sees the mark or gets the signal.
pub fn join(dir: &Path, id: Option<&str>) -> Result<Joined, Refused> {
    let lock = open_lock(dir, false).map_err(|_| Refused::NoSession)?;
    match flock(&lock, libc::LOCK_SH | libc::LOCK_NB) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Err(Refused::Ending),
        Err(_) => return Err(Refused::NoSession),
    }
    let pid_file = dir.join(format!("{COPY_PREFIX}{}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&pid_file)
        .map_err(|_| Refused::NoSession)?;
    let joined = Joined { _lock: lock, pid_file };
    if let Some(id) = id {
        if io::Write::write_all(&mut file, id.as_bytes()).is_err() {
            joined.leave();
            return Err(Refused::NoSession);
        }
    }
    if ended(dir) {
        joined.leave();
        return Err(Refused::Ending);
    }
    if let Some(id) = id {
        let mark = dir.join(format!("{HANGUP_PREFIX}{id}"));
        if fs::symlink_metadata(&mark).is_ok() {
            joined.leave();
            let _ = fs::remove_file(mark);
            return Err(Refused::HungUp);
        }
    }
    Ok(joined)
}

fn sigset(signals: &[libc::c_int]) -> libc::sigset_t {
    let mut set = MaybeUninit::<libc::sigset_t>::uninit();
    // SAFETY: sigemptyset initializes the set before sigaddset reads it.
    unsafe {
        libc::sigemptyset(set.as_mut_ptr());
        for &sig in signals {
            libc::sigaddset(set.as_mut_ptr(), sig);
        }
        set.assume_init()
    }
}

/// The signals the entry waits for: the forwarded ones, the job-control
/// ones and SIGCHLD.
fn waited_list() -> Vec<libc::c_int> {
    let mut all = FORWARDED.to_vec();
    all.extend(JOB_CONTROL);
    all.push(libc::SIGCHLD);
    all
}

fn waited() -> libc::sigset_t {
    sigset(&waited_list())
}

/// Whether `sig` is a job-control signal, which the entry ignores.
pub fn job_control(sig: libc::c_int) -> bool {
    JOB_CONTROL.contains(&sig)
}

extern "C" fn ignore(_: libc::c_int) {}

/// Makes the waited signals wait for [`next_signal`]. A handler, never
/// SIG_IGN, is installed for each first, since a signal whose action is to
/// ignore it can be dropped even while it is blocked. exec resets a handler,
/// so the client starts with the default actions.
pub fn block_signals() {
    for sig in waited_list() {
        // SAFETY: the handler does nothing, and it never runs while blocked.
        unsafe { libc::signal(sig, ignore as extern "C" fn(libc::c_int) as libc::sighandler_t) };
    }
    let set = waited();
    // SAFETY: set is initialized; the old mask is not needed.
    unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, &set, ptr::null_mut()) };
}

/// The next waited signal.
pub fn next_signal() -> libc::c_int {
    let set = waited();
    let mut sig = 0;
    loop {
        // SAFETY: set and sig are valid for the call.
        if unsafe { libc::sigwait(&set, &mut sig) } == 0 {
            return sig;
        }
    }
}

/// The exit code of a wait status: the process's own, or 128 plus the
/// signal that ended it, as a shell reports it.
pub fn exit_code(status: libc::c_int) -> i32 {
    if libc::WIFEXITED(status) {
        libc::WEXITSTATUS(status)
    } else if libc::WIFSIGNALED(status) {
        128 + libc::WTERMSIG(status)
    } else {
        1
    }
}

static RESUMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Reaps every child that has exited, and returns the exit code of `pid`
/// when it is one of them. A child that Ctrl-Z or SIGSTOP stopped is
/// resumed, and the first time this happens the entry says why. A child
/// stopped by a terminal read or write from the background stays stopped,
/// since it would stop again at once, and the shell that owns the terminal
/// resumes it.
pub fn reap(pid: libc::pid_t) -> Option<i32> {
    let mut found = None;
    loop {
        let mut status = 0;
        // SAFETY: status is valid for the call.
        let got = unsafe { libc::waitpid(-1, &mut status, libc::WNOHANG | libc::WUNTRACED) };
        if got <= 0 {
            return found;
        }
        if libc::WIFSTOPPED(status) {
            if matches!(libc::WSTOPSIG(status), libc::SIGTSTP | libc::SIGSTOP) {
                if !RESUMED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    say("[launch] Ctrl-Z cannot suspend a client in the container, so it goes \
                         on running.");
                }
                resume(got);
            }
            continue;
        }
        if got == pid {
            found = Some(exit_code(status));
        }
    }
}

/// Sends SIGCONT to the process group of `pid`, so the processes that
/// stopped with it go on too.
fn resume(pid: libc::pid_t) {
    // SAFETY: getpgid and kill have no memory-safety preconditions.
    unsafe {
        let group = libc::getpgid(pid);
        if group <= 0 || libc::kill(-group, libc::SIGCONT) != 0 {
            libc::kill(pid, libc::SIGCONT);
        }
    }
}

/// Sends `sig` to the process group `group`, or to the process of that ID
/// when no such group exists, such as a client that stays in the entry's
/// group.
pub fn forward(group: libc::pid_t, sig: libc::c_int) {
    // SAFETY: kill has no memory-safety preconditions.
    unsafe {
        if libc::kill(-group, sig) != 0 {
            libc::kill(group, sig);
        }
    }
}

/// Runs `f` with SIGTTOU blocked, so a terminal call from a background
/// process group does not stop the caller.
fn without_ttou(f: impl FnOnce()) {
    let ttou = sigset(&[libc::SIGTTOU]);
    let mut old = MaybeUninit::<libc::sigset_t>::uninit();
    // SAFETY: both sets are valid; old is written before it is read.
    unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, &ttou, old.as_mut_ptr()) };
    f();
    // SAFETY: old was written by the call above.
    unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, old.as_ptr(), ptr::null_mut()) };
}

/// The controlling terminal, when this process's group has it in the
/// foreground.
pub struct Terminal {
    file: File,
    group: libc::pid_t,
    modes: Option<libc::termios>,
}

/// The entry's controlling terminal, if it has one.
pub enum Tty {
    None,
    /// Another process group has the terminal in the foreground.
    Background,
    Foreground(Terminal),
}

impl Tty {
    pub fn probe() -> Tty {
        let Ok(file) = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOCTTY)
            .open("/dev/tty")
        else {
            return Tty::None;
        };
        let fd = file.as_raw_fd();
        // SAFETY: fd is open; getpgrp has no preconditions.
        let (group, own) = unsafe { (libc::tcgetpgrp(fd), libc::getpgrp()) };
        if group < 0 || group != own {
            return Tty::Background;
        }
        let mut modes = MaybeUninit::<libc::termios>::uninit();
        // SAFETY: tcgetattr fills modes when it returns 0.
        let modes = (unsafe { libc::tcgetattr(fd, modes.as_mut_ptr()) } == 0)
            .then(|| unsafe { modes.assume_init() });
        Tty::Foreground(Terminal { file, group, modes })
    }

    pub fn foreground(&self) -> Option<&Terminal> {
        match self {
            Tty::Foreground(terminal) => Some(terminal),
            _ => None,
        }
    }

    /// Whether the client gets a process group of its own. A client in the
    /// background of a terminal would stop at its first read, so it stays
    /// in the entry's group then, as it would after an exec.
    pub fn own_group(&self) -> bool {
        !matches!(self, Tty::Background)
    }
}

impl Terminal {
    fn fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }

    /// Gives the terminal back to the entry's group with the modes it had
    /// when the entry started, since a client that was killed can leave it
    /// in raw mode.
    pub fn take_back(&self) {
        let fd = self.fd();
        without_ttou(|| {
            // SAFETY: fd is open, and modes came from tcgetattr.
            unsafe {
                libc::tcsetpgrp(fd, self.group);
                if let Some(modes) = &self.modes {
                    libc::tcsetattr(fd, libc::TCSADRAIN, modes);
                }
            }
        });
    }
}

/// Starts the client, in a process group of its own when `tty` allows one,
/// which becomes the terminal's foreground group when the entry has the
/// terminal. The client starts with no signal blocked and SIGPIPE at its
/// default action.
pub fn spawn(mut command: Command, tty: &Tty) -> io::Result<libc::pid_t> {
    let own_group = tty.own_group();
    let fd = tty.foreground().map(Terminal::fd);
    let ttou = sigset(&[libc::SIGTTOU]);
    let empty = sigset(&[]);
    // SAFETY: the closure runs in the child between fork and exec, and it
    // makes only async-signal-safe calls on sets built before the fork.
    unsafe {
        command.pre_exec(move || {
            if own_group {
                libc::setpgid(0, 0);
            }
            if let Some(fd) = fd {
                libc::pthread_sigmask(libc::SIG_BLOCK, &ttou, ptr::null_mut());
                libc::tcsetpgrp(fd, libc::getpid());
            }
            libc::pthread_sigmask(libc::SIG_SETMASK, &empty, ptr::null_mut());
            libc::signal(libc::SIGPIPE, libc::SIG_DFL);
            Ok(())
        });
    }
    let child = command.spawn()?;
    let pid = child.id() as libc::pid_t;
    if own_group {
        // The entry sets the group too, so a signal it passes on at once
        // finds the group even before the child has set it.
        // SAFETY: setpgid has no memory-safety preconditions.
        unsafe { libc::setpgid(pid, pid) };
    }
    Ok(pid)
}

/// What happened while the client ran.
pub struct Outcome {
    pub code: i32,
    /// When the first SIGTERM or SIGHUP arrived, which means that the
    /// container is stopping.
    pub stopping: Option<Instant>,
}

/// Waits for the client `pid`, passing each forwarded signal on to its
/// group. A client that stays in the entry's group, as `own_group` false
/// says, gets SIGINT and SIGQUIT from the terminal itself, so only SIGTERM
/// and SIGHUP go to it. Those two are followed by SIGCONT, as a shell does
/// for a stopped job, since a stopped client keeps them pending, and they
/// also call `on_stop`.
pub fn wait_client(pid: libc::pid_t, own_group: bool, on_stop: impl Fn()) -> Outcome {
    let mut stopping = None;
    loop {
        let sig = next_signal();
        if sig == libc::SIGCHLD {
            if let Some(code) = reap(pid) {
                return Outcome { code, stopping };
            }
            continue;
        }
        if job_control(sig) {
            continue;
        }
        let ends = sig == libc::SIGTERM || sig == libc::SIGHUP;
        if own_group || ends {
            forward(pid, sig);
        }
        if ends {
            resume(pid);
            stopping.get_or_insert_with(Instant::now);
            on_stop();
        }
    }
}

static EXITING: Mutex<()> = Mutex::new(());

/// Exits with `code`. Two threads can end the entry at once, and only the
/// first one runs the exit.
pub fn finish(code: i32) -> ! {
    let _only = EXITING.lock();
    std::process::exit(code)
}

/// "1 other copy runs", or the count of them.
pub fn others(count: usize) -> String {
    match count {
        0 => "other copies run".into(),
        1 => "1 other copy runs".into(),
        n => format!("{n} other copies run"),
    }
}

/// "the other copy", or the count of them.
fn them(count: usize) -> String {
    match count {
        1 => "the other copy".into(),
        0 => "the other copies".into(),
        n => format!("the {n} other copies"),
    }
}

/// Writes one line to stderr. A terminal that is gone is not an error.
pub fn say(line: &str) {
    use std::io::Write;
    let _ = writeln!(io::stderr(), "{line}");
}

/// Exits with `code` after `delay`, unless the entry exits first.
fn finish_after(delay: Duration, code: i32) {
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        finish(code)
    });
}

/// Waits for the joined copies after the main entry's own client has
/// exited, and exits with that client's code. A first Ctrl-C says how to
/// end the session. A second one, SIGTERM, SIGHUP or SIGQUIT ends it, and
/// the copies then get SIGHUP and [`COPY_GRACE`] to exit. One more signal
/// ends the wait at once. Ctrl-Z cannot suspend the wait, and the first one
/// says how to end the session instead.
pub fn wait_for_copies(dir: PathBuf, lock: File, name: &OsStr, code: i32) -> ! {
    let count = copies(&dir).len();
    say(&format!("[launch] {} exited. The session stays open while {}.",
                 crate::shown(name), others(count)));
    let waiting = dir.clone();
    std::thread::spawn(move || {
        wait_end(&waiting, &lock);
        finish(code)
    });
    let mut interrupted = false;
    let mut ending = false;
    let mut suspended = false;
    loop {
        match next_signal() {
            libc::SIGCHLD => {
                reap(0);
            }
            libc::SIGTSTP if !ending && !suspended => {
                suspended = true;
                say(&format!("[launch] Ctrl-Z cannot suspend this session. Press Ctrl-C {} \
                              to end it, which stops {}.",
                             if interrupted { "again" } else { "twice" },
                             them(copies(&dir).len())));
            }
            sig if job_control(sig) => {}
            _ if ending => finish(code),
            libc::SIGINT if !interrupted => {
                interrupted = true;
                say(&format!("[launch] Press Ctrl-C again to end the session, which stops {}.",
                             them(copies(&dir).len())));
            }
            _ => {
                ending = true;
                let left = stop_copies(&dir);
                say(&format!("[launch] Ending the session, which gives {} {} seconds to exit. \
                              Press Ctrl-C to end it at once.",
                             them(left), COPY_GRACE.as_secs()));
                finish_after(COPY_GRACE, code);
            }
        }
    }
}

/// Waits for the joined copies after a stopping container ended the
/// session while the main entry's own client ran. The copies got SIGHUP
/// then, and they have until [`COPY_GRACE`] after `since` to exit. Any
/// signal but a job-control one ends the wait at once.
pub fn wait_for_stopped(dir: PathBuf, lock: File, code: i32, since: Instant) -> ! {
    finish_after(COPY_GRACE.saturating_sub(since.elapsed()), code);
    std::thread::spawn(move || {
        wait_end(&dir, &lock);
        finish(code)
    });
    loop {
        match next_signal() {
            libc::SIGCHLD => {
                reap(0);
            }
            sig if job_control(sig) => {}
            _ => finish(code),
        }
    }
}

/// A child holds a copy of every open lock from its fork until it execs or
/// exits. Tests that start a process hold this for reading, so a test that
/// checks a released lock can hold it for writing and see no such copy.
#[cfg(test)]
pub fn test_forks() -> &'static std::sync::RwLock<()> {
    static FORKS: std::sync::RwLock<()> = std::sync::RwLock::new(());
    &FORKS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn forking() -> std::sync::RwLockReadGuard<'static, ()> {
        test_forks().read().unwrap_or_else(|e| e.into_inner())
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("gmlx-session-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir.join("s")
    }

    fn cloexec(file: &File) -> bool {
        // SAFETY: the descriptor is open.
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFD) & libc::FD_CLOEXEC != 0 }
    }

    #[test]
    fn create_makes_a_private_folder_and_clears_an_old_session() {
        let dir = scratch("create");
        let lock = create(&dir).unwrap();
        assert!(cloexec(&lock));
        assert_eq!(fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
        fs::write(dir.join(ENDED), "").unwrap();
        fs::write(dir.join("copy-1"), "").unwrap();
        fs::write(dir.join("hangup-0f"), "").unwrap();
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
        drop(lock);
        create(&dir).unwrap();
        assert_eq!(fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
        for name in [ENDED, "copy-1", "hangup-0f"] {
            assert!(!dir.join(name).exists(), "{name}");
        }
    }

    #[test]
    fn create_refuses_a_link() {
        let dir = scratch("link");
        let real = dir.with_file_name("real");
        fs::create_dir_all(&real).unwrap();
        std::os::unix::fs::symlink(&real, &dir).unwrap();
        assert!(create(&dir).is_err());
    }

    #[test]
    fn create_makes_a_missing_parent_folder() {
        let dir = scratch("parent").with_file_name("run").join("gmlx-session");
        create(&dir).unwrap();
        let mode = |path: &Path| fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(dir.parent().unwrap()) & 0o700, 0o700);
        assert_eq!(mode(&dir), 0o700);
    }

    #[test]
    fn a_copy_joins_until_the_main_entry_ends() {
        let _forks = test_forks().write().unwrap_or_else(|e| e.into_inner());
        let dir = scratch("join");
        let lock = create(&dir).unwrap();
        let joined = join(&dir, None).unwrap();
        assert!(cloexec(&joined._lock));
        assert_eq!(copies(&dir), vec![std::process::id() as libc::pid_t]);
        assert!(!try_end(&dir, &lock));             // the copy holds the lock
        joined.leave();
        drop(joined);
        assert!(copies(&dir).is_empty());
        assert!(try_end(&dir, &lock));
        assert_eq!(join(&dir, None).err(), Some(Refused::Ending));   // the lock is exclusive now
        drop(lock);
        assert_eq!(join(&dir, None).err(), Some(Refused::Ending));   // the main entry is gone
    }

    #[test]
    fn a_copy_that_joins_as_the_session_ends_leaves_no_file() {
        // The end mark can come after the shared lock is taken, so the copy
        // looks for it once its own file is written.
        let _forks = test_forks().write().unwrap_or_else(|e| e.into_inner());
        let dir = scratch("late");
        let _lock = create(&dir).unwrap();
        mark_ended(&dir);
        for id in [None, Some("0f3a")] {
            assert_eq!(join(&dir, id).err(), Some(Refused::Ending));
            assert!(copies(&dir).is_empty());
            assert!(fs::read_dir(&dir).unwrap().flatten()
                .all(|e| !e.file_name().to_string_lossy().starts_with(COPY_PREFIX)));
        }
    }

    #[test]
    fn a_container_without_a_session_takes_no_copy() {
        let dir = scratch("none");
        assert_eq!(join(&dir, None).err(), Some(Refused::NoSession));
    }

    #[test]
    fn the_copy_file_of_a_process_that_is_gone_does_not_count() {
        let dir = scratch("gone");
        let _lock = create(&dir).unwrap();
        let _forks = forking();
        let mut child = Command::new("true").spawn().unwrap();
        let pid = child.id();
        child.wait().unwrap();
        fs::write(dir.join(format!("copy-{pid}")), "").unwrap();
        assert!(copies(&dir).is_empty());
        assert!(!dir.join(format!("copy-{pid}")).exists());
    }

    #[test]
    fn a_hangup_reaches_only_the_copy_of_its_id() {
        let dir = scratch("hangup");
        let _lock = create(&dir).unwrap();
        let _forks = forking();
        let mut copy = Command::new("sleep").arg("60").spawn().unwrap();
        let mut other = Command::new("sleep").arg("60").spawn().unwrap();
        fs::write(dir.join(format!("copy-{}", copy.id())), "0f3a").unwrap();
        fs::write(dir.join(format!("copy-{}", other.id())), "77").unwrap();
        assert!(!hung_up(&dir, "0f3a"));
        assert!(hangup(&dir, "0f3a"));
        assert!(hung_up(&dir, "0f3a") && !hung_up(&dir, "77"));
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(copy.wait().unwrap().signal(), Some(libc::SIGHUP));
        assert!(other.try_wait().unwrap().is_none());
        assert!(!hangup(&dir, "5e"));            // no copy has that ID
        other.kill().unwrap();
        other.wait().unwrap();
    }

    #[test]
    fn a_copy_whose_hangup_came_first_does_not_join() {
        let _forks = test_forks().write().unwrap_or_else(|e| e.into_inner());
        let dir = scratch("early");
        let _lock = create(&dir).unwrap();
        assert!(!hangup(&dir, "0f3a"));
        assert_eq!(join(&dir, Some("0f3a")).err(), Some(Refused::HungUp));
        assert!(copies(&dir).is_empty() && !dir.join("hangup-0f3a").exists());
        let joined = join(&dir, Some("77")).unwrap();
        assert_eq!(fs::read_to_string(&joined.pid_file).unwrap(), "77");
        joined.leave();
    }

    #[test]
    fn copy_ids_are_short_lowercase_hex() {
        assert!(valid_copy_id("0f3a") && valid_copy_id(&"a".repeat(64)));
        for bad in ["", "0F", "g", "../a", "a/b", &"a".repeat(65)] {
            assert!(!valid_copy_id(bad), "{bad}");
        }
    }

    #[test]
    fn exit_codes_follow_the_shell() {
        let _forks = forking();
        let code = |cmd: &str| {
            let status = Command::new("sh").arg("-c").arg(cmd).status().unwrap();
            use std::os::unix::process::ExitStatusExt;
            exit_code(status.into_raw())
        };
        assert_eq!(code("exit 7"), 7);
        assert_eq!(code("kill -TERM $$"), 128 + libc::SIGTERM);
    }

    /// Blocks until the child `pid` stops or exits, and leaves that state
    /// for `reap` to collect.
    fn wait_change(pid: libc::pid_t) {
        let mut info = MaybeUninit::<libc::siginfo_t>::zeroed();
        // SAFETY: info is valid for the call.
        unsafe {
            libc::waitid(libc::P_PID, pid as libc::id_t, info.as_mut_ptr(),
                         libc::WEXITED | libc::WSTOPPED | libc::WNOWAIT)
        };
    }

    #[test]
    fn reap_resumes_a_client_that_stops_itself() {
        // reap waits for any child, so no other test may start one now.
        let _forks = test_forks().write().unwrap_or_else(|e| e.into_inner());
        use std::os::unix::process::CommandExt;
        let child = Command::new("sh").arg("-c").arg("kill -TSTP 0; kill -STOP $$; exit 7")
            .process_group(0).spawn().unwrap();
        let pid = child.id() as libc::pid_t;
        let mut code = None;
        for _ in 0..10 {
            wait_change(pid);
            code = reap(pid);
            if code.is_some() {
                break;
            }
        }
        if code.is_none() {
            forward(pid, libc::SIGKILL);
            reap(pid);
        }
        assert_eq!(code, Some(7));
        assert!(RESUMED.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[test]
    fn others_counts_copies() {
        assert_eq!(others(0), "other copies run");
        assert_eq!(others(1), "1 other copy runs");
        assert_eq!(others(3), "3 other copies run");
    }

    #[test]
    fn the_hints_count_more_than_one_copy() {
        assert_eq!(them(0), "the other copies");
        assert_eq!(them(1), "the other copy");
        assert_eq!(them(2), "the 2 other copies");
    }
}
