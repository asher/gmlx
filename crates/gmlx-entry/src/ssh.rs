//! Links root's `.ssh` to the one in the private home.
//!
//! The client runs as root with `HOME` set to its private home, but ssh
//! finds `~/.ssh` from the home in the password file, which is `/root`, and
//! no session keeps `/root`. A link from `/root/.ssh` to `$HOME/.ssh` keeps
//! the known hosts, keys and config that ssh writes in the private home.
//! An image that has its own `/root/.ssh` keeps it, and a root home on
//! another file system than `/`, such as a share or a volume, gets no link,
//! since the link would outlive the session there.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, MetadataExt};
use std::path::{Path, PathBuf};

/// The password file.
pub const PASSWD: &str = "/etc/passwd";
/// Names another password file, for the tests that run the entry outside a
/// guest. The client never sees it.
pub const PASSWD_ENV: &str = "GMLX_ENTRY_PASSWD";

/// Names the folder whose file system stands for `/`, for the tests that
/// run the entry outside a guest. The client never sees it.
pub const ROOTFS_ENV: &str = "GMLX_ENTRY_ROOTFS";

/// The folder whose file system is the guest's own: `/`, or the folder
/// [`ROOTFS_ENV`] names.
pub fn rootfs() -> PathBuf {
    std::env::var_os(ROOTFS_ENV)
        .filter(|p| !p.is_empty())
        .map_or_else(|| PathBuf::from("/"), PathBuf::from)
}

/// The password file: [`PASSWD`], or the file [`PASSWD_ENV`] names.
pub fn passwd() -> PathBuf {
    std::env::var_os(PASSWD_ENV)
        .filter(|p| !p.is_empty())
        .map_or_else(|| PathBuf::from(PASSWD), PathBuf::from)
}

/// The home of the first user with ID 0 in the password file `text`.
fn root_home(text: &[u8]) -> Option<PathBuf> {
    text.split(|&b| b == b'\n').find_map(|line| {
        let fields: Vec<&[u8]> = line.split(|&b| b == b':').collect();
        (fields.len() >= 7 && fields[2] == b"0")
            .then(|| PathBuf::from(OsStr::from_bytes(fields[5])))
    })
}

/// Whether `dir` is on the same file system as `rootfs`. A folder that
/// cannot be read counts as another one.
fn on_rootfs(dir: &Path, rootfs: &Path) -> bool {
    match (fs::metadata(dir), fs::metadata(rootfs)) {
        (Ok(a), Ok(b)) => a.dev() == b.dev(),
        _ => false,
    }
}

/// Links `<root's home>/.ssh` to `<home>/.ssh`, and makes `<home>/.ssh`
/// with mode 700 when it is missing. Nothing changes when the password file
/// has no root, when `home` is missing, relative or root's home, when
/// root's home does not exist or is on another file system than `rootfs`,
/// or when anything is at its `.ssh` already. Returns whether the link was
/// made. A step that fails leaves ssh as it was, and removes the
/// `<home>/.ssh` this call made, so the session starts all the same.
pub fn link_home(passwd: &Path, rootfs: &Path, home: Option<OsString>) -> bool {
    let Some(home) = home.map(PathBuf::from).filter(|h| h.is_absolute()) else { return false };
    let Some(root) = fs::read(passwd).ok().and_then(|text| root_home(&text)) else {
        return false;
    };
    if !root.is_absolute() || root == home || !root.is_dir() || !on_rootfs(&root, rootfs) {
        return false;
    }
    let link = root.join(".ssh");
    match fs::symlink_metadata(&link) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        _ => return false,
    }
    let target = home.join(".ssh");
    let made = match fs::DirBuilder::new().mode(0o700).create(&target) {
        Ok(()) => true,
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => false,
        Err(_) => return false,
    };
    if target.is_dir() && std::os::unix::fs::symlink(&target, &link).is_ok() {
        return true;
    }
    if made {
        // Only an empty folder goes, so nothing that ssh wrote is lost.
        let _ = fs::remove_dir(&target);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    struct Scratch {
        passwd: PathBuf,
        root: PathBuf,
        home: PathBuf,
        /// A folder on the file system of `root`, which stands for `/`.
        rootfs: PathBuf,
    }

    fn scratch(name: &str) -> Scratch {
        let dir = std::env::temp_dir()
            .join(format!("gmlx-ssh-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let (root, home) = (dir.join("root"), dir.join("home"));
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&home).unwrap();
        let passwd = dir.join("passwd");
        fs::write(&passwd, format!("daemon:x:1:1::/usr/sbin:/bin/false\n\
                                    root:x:0:0:root:{}:/bin/sh\n", root.display())).unwrap();
        Scratch { passwd, root, home, rootfs: dir }
    }

    impl Scratch {
        fn link(&self, home: Option<OsString>) -> bool {
            link_home(&self.passwd, &self.rootfs, home)
        }
    }

    fn linked(s: &Scratch) -> Option<PathBuf> {
        fs::read_link(s.root.join(".ssh")).ok()
    }

    #[test]
    fn root_home_is_the_first_user_with_id_0() {
        assert_eq!(root_home(b"a:x:10:0::/a:/bin/sh\ntoor:x:0:0::/var/toor:/bin/sh\n\
                               root:x:0:0::/root:/bin/sh"), Some(PathBuf::from("/var/toor")));
        assert_eq!(root_home(b"a:x:10:0::/a:/bin/sh\n"), None);
        assert_eq!(root_home(b"root:x:0:0:/root\n"), None);
    }

    #[test]
    fn links_root_ssh_to_a_new_private_ssh_folder() {
        let s = scratch("new");
        assert!(s.link(Some(s.home.clone().into())));
        assert_eq!(linked(&s), Some(s.home.join(".ssh")));
        assert_eq!(fs::metadata(s.home.join(".ssh")).unwrap().permissions().mode() & 0o777,
                   0o700);
    }

    #[test]
    fn keeps_a_private_ssh_folder_that_exists() {
        let s = scratch("kept");
        fs::create_dir(s.home.join(".ssh")).unwrap();
        fs::write(s.home.join(".ssh/known_hosts"), "host").unwrap();
        assert!(s.link(Some(s.home.clone().into())));
        assert_eq!(fs::read(s.root.join(".ssh/known_hosts")).unwrap(), b"host");
    }

    #[test]
    fn leaves_anything_at_root_ssh() {
        for (name, make) in [
            ("folder", (|p: &Path| fs::create_dir(p).unwrap()) as fn(&Path)),
            ("file", |p: &Path| fs::write(p, "").unwrap()),
            ("link", |p: &Path| std::os::unix::fs::symlink("/nowhere", p).unwrap()),
        ] {
            let s = scratch(name);
            make(&s.root.join(".ssh"));
            assert!(!s.link(Some(s.home.clone().into())), "{name}");
            assert!(!s.home.join(".ssh").exists(), "{name}");
        }
    }

    #[test]
    fn does_nothing_without_a_usable_home_or_root() {
        let s = scratch("skip");
        assert!(!s.link(None));
        assert!(!s.link(Some("home".into())));
        assert!(!s.link(Some(s.root.clone().into())));
        assert!(!s.root.join(".ssh").exists());      // the home is root's own
        assert!(!link_home(&s.passwd.with_file_name("none"), &s.rootfs,
                           Some(s.home.clone().into())));
        fs::write(&s.passwd, "root:x:0:0::/nowhere/root:/bin/sh\n").unwrap();
        assert!(!s.link(Some(s.home.clone().into())));
        assert!(!s.home.join(".ssh").exists());
    }

    #[test]
    fn a_root_home_on_another_file_system_gets_no_link() {
        // /dev is a file system of its own on Linux and on macOS, so it
        // stands for the guest's `/` while root's home is a share.
        let s = scratch("device");
        let dev = Path::new("/dev");
        assert_ne!(fs::metadata(dev).unwrap().dev(), fs::metadata(&s.root).unwrap().dev());
        assert!(!link_home(&s.passwd, dev, Some(s.home.clone().into())));
        assert!(!link_home(&s.passwd, &s.rootfs.join("gone"), Some(s.home.clone().into())));
        assert_eq!(linked(&s), None);
        assert!(!s.home.join(".ssh").exists());
    }

    #[test]
    fn a_link_that_fails_removes_only_the_folder_it_made() {
        // SAFETY: geteuid has no preconditions.
        if unsafe { libc::geteuid() } == 0 {
            return;                               // root writes a read-only folder anyway
        }
        let s = scratch("readonly");
        fs::set_permissions(&s.root, fs::Permissions::from_mode(0o555)).unwrap();
        assert!(!s.link(Some(s.home.clone().into())));
        assert!(!s.home.join(".ssh").exists());
        fs::create_dir(s.home.join(".ssh")).unwrap();
        assert!(!s.link(Some(s.home.clone().into())));
        assert!(s.home.join(".ssh").is_dir());    // it was there before
        fs::set_permissions(&s.root, fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn a_home_that_cannot_hold_ssh_gets_no_link() {
        let s = scratch("nohome");
        assert!(!s.link(Some(s.home.join("gone").into())));
        fs::write(s.home.join(".ssh"), "").unwrap();
        assert!(!s.link(Some(s.home.clone().into())));
        assert_eq!(linked(&s), None);
    }
}
