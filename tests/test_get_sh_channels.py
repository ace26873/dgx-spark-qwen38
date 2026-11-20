#!/usr/bin/env python3
"""get.sh's channels, offline, against a real git fixture (decision 1529, 2026-11-19).

The one command installs and updates the stable channel: the latest published release,
not main anymore. --edge opts a box into main, --stable back, and the choice is kept in
~/.config/qwen38/channel. The fixture is a bare origin whose path carries the project's
slug (so get.sh's origin check passes), a box cloned from it one release behind, and a
curl shim on PATH: the release API is answered from a file, never the network, and its
failure is what exercises the ls-remote fallback. This is the update from N-1 to N the
CI runs on every push."""
import os
import shutil
import subprocess
import sys
import tempfile

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GET = os.path.join(REPO_DIR, "get.sh")
STUB = '#!/usr/bin/env bash\necho "INSTALL_RAN args=$*"\n'


# git here never inherits the variables that point it at a repository (GIT_DIR, GIT_WORK_TREE,
# GIT_INDEX_FILE, the rest of `git rev-parse --local-env-vars`: a git hook or `rebase -x` exports
# them), which would turn the fixture's commands onto that repository, and keeps the others,
# GIT_CONFIG_GLOBAL among them, which a caller may set to keep a personal config out
# (code review, 2026-11-01)
LOCAL_VARS = set(subprocess.run(["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True).stdout.split()) or {
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CONFIG", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "GIT_OBJECT_DIRECTORY",
    "GIT_DIR", "GIT_WORK_TREE", "GIT_IMPLICIT_WORK_TREE", "GIT_GRAFT_FILE", "GIT_INDEX_FILE", "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE", "GIT_PREFIX", "GIT_SHALLOW_FILE", "GIT_COMMON_DIR"}
CLEAN = {k: v for k, v in os.environ.items() if k not in LOCAL_VARS}


def git(*a, cwd, check=True):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=check, env=CLEAN)


def fixture(base):
    """A bare 'origin' whose URL carries the slug, a publisher clone, and the box: a clone of
    the origin as it stands, which is main at release N-1 once the publisher moves it to N."""
    origin = os.path.join(base, "hasso5703", "dgx-spark-qwen38.git")
    os.makedirs(origin)
    git("init", "-q", "--bare", "-b", "main", ".", cwd=origin)
    pub = os.path.join(base, "pub")
    git("clone", "-q", origin, pub, cwd=base)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        git("config", k, v, cwd=pub)
    with open(os.path.join(pub, "install.sh"), "w") as f:
        f.write(STUB)
    os.chmod(os.path.join(pub, "install.sh"), 0o755)
    git("add", "-A", cwd=pub); git("commit", "-qm", "the N-1 release", cwd=pub)
    git("tag", "v1.54.0", cwd=pub)
    git("push", "-q", "origin", "main", "v1.54.0", cwd=pub)
    box = os.path.join(base, "box")
    git("clone", "-q", origin, box, cwd=base)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        git("config", k, v, cwd=box)
    return pub, box


def publish(pub, subject, tag):
    """The next release on main, tagged: what the stable channel updates to."""
    with open(os.path.join(pub, "CHANGELOG.md"), "a") as f:
        f.write(f"## {tag}\n{subject}\n")
    git("add", "-A", cwd=pub); git("commit", "-qm", subject, cwd=pub)
    git("tag", tag, cwd=pub)
    git("push", "-q", "origin", "main", tag, cwd=pub)
    return git("rev-parse", "HEAD", cwd=pub).stdout.strip()


def curl_shim(base, tag=None):
    """A curl first on PATH: the release API answering `tag`, or failing (its anonymous budget
    spent, no network), which is what sends get.sh to the remote's tags. The real curl never
    runs, so no test here reaches the network."""
    bin_dir = os.path.join(base, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    shim = os.path.join(bin_dir, "curl")
    with open(shim, "w") as f:
        if tag:
            f.write('#!/usr/bin/env bash\nprintf \'{"tag_name":"%s"}\\n\' "' + tag + '"\n')
        else:
            f.write("#!/usr/bin/env bash\nexit 22\n")
    os.chmod(shim, 0o755)
    return bin_dir


def git_shim(base):
    """A git first on PATH that fails exactly the fetches that bring tags (a tag refspec,
    --tags): what a shallow clone (a manual git clone --depth) meets when the release's
    tag cannot be fetched into it. Everything else, the fetch of main included, goes to
    the real git."""
    bin_dir = os.path.join(base, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    real = shutil.which("git")
    with open(os.path.join(bin_dir, "git"), "w") as f:
        f.write('#!/usr/bin/env bash\n'
                'want=0\n'
                'for a in "$@"; do\n'
                '  case "$a" in fetch) want=1 ;; refs/tags/*|--tags*) [ "$want" = 1 ] && exit 128 ;; esac\n'
                'done\n'
                f'exec "{real}" "$@"\n')
    os.chmod(os.path.join(bin_dir, "git"), 0o755)
    return bin_dir


def isolated(base, bin_dir, extra=None):
    """What every run of get.sh here gets: PATH (the shim's folder first) and nothing else of
    this shell's environment, English messages, the fixture's folder for HOME, git with no
    network protocol but local paths, and no repository looked for above that folder. So no
    run reaches the network, a real clone of this repo or a real installer."""
    return {"PATH": bin_dir + os.pathsep + os.environ["PATH"], "LANG": "C.UTF-8", "HOME": base,
            "GIT_ALLOW_PROTOCOL": "file", "GIT_CEILING_DIRECTORIES": os.path.dirname(base), **(extra or {})}


def run_get(box, bin_dir, args=(), env_extra=None):
    base = os.path.dirname(os.path.abspath(box))
    return subprocess.run(["bash", GET, *args], cwd=base, capture_output=True, text=True,
                          env=isolated(base, bin_dir, env_extra) | {"DIR": box})


def state(box, *a):
    return git(*a, cwd=box).stdout.strip()


def main() -> None:
    base = tempfile.mkdtemp()
    try:
        pub, box = fixture(base)
        channel_file = os.path.join(base, ".config", "qwen38", "channel")

        # 1. A box on main at N-1, no channel file: the update takes the stable channel, the
        #    latest published release, on a branch of its own. The old code stayed on main and
        #    reset to its tip: this is the migration of every box installed before.
        n = publish(pub, "the N release", "v1.55.0")
        r = run_get(box, curl_shim(base, "v1.55.0"))
        assert r.returncode == 0, f"the first stable update failed: {r.stderr}"
        assert "INSTALL_RAN" in r.stdout, r.stdout
        assert "Channel: stable" in r.stdout and "v1.55.0" in r.stdout, r.stdout
        assert state(box, "symbolic-ref", "-q", "--short", "HEAD") == "stable", "not on the stable branch"
        assert state(box, "rev-parse", "HEAD") == n, "not at the release the tag names"
        assert open(channel_file).read().strip() == "stable", "the channel file was not written"
        assert state(box, "rev-parse", "--short", "main") == state(box, "rev-parse", "--short", "v1.54.0"), \
            "the main branch was moved by a stable update"

        # 2. The update from N-1 to N on the stable branch: a new release out, the same command.
        n1 = publish(pub, "the N+1 release", "v1.56.0")
        r = run_get(box, curl_shim(base, "v1.56.0"))
        assert r.returncode == 0 and "INSTALL_RAN" in r.stdout, f"the stable update failed: {r.stderr}"
        assert state(box, "symbolic-ref", "-q", "--short", "HEAD") == "stable"
        assert state(box, "rev-parse", "HEAD") == n1, "the stable clone did not move to the new release"

        # 3. --edge: main, and the choice kept.
        r = run_get(box, curl_shim(base, "v1.56.0"), args=["--edge"])
        assert r.returncode == 0 and "Channel: edge" in r.stdout, r.stdout + r.stderr
        assert state(box, "symbolic-ref", "-q", "--short", "HEAD") == "main"
        assert state(box, "rev-parse", "HEAD") == n1, "edge did not install main's tip"
        assert open(channel_file).read().strip() == "edge", "the edge choice was not kept"

        # 4. --stable: back to the latest published release, on the stable branch.
        r = run_get(box, curl_shim(base, "v1.56.0"), args=["--stable"])
        assert r.returncode == 0 and "INSTALL_RAN" in r.stdout, f"back to stable failed: {r.stderr}"
        assert state(box, "symbolic-ref", "-q", "--short", "HEAD") == "stable"
        assert state(box, "rev-parse", "HEAD") == n1
        assert open(channel_file).read().strip() == "stable"

        # 5. The channel flags never reach install.sh; everything else does.
        r = run_get(box, curl_shim(base, "v1.56.0"), args=["--edge", "--no-service"])
        assert r.returncode == 0, r.stderr
        assert "INSTALL_RAN args=--no-service" in r.stdout, r.stdout
        run_get(box, curl_shim(base, "v1.56.0"), args=["--stable"])

        # 6. The release API failing (no network, its anonymous budget spent): the remote's own
        #    tags answer, which no API limit touches.
        n2 = publish(pub, "the N+2 release", "v1.57.0")
        r = run_get(box, curl_shim(base))
        assert r.returncode == 0 and "INSTALL_RAN" in r.stdout, f"the ls-remote fallback failed: {r.stderr}"
        assert state(box, "rev-parse", "HEAD") == n2, "the fallback did not find the newest tag"

        # 6b. A tag pushed but not published as a release (Hasan tags, the release comes
        #     later): the API's answer still wins when it comes, and when it does not, the
        #     fallback says what it took, the highest tag, not "the latest published release"
        #     (M1 of the review, 2026-11-20).
        git("tag", "v1.99.0", cwd=pub); git("push", "-q", "origin", "v1.99.0", cwd=pub)
        r = run_get(box, curl_shim(base, "v1.57.0"))
        assert r.returncode == 0, r.stderr
        assert "the latest published release, v1.57.0" in r.stdout, r.stdout
        assert state(box, "rev-parse", "HEAD") == n2, "the API's answer did not win over the higher tag"
        r = run_get(box, curl_shim(base))
        assert r.returncode == 0 and "INSTALL_RAN" in r.stdout, r.stderr
        assert "the highest tag, v1.99.0, which may not be a published release yet" in r.stdout, r.stdout
        assert "the latest published release" not in r.stdout, r.stdout
        assert state(box, "rev-parse", "HEAD") == state(box, "rev-parse", "v1.99.0"), \
            "the fallback did not take the highest tag"

        # 7. A channel file that says nothing get.sh knows: stable, the default.
        os.makedirs(os.path.dirname(channel_file), exist_ok=True)
        with open(channel_file, "w") as f:
            f.write("nonsense\n")
        r = run_get(box, curl_shim(base, "v1.57.0"))
        assert r.returncode == 0 and "Channel: stable" in r.stdout, r.stdout + r.stderr
        assert state(box, "symbolic-ref", "-q", "--short", "HEAD") == "stable"

        # 8. A commit of the user's own on the stable branch: the same refusal as on main.
        with open(os.path.join(box, "my-change.txt"), "w") as f:
            f.write("mine\n")
        git("add", "my-change.txt", cwd=box); git("commit", "-qm", "mine", cwd=box)
        r = run_get(box, curl_shim(base, "v1.57.0"))
        assert r.returncode == 1, "a local commit on stable did not stop the update"
        assert "local commits not on origin/main" in r.stderr, r.stderr
        # and --edge leaves the commit where it is, on the stable branch, and installs main
        r = run_get(box, curl_shim(base, "v1.57.0"), args=["--edge"])
        assert r.returncode == 0 and "INSTALL_RAN" in r.stdout, f"the switch to edge failed: {r.stderr}"
        assert state(box, "symbolic-ref", "-q", "--short", "HEAD") == "main"
        assert state(box, "rev-list", "--count", "origin/main..stable") == "1", \
            "the commit left on the stable branch was lost by the switch to edge"
        # a stable branch that holds commits of its own is never moved without a backup either
        r = run_get(box, curl_shim(base, "v1.57.0"), args=["--stable"])
        assert r.returncode == 0 and "kept on branch" in r.stdout, r.stdout + r.stderr
        assert state(box, "for-each-ref", "--format=%(refname:short)", "refs/heads/backup-*"), \
            "no backup branch keeps the commit the stable branch held"
        assert state(box, "symbolic-ref", "-q", "--short", "HEAD") == "stable"
        assert state(box, "rev-parse", "HEAD") == n2, "stable did not return to the release"

        # 9. A modified tracked file on stable: the same refusal as on main.
        with open(os.path.join(box, "install.sh"), "a") as f:
            f.write("# touched\n")
        r = run_get(box, curl_shim(base, "v1.57.0"))
        assert r.returncode == 1 and "modified tracked files" in r.stderr, r.stderr
        git("checkout", "--", "install.sh", cwd=box)

        # 10. No published release to find (a repo with no tag, the API down): nothing is touched.
        bare = os.path.join(base, "hasso5703", "alzeo.git")
        os.makedirs(bare)
        git("init", "-q", "--bare", "-b", "main", ".", cwd=bare)
        empty = os.path.join(base, "empty")
        git("clone", "-q", bare, empty, cwd=base)
        # an identity of its own, as the other clones here: the CI's throwaway HOME has no git identity
        for k, v in (("user.email", "t@t"), ("user.name", "t")):
            git("config", k, v, cwd=empty)
        with open(os.path.join(empty, "install.sh"), "w") as f:
            f.write(STUB)
        os.chmod(os.path.join(empty, "install.sh"), 0o755)
        git("add", "-A", cwd=empty); git("commit", "-qm", "no release yet", cwd=empty)
        git("push", "-q", "origin", "main", cwd=empty)
        before = state(empty, "rev-parse", "HEAD")
        # a refused run writes nothing anywhere, the channel file included (M2 of the
        # review, 2026-11-20): the choice is recorded only once the update has succeeded
        if os.path.exists(channel_file):
            os.remove(channel_file)
        r = run_get(empty, curl_shim(base))
        assert r.returncode == 1, "a repo with no release updated anyway"
        assert "could not find the latest published release" in r.stderr, r.stderr
        assert "Nothing was touched" in r.stderr, r.stderr
        assert not os.path.exists(channel_file), "a refused run wrote the channel file"
        assert state(empty, "rev-parse", "HEAD") == before, "the clone moved"
        assert state(empty, "symbolic-ref", "-q", "--short", "HEAD") == "main", "the clone left main"
        assert not state(empty, "branch", "--list", "stable"), "a stable branch was created"

        # 11. The API's tag_name is read whole: a suffix of the tag's own (-rc1, +build) is
        #     never cut off (B1 of the review, 2026-11-20). The old sed kept v[0-9][0-9.]* and
        #     read "v1.58.0" out of "v1.58.0-rc1": a tag that does not exist, and a refusal.
        rc_tag = "v1.58.0-rc1"
        rc_commit = publish(pub, "the rc release", rc_tag)
        r = run_get(box, curl_shim(base, rc_tag))
        assert r.returncode == 0 and "INSTALL_RAN" in r.stdout, f"the suffixed tag failed: {r.stderr}"
        assert f"the latest published release, {rc_tag}" in r.stdout, r.stdout
        assert state(box, "rev-parse", "HEAD") == rc_commit, "the suffixed tag was truncated to another"

        # 12. Offline: a clear refusal, not git's raw error, and nothing touched (B3 of the
        #     review, 2026-11-20). The old code died on the fetch of main with git's own
        #     message (rc 128) before any source of tag was tried.
        offbox = os.path.join(base, "offline-box")
        origin = os.path.join(base, "hasso5703", "dgx-spark-qwen38.git")
        git("clone", "-q", origin, offbox, cwd=base)
        for k, v in (("user.email", "t@t"), ("user.name", "t")):
            git("config", k, v, cwd=offbox)
        # an origin this script accepts as this repo, at a path that does not exist: every
        # network call of the run fails, the fetch of main first
        git("remote", "set-url", "origin", os.path.join(base, "hasso5703", "dgx-spark-qwen38-absent.git"),
            cwd=offbox)
        before = state(offbox, "rev-parse", "HEAD")
        if os.path.exists(channel_file):
            os.remove(channel_file)
        r = run_get(offbox, curl_shim(base))
        assert r.returncode == 1, "an offline run did not refuse"
        assert "cannot reach GitHub" in r.stderr, r.stderr
        assert "Nothing was touched" in r.stderr, r.stderr
        assert "fatal" not in r.stderr, f"git's raw error leaked through: {r.stderr}"
        assert state(offbox, "rev-parse", "HEAD") == before, "the offline run moved the clone"
        assert state(offbox, "symbolic-ref", "-q", "--short", "HEAD") == "main", "the clone left main"
        assert not os.path.exists(channel_file), "the offline run wrote the channel file"

        # 13. A shallow clone (a manual git clone --depth) whose release tag cannot be fetched:
        #     a refusal that says what to do, and nothing touched (B4 of the review, 2026-11-20).
        #     The old refusal said only "could not find the latest published release". The origin
        #     is cloned over the file:// transport, which keeps the clone shallow the way a
        #     network remote does (a plain local path fetch unshallows it); the tag the API
        #     names is taken out of the clone and its fetches fail, the way a shallow clone
        #     meets a tag its cut history cannot take.
        shabox = os.path.join(base, "shallow-box")
        git("clone", "-q", "--depth", "1", "file://" + origin, shabox, cwd=base)
        for k, v in (("user.email", "t@t"), ("user.name", "t")):
            git("config", k, v, cwd=shabox)
        for line in git("tag", "-l", cwd=shabox).stdout.split():
            git("update-ref", "-d", "refs/tags/" + line, cwd=shabox)
        before = state(shabox, "rev-parse", "HEAD")
        if os.path.exists(channel_file):
            os.remove(channel_file)
        bin_dir = curl_shim(base, "v1.55.0")
        git_shim(base)
        r = run_get(shabox, bin_dir)
        assert r.returncode == 1, "a shallow clone that cannot fetch the tag updated anyway"
        assert "shallow" in r.stderr, r.stderr
        assert "fetch --unshallow" in r.stderr, r.stderr
        assert "--edge" in r.stderr, r.stderr
        assert "Nothing was touched" in r.stderr, r.stderr
        assert state(shabox, "rev-parse", "HEAD") == before, "the refused run moved the clone"
        assert state(shabox, "symbolic-ref", "-q", "--short", "HEAD") == "main", "the clone left main"
        assert not state(shabox, "branch", "--list", "stable"), "a stable branch was created"
        assert not os.path.exists(channel_file), "the refused run wrote the channel file"

        # 14. A box on edge whose main runs past the release, taking --stable: the update says
        #     it moves the clone back before it does (how many commits, from which version to
        #     which, the way back), then goes on as always (B5 of the review, 2026-11-20).
        for _ in range(2):
            # the CHANGELOG's top heading is the version main runs, the newest first as the
            # real one goes; the version the warning names is read from there
            old = open(os.path.join(pub, "CHANGELOG.md")).read()
            with open(os.path.join(pub, "CHANGELOG.md"), "w") as f:
                f.write("## v1.59.0 (unreleased)\nwork on main past the release\n" + old)
            git("add", "-A", cwd=pub); git("commit", "-qm", "main past the release", cwd=pub)
        git("push", "-q", "origin", "main", cwd=pub)
        r = run_get(box, curl_shim(base, "v1.58.0-rc1"), args=["--edge"])
        assert r.returncode == 0 and "Channel: edge" in r.stdout, r.stdout + r.stderr
        assert state(box, "rev-parse", "HEAD") != rc_commit, "edge did not take main's new tip"
        r = run_get(box, curl_shim(base, "v1.58.0-rc1"), args=["--stable"])
        assert r.returncode == 0 and "INSTALL_RAN" in r.stdout, f"the move back failed: {r.stderr}"
        assert "moves the clone back" in r.stdout, r.stdout
        assert "2 commit" in r.stdout, r.stdout
        assert "from v1.59.0" in r.stdout and "v1.58.0-rc1" in r.stdout, r.stdout
        assert "--edge" in r.stdout, r.stdout
        assert state(box, "rev-parse", "HEAD") == rc_commit, "the clone did not return to the release"
        assert state(box, "symbolic-ref", "-q", "--short", "HEAD") == "stable"

        print("test_get_sh_channels: OK")
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
