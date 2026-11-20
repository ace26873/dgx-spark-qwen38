#!/usr/bin/env bash
# One-liner bootstrap: first install AND update, same command, any directory:
#   curl -fsSL https://get.alzeo.ai | bash
# (get.alzeo.ai sends curl on to this very file, on GitHub's raw.githubusercontent.com)
# NO sudo in front, ever: install.sh calls sudo itself for the steps that need
# it, and under sudo every path it writes moves to /root (refused below).
# It installs the box: the lane that serves, the proxy and the cockpit, and
# every other lane on demand (its first Load from the cockpit builds it), and
# prints the cockpit URL when it is done.
# Pass install.sh options after `bash -s --`, e.g.:
#   curl -fsSL https://get.alzeo.ai | bash -s -- --no-service
# The channel: stable by default (the latest published release), --edge to follow main:
#   curl -fsSL https://get.alzeo.ai | bash -s -- --edge
# Env overrides pass through (PORT=, HF_CACHE=, DIR= for a custom clone location).
# Everything is wrapped in main() so nothing runs on a partial download.
set -euo pipefail
# One spelling of the home, without a trailing slash, as install.sh says why: the cockpit's
# exact-argv sudoers is rendered from it and has to match the cockpit's own calls (2026-10-22).
while [ -n "${HOME:-}" ] && [ "$HOME" != / ] && [ "${HOME%/}" != "$HOME" ]; do HOME="${HOME%/}"; done; export HOME

main() {
  # ── This does not run under sudo, and the refusal comes before the clone ──
  # Piped into `sudo bash`, every path below moves to /root: the clone lands in
  # /root/dgx-spark-qwen38 and install.sh then writes /root/.config/qwen38 with
  # a brand new API key, leaving a healthy engine that answers 401 to every
  # client reading ~/.config/qwen38/api-key. That happened on the reference box
  # on 2026-09-13. install.sh calls sudo itself, for the steps that need it.
  # install.sh refuses again on its own; this one exists so the refusal lands
  # before anything is written, and names the command that was actually typed.
  if [ "$(id -u)" = "0" ]; then
    if [ -n "${SUDO_USER:-}" ]; then
      printf '\n\033[1;31mERROR:\033[0m do not pipe this into "sudo bash" (your login is %s).\n\n' "$SUDO_USER" >&2
      printf '  run : curl -fsSL https://get.alzeo.ai | bash\n' >&2
      printf '        (no sudo in front: the installer calls sudo itself, for the steps that need it)\n\n' >&2
      printf 'Under sudo, HOME is /root: the clone and the API key land there, the units point\n' >&2
      printf 'at them, and every client reading ~/.config/qwen38/api-key gets 401 from an engine\n' >&2
      printf 'that installed, started and served perfectly well.\n' >&2
      exit 1
    fi
    if [ "${ALLOW_ROOT:-0}" != "1" ]; then
      printf '\n\033[1;31mERROR:\033[0m run this as the user who will use the box, not as root.\n' >&2
      printf '  everything it installs is addressed from $HOME, which as root is /root.\n' >&2
      printf '  if this box genuinely has no other user: ALLOW_ROOT=1 before the pipe.\n' >&2
      exit 1
    fi
  fi

  REPO_URL="https://github.com/hasso5703/alzeo"
  DEFAULT_DIR="$HOME/dgx-spark-qwen38"
  # Is this origin a clone of THIS repo? The repository was renamed hasso5703/alzeo on 2026-11-01:
  # every box installed before keeps the old name in its origin, which GitHub redirects, and a
  # clone made from the new address (REPO_URL, since v1.41.0) carries the new one. Both are this repo, for
  # good. The old name keeps the pattern it always had, now whatever the case, which only takes more,
  # so that no box installed before is refused.
  # The new name is matched as the last two parts of the path, HTTPS or SSH, with or without .git,
  # on any host as the old name always was (a mirror of this repository counts): hasso5703/alzeo-site,
  # or xhasso5703/alzeo, is another repository, which this script must not update or install from.
  # GitHub's names ignore case, and so does this: a clone of https://github.com/Hasso5703/Alzeo is
  # this repo too. Lowered by tr in the C locale, where every capital lowers as in English: bash 3.2
  # (macOS) takes ${1,,} for a bad substitution (code review, 2026-11-01).
  this_repo() {
    case "$(printf '%s' "$1" | LC_ALL=C tr '[:upper:]' '[:lower:]')" in
      *hasso5703/dgx-spark-qwen38*|*[/:]hasso5703/alzeo|*[/:]hasso5703/alzeo.git|*[/:]hasso5703/alzeo/|*[/:]hasso5703/alzeo.git/|*[/:]hasso5703/alzeo/.git|*[/:]hasso5703/alzeo/.git/) return 0 ;;
    esac
    return 1
  }
  # The highest vX.Y.Z of what it reads, numbers compared left to right: sort -V is a GNU
  # extension the BSD sort of a mac does not have, and plain sort reads v1.10 below v1.9.
  highest_tag() {
    awk '$0 ~ /^v[0-9]+\.[0-9]+(\.[0-9]+)?$/ { split(substr($0, 2), p, ".")
         n = p[1] * 1000000 + p[2] * 1000 + p[3]
         if (t == "" || n > m) { m = n; t = $0 } } END { print t }'
  }
  command -v git >/dev/null || { echo "ERROR: git is required (stock on DGX OS)." >&2; exit 1; }

  # Case 1: running from inside an existing clone of this repo (any path) -> use it.
  DIR="${DIR:-}"
  if [ -z "$DIR" ] && TOP=$(git rev-parse --show-toplevel 2>/dev/null); then
    ORIGIN=$(git -C "$TOP" remote get-url origin 2>/dev/null || true)
    if this_repo "$ORIGIN"; then DIR="$TOP"; fi
  fi

  # Case 2: default location -> reuse the clone or create it.
  if [ -z "$DIR" ]; then
    DIR="$DEFAULT_DIR"
    if [ -d "$DIR/.git" ]; then
      :
    elif [ -e "$DIR" ]; then
      echo "ERROR: $DIR exists but is not a clone of this repo. Move it, or rerun with DIR=/path/to/clone" >&2
      exit 1
    else
      echo "── Cloning $REPO_URL into $DIR"
      git clone "$REPO_URL" "$DIR"
    fi
  fi

  # Whatever resolved DIR: it must be a clone of THIS repo before we touch it (this_repo above).
  DIR_ORIGIN=$(git -C "$DIR" remote get-url origin 2>/dev/null || true)
  if ! this_repo "$DIR_ORIGIN"; then
    # the origin as git keeps it, but for a user name and password or token in it, which a refusal
    # pasted into an issue would publish (code review, 2026-10-31). Everything up to the last @ goes,
    # after a remote helper's prefix if there is one (gcrypt::https://...), and the user of an
    # scp-like address: a password holding an @ or a / of its own goes too, and in the C locale one
    # holding a byte that is not UTF-8 (code review, 2026-11-01). An @ in a repository's own path,
    # which GitHub's names never hold, shows cut. Its lines are joined first: sed reads line by line, and a
    # line break in a password left its first part shown (code review, 2026-11-01).
    SHOWN=$(printf '%s' "$DIR_ORIGIN" | LC_ALL=C tr '\n\r' '  ' \
              | LC_ALL=C sed -E -e 's#^(([A-Za-z0-9+.-]+::)?[A-Za-z][A-Za-z0-9+.-]*://).*@#\1#' \
                                -e 's#^[^/:@]+@([^/:]+:)#\1#')
    # and any form left with an @ in it (a remote helper around an scp-like address, two helpers, a space before
    # the scheme): only what follows its last @. A token in the query string goes too, and an ext:: address,
    # a command that can carry a password, is not shown at all (code review, 2026-11-01).
    case "$SHOWN" in *@*) SHOWN="${SHOWN##*@}" ;; esac
    SHOWN="${SHOWN%%\?*}"
    case "$SHOWN" in ext::*) SHOWN="ext::(the command is not shown)" ;; esac
    echo "ERROR: $DIR is not a clone of this repo (origin: ${SHOWN:-none}). Refusing to touch it." >&2; exit 1
  fi

  echo "── Repo: $DIR"

  # ── The channel: stable (the latest published release) by default, edge (main) opted into ──
  # --stable and --edge are consumed here (never passed to install.sh) and the choice is kept in
  # ~/.config/qwen38/channel: a file of the config dir, which no reset of the clone touches (a
  # tracked file of the clone would be reset over, an untracked one copied aside by keep_aside).
  # No file yet: stable, which is also what a box installed before this channel exists migrates
  # to at its next update, from main, without downloading anything the tags do not already need.
  CHANNEL_FILE="$HOME/.config/qwen38/channel"
  CHANNEL=""; ARGS=()
  for a in "$@"; do
    case "$a" in
      --stable) CHANNEL=stable ;;
      --edge)   CHANNEL=edge ;;
      *) ARGS+=("$a") ;;
    esac
  done
  if [ -z "$CHANNEL" ]; then
    CHANNEL=stable
    if [ -r "$CHANNEL_FILE" ]; then
      case "$(tr -d '[:space:]' < "$CHANNEL_FILE")" in edge) CHANNEL=edge ;; esac
    fi
  fi
  # the choice is written before anything moves: a run that fails later has still kept it,
  # which is what the one who made it asked for (and the file is one word, read above)
  if [ ! -s "$CHANNEL_FILE" ] || [ "$(tr -d '[:space:]' < "$CHANNEL_FILE" 2>/dev/null)" != "$CHANNEL" ]; then
    mkdir -p "$HOME/.config/qwen38" && printf '%s\n' "$CHANNEL" > "$CHANNEL_FILE" \
      || echo "── NOTE: could not write the channel to $CHANNEL_FILE (this update still follows $CHANNEL)"
  fi

  git -C "$DIR" fetch -q origin main
  # The release tags too. A fetch of main alone brings none, and the cockpit's release check
  # and `git describe` read them: a box updated by this command kept naming the release it
  # was first cloned at (v1.19.0 on the reference box running v1.20.1, 2026-10-01). A tag of
  # the same name made on this box is kept, and the update goes on.
  git -C "$DIR" fetch -q --tags origin 2>/dev/null \
    || echo "── NOTE: the release tags did not all come (a tag of the same name exists here?); the update goes on"

  # What this update installs: origin/main on the edge channel, the latest published release on
  # the stable one. Three sources in order (DESIGN, decision 1529): GitHub's own "latest
  # release", the same answer the cockpit reads (published releases, prereleases excluded,
  # anonymous, no token); then the tags of the remote itself, which no API budget limits; then
  # what the fetch above already brought, for a box whose API call and ls-remote both fail.
  TAG=""
  if [ "$CHANNEL" = stable ]; then
    if command -v curl >/dev/null 2>&1; then
      TAG=$(curl -fsSL --max-time 8 -H 'Accept: application/vnd.github+json' \
              https://api.github.com/repos/hasso5703/alzeo/releases/latest 2>/dev/null \
            | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\(v[0-9][0-9.]*\)".*/\1/p') || TAG=""
    fi
    if [ -z "$TAG" ]; then
      TAG=$(git -C "$DIR" ls-remote --tags origin 2>/dev/null \
            | sed -n 's#.*[[:space:]]refs/tags/\(v[0-9]*\.[0-9]*\(\.[0-9]*\)\)$#\1#p' | highest_tag) || TAG=""
    fi
    if [ -z "$TAG" ]; then
      TAG=$(git -C "$DIR" tag -l | sed -n 's/^\(v[0-9]*\.[0-9]*\(\.[0-9]*\)\)$/\1/p' | highest_tag)
    fi
    # the tag must be one this clone has: the API can name a release the fetch above did not
    # bring (its fetch of the tags failed, or the cache of a redirect served an older list)
    if [ -n "$TAG" ] && ! git -C "$DIR" rev-parse -q --verify "refs/tags/$TAG" >/dev/null 2>&1; then
      git -C "$DIR" fetch -q origin "refs/tags/$TAG:refs/tags/$TAG" 2>/dev/null || true
    fi
    if [ -z "$TAG" ] || ! git -C "$DIR" rev-parse -q --verify "refs/tags/$TAG" >/dev/null 2>&1; then
      echo "ERROR: could not find the latest published release of this repo (no answer from GitHub's" >&2
      echo "  release API, none from the remote's tags, none in this clone). Nothing was touched." >&2
      echo "  try again online, or follow main instead:  bash -s -- --edge" >&2
      exit 1
    fi
    TARGET_BRANCH=stable
    TARGET_REF="refs/tags/$TAG"
    echo "── Channel: stable (the latest published release, $TAG)"
  else
    TARGET_BRANCH=main
    TARGET_REF=origin/main
    echo "── Channel: edge (main)"
  fi

  # Must be on a branch this script follows (a detached HEAD or a side branch would silently
  # pin an old version): the channel's own, or the other one, which a switch of channel or a
  # box installed before the stable channel existed is on.
  CUR=$(git -C "$DIR" symbolic-ref -q --short HEAD || echo DETACHED)
  case "$CUR" in
    main|stable) ;;
    *) if [ "${FORCE_UPDATE:-0}" != "1" ]; then
         echo "ERROR: $DIR is on '$CUR', not 'main' or 'stable'; refusing to install from it." >&2
         echo "  rerun with FORCE_UPDATE=1 to switch to $TARGET_BRANCH (your branch/commit is kept)" >&2
         exit 1
       fi ;;
  esac

  # Invariant: if this script completes, you ARE on the latest origin/main.
  # Only TRACKED modifications can make the checkout stale or altered. Untracked
  # files cannot, and `git reset --hard` below leaves them alone, so blocking on
  # them turned an unrelated notes file into a failed install and, under
  # FORCE_UPDATE, swept it into a stash the user never asked for.
  if [ -n "$(git -C "$DIR" status --porcelain --untracked-files=no)" ]; then
    if [ "${FORCE_UPDATE:-0}" = "1" ]; then
      echo "── FORCE_UPDATE=1: stashing your tracked changes (recoverable: git stash list)"
      git -C "$DIR" stash push -m "get.sh auto-stash before update"
    else
      echo "ERROR: $DIR has modified tracked files, refusing to install a stale or altered version." >&2
      # sed -n reads to the end where head would close the pipe early: under pipefail, git's SIGPIPE ended the
      # script with 141 before the advice below (code review, 2026-11-01)
      git -C "$DIR" status --short --untracked-files=no | sed -n '1,10p' >&2
      echo "Fix with ONE of:" >&2
      echo "  keep your changes aside :  FORCE_UPDATE=1  then rerun this command (recover later: git -C $DIR stash pop)" >&2
      # not `git clean -fd` as well: it deletes the untracked files this script keeps
      # on purpose (found in review, 2026-09-24)
      echo "  discard your changes    :  git -C $DIR checkout -- .  then rerun" >&2
      exit 1
    fi
  fi
  # A file of the clone that git does not track (untracked or ignored) is written over or removed where the release
  # puts something of its own: a note, an AGENTS.md an agent wrote, a file a later release adds. It is copied aside
  # first, under a folder of the user's own, and said; a copy that fails stops the update before its reset (third
  # review of v1.40.8, 2026-11-01; fixed in v1.45.0). Only the paths a target adds to HEAD can collide, each one
  # three ways: a file of the user's at that path, a folder of the user's there (the release adds a file named
  # drafts, the user has drafts/idea.txt), and a file of the user's where the release needs a folder (the release
  # adds extra/notes.md, the user has a file named extra). Comparing whole lists missed the last two, and listed every
  # ignored file of the clone, a .venv-mirror/ of thousands (code review of v1.45.0). Names go NUL-separated (-z,
  # find -print0), so a name with a newline or a quote is found as it is.
  NKEPT=0; KEPTLIST=""; KEPT=""; SAID_KEPT=""; REPLACED_DIRS=()
  keep_aside() {   # $1: a path of the clone; $2 "untracked" when the caller read it among git's untracked files
    if [ "${2:-}" != untracked ]; then
      git --literal-pathspecs -C "$DIR" ls-files --error-unmatch -- "$1" >/dev/null 2>&1 && return 0   # the release's
    fi
    case $'\n'"$KEPTLIST" in *$'\n'"$1"$'\n'*) return 0 ;; esac                                     # copied already
    # a folder of its own each time: two updates within one second shared one, the second's copies over the first's
    [ -n "$KEPT" ] || KEPT=$(mkdir -p "$HOME/.local/share/alzeo" &&
                             mktemp -d "$HOME/.local/share/alzeo/kept-before-update-$(date +%Y%m%d-%H%M%S)-XXXXXX") || {
      echo "ERROR: could not make a folder under $HOME/.local/share/alzeo to copy your files aside: the update stops" \
           "before its reset, and no file of yours was written over." >&2; exit 1; }
    mkdir -p "$KEPT/$(dirname "$1")" && cp -pP "$DIR/$1" "$KEPT/$1" || {
      echo "ERROR: could not copy $DIR/$1 aside before the update would write over it: the update stops before its" \
           "reset, and no file of yours was written over." >&2; exit 1; }
    NKEPT=$((NKEPT + 1)); KEPTLIST+="$1"$'\n'
  }
  keep_aside_against() {   # each target the update will write: the paths it has and HEAD lacks
    local target p f d
    local -a ups
    for target in "$@"; do
      while IFS= read -r -d '' p; do
        # each folder above it first, from the top: a file or a link of the user's where the release needs a folder is
        # what the reset replaces, and what lies past a link is not the clone's (code review of v1.45.0)
        ups=(); d=$p
        while [ "${d%/*}" != "$d" ]; do d=${d%/*}; ups=("$d" ${ups[@]+"${ups[@]}"}); done
        for d in ${ups[@]+"${ups[@]}"}; do
          if [ -L "$DIR/$d" ] || [ -f "$DIR/$d" ]; then keep_aside "$d"; continue 2; fi
        done
        if [ -L "$DIR/$p" ] || [ -f "$DIR/$p" ]; then keep_aside "$p"
        elif [ -d "$DIR/$p" ]; then
          REPLACED_DIRS+=("$p/")
          # the user's files in it, untracked or ignored, as git lists them: no tracked one, no git process each
          while IFS= read -r -d '' f; do keep_aside "$f" untracked; done < <(
            git --literal-pathspecs -C "$DIR" ls-files -z --others -- "$p")
        fi
      done < <(git -C "$DIR" diff -z --name-only --no-renames --diff-filter=A HEAD "$target")
    done
    say_kept
  }
  say_kept() {   # said once, as soon as the copies are made: a switch that fails after them still leaves this said
    [ "$NKEPT" -gt 0 ] && [ -z "$SAID_KEPT" ] || return 0
    SAID_KEPT=1
    echo "── $NKEPT file(s) of yours where this release puts its own, copied aside first to:"
    echo "     $KEPT"
    printf '%s' "$KEPTLIST" | sed -n '1,5s/^/     /p'
    [ "$NKEPT" -le 5 ] || echo "     ... and $((NKEPT - 5)) more, in that folder"
  }
  # Off the target branch, the switch comes after the stash, and a detached HEAD's
  # commit is put on a branch first. Switched first, a change that conflicted with the
  # target ended the run (rc 128), and a commit made on a detached HEAD was left on no branch
  # (found in review, 2026-09-24). It runs under FORCE_UPDATE from a side branch, and without
  # it for a change of channel or a box on main taking the stable one for the first time.
  if [ "$CUR" != "$TARGET_BRANCH" ]; then
    if [ "$CUR" = "DETACHED" ]; then
      BK="backup-$(git -C "$DIR" rev-parse --short HEAD)"
      git -C "$DIR" branch -f "$BK" >/dev/null
      echo "── FORCE_UPDATE=1: the detached commit is kept on branch '$BK'"
    fi
    # a stable branch that holds commits of its own (past origin/main, which a stable clone at
    # its tag never has) would be moved by the checkout below: they are kept on a branch first
    if [ "$TARGET_BRANCH" = stable ] && git -C "$DIR" rev-parse -q --verify stable >/dev/null 2>&1; then
      if [ "$(git -C "$DIR" rev-list --count origin/main..stable 2>/dev/null || echo 1)" != "0" ]; then
        BK="backup-$(git -C "$DIR" rev-parse --short stable)"
        git -C "$DIR" branch -f "$BK" stable >/dev/null
        echo "── the branch 'stable' held commits of its own: kept on branch '$BK'"
      fi
    fi
    # the checkout below writes the target branch over ignored files too, and the reset then
    # the release: both are looked at from here, before either (code review of v1.45.0)
    if git -C "$DIR" rev-parse -q --verify "$TARGET_BRANCH" >/dev/null 2>&1; then
      keep_aside_against "$TARGET_REF" "$TARGET_BRANCH"
    else
      keep_aside_against "$TARGET_REF"
    fi
    echo "── switching from '$CUR' to $TARGET_BRANCH (your branch is kept)"
    # forced from a side branch, or a change of channel: the tracked changes are stashed above
    # and every file of the user's in its way copied aside
    if [ "$TARGET_BRANCH" = main ]; then
      if git -C "$DIR" rev-parse -q --verify main >/dev/null 2>&1; then git -C "$DIR" checkout -qf main
      else git -C "$DIR" checkout -qfb main origin/main; fi
    else
      git -C "$DIR" checkout -qB stable "$TARGET_REF"
    fi
  fi
  AHEAD=$(git -C "$DIR" rev-list --count origin/main..HEAD 2>/dev/null || echo 1)
  if [ "$AHEAD" != "0" ]; then
    if [ "${FORCE_UPDATE:-0}" = "1" ]; then
      BK="backup-$(git -C "$DIR" rev-parse --short HEAD)"
      git -C "$DIR" branch -f "$BK" >/dev/null
      echo "── FORCE_UPDATE=1: local commits kept on branch '$BK', resetting to $TARGET_BRANCH"
    else
      echo "ERROR: $DIR has local commits not on origin/main, refusing to install a stale version." >&2
      echo "  keep your commits aside :  FORCE_UPDATE=1  then rerun (they stay on a backup branch)" >&2
      exit 1
    fi
  fi
  # Copied aside, then said: here when the clone is on the target branch; off it, before the switch above.
  [ "$CUR" != "$TARGET_BRANCH" ] || keep_aside_against "$TARGET_REF"
  # Untracked files are kept, but say so: neither those copied aside, which the release's own replace, nor a folder
  # the release puts a file in place of (drafts/, removed by the reset: code review of v1.45.0). A folder untracked
  # whole is named once (--directory): a .venv-mirror/ in the clone, as MIRROR.md makes one, listed its thousands of
  # files. Names read NUL-separated, as the copy reads them: git quotes a name with a quote, a backslash or a control
  # character in its listing, and such a name copied aside was said kept in place too.
  NUNTRACKED=0; UNTRACKED=""
  while IFS= read -r -d '' u; do
    case $'\n'"$KEPTLIST" in *$'\n'"$u"$'\n'*) continue ;; esac
    gone=0
    for r in ${REPLACED_DIRS[@]+"${REPLACED_DIRS[@]}"}; do case "$u" in "$r"*) gone=1; break ;; esac; done
    [ "$gone" = 0 ] || continue
    NUNTRACKED=$((NUNTRACKED + 1)); [ "$NUNTRACKED" -gt 5 ] || UNTRACKED+="     $u"$'\n'
  done < <(git -C "$DIR" ls-files -z --others --exclude-standard --directory)
  if [ "$NUNTRACKED" -gt 0 ]; then
    echo "── Untracked files kept in place (not part of the update):"
    printf '%s' "$UNTRACKED"
    [ "$NUNTRACKED" -le 5 ] || echo "     ... and $((NUNTRACKED - 5)) more"
  fi
  # Tree verified clean and no local commits (or backed up): hard sync is lossless
  # and, unlike ff-only, immune to shallow-clone ancestry gaps.
  git -C "$DIR" reset -q --hard "$TARGET_REF"
  echo "── At: $(git -C "$DIR" log -1 --format='%h %s')"

  cd "$DIR"
  # exec from a real file. stdin is still the pipe this script came through (read to
  # its end by now), so nothing in install.sh may read from it; sudo prompts on the tty
  # it opens itself. The channel flags never reach install.sh: they were consumed above.
  exec bash ./install.sh ${ARGS[@]+"${ARGS[@]}"}
}

main "$@"
