#!/bin/bash
# Double-click this file to put Pylon on GitHub as a public repository and
# turn on the hosted site.
#
# It does four things and nothing else: installs the GitHub CLI if it is
# missing, signs you in if you are not, creates the repository and pushes what
# is already committed here, and switches on GitHub Pages. It never deletes
# anything and it never touches a repository other than pylon-nfl.
set -u

cd "$(dirname "$0")" || exit 1
# macOS still ships bash 3.2, so nothing in here uses bash 4 syntax.
REPO="pylon-nfl"
lower(){ printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

say(){ printf '\n\033[1m%s\033[0m\n' "$*"; }
die(){ printf '\n\033[31m%s\033[0m\n\n' "$*"; printf 'Press return to close.'; read -r _; exit 1; }

say "Publishing Pylon to GitHub"
printf 'Folder: %s\n' "$PWD"

[ -d .git ] || die "This folder is not a git repository. Run this from inside pylon-nfl."

# ---- 1. the GitHub CLI ----------------------------------------------------
if ! command -v gh >/dev/null 2>&1; then
  say "The GitHub CLI is not installed."
  if command -v brew >/dev/null 2>&1; then
    printf 'Install it with Homebrew now? [y/N] '
    read -r ans
    case "$ans" in
      [yY]*) brew install gh || die "Homebrew could not install gh." ;;
      *) die "Nothing was changed. Install gh, then run this again." ;;
    esac
  else
    die "Install Homebrew from https://brew.sh, then run: brew install gh"
  fi
fi

# ---- 2. sign in -----------------------------------------------------------
if ! gh auth status >/dev/null 2>&1; then
  say "Signing you in to GitHub. Follow the prompts in this window."
  gh auth login || die "Sign-in did not complete. Nothing was changed."
fi

WHO="$(gh api user --jq .login 2>/dev/null)"
[ -n "$WHO" ] || die "Signed in, but GitHub did not return your username."
printf 'Signed in as %s\n' "$WHO"

# ---- 3. create and push ---------------------------------------------------
if gh repo view "$WHO/$REPO" >/dev/null 2>&1; then
  say "$WHO/$REPO already exists. Pushing to it."
  git remote get-url origin >/dev/null 2>&1 || \
    git remote add origin "https://github.com/$WHO/$REPO.git"
  git branch -M main
  git push -u origin main || die "Push failed. The message above says why."
else
  say "Creating $WHO/$REPO as a public repository and pushing."
  gh repo create "$REPO" --public --source=. --remote=origin --push \
    --description "NFL projection engine by Joshua Powers: opponent-adjusted ridge model on nine seasons of play-by-play, out-of-sample calibrated shrinkage, 25,000-season Monte Carlo, a de-vigged market board, and a fantasy roster import for Sleeper and ESPN." \
    || die "Could not create the repository. The message above says why."
fi

# ---- 4. turn on the site --------------------------------------------------
say "Turning on GitHub Pages, served from the docs folder on main."
if gh api "repos/$WHO/$REPO/pages" >/dev/null 2>&1; then
  gh api -X PUT "repos/$WHO/$REPO/pages" \
    -f 'source[branch]=main' -f 'source[path]=/docs' >/dev/null 2>&1 \
    && printf 'Pages was already on; the source is set to main /docs.\n'
else
  gh api -X POST "repos/$WHO/$REPO/pages" \
    -f 'source[branch]=main' -f 'source[path]=/docs' >/dev/null 2>&1 \
    && printf 'Pages is on.\n' \
    || printf 'Could not switch Pages on from here. Do it in Settings, Pages: deploy from branch main, folder /docs.\n'
fi

# Description, homepage and topics, so the repository page reads as a project.
gh repo edit "$WHO/$REPO" --homepage "https://$(lower "$WHO").github.io/$REPO/" \
  --add-topic nfl --add-topic sports-analytics --add-topic fantasy-football \
  --add-topic monte-carlo --add-topic ridge-regression --add-topic python \
  --add-topic data-visualization >/dev/null 2>&1

say "Done."
printf 'Code:  https://github.com/%s/%s\n' "$WHO" "$REPO"
printf 'Site:  https://%s.github.io/%s/\n' "$(lower "$WHO")" "$REPO"
printf '\nThe site takes two or three minutes to build the first time.\n'
printf 'Press return to close this window.'
read -r _
