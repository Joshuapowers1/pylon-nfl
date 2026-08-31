# Publishing pylon-nfl

The repository is already built and committed here. Twenty-four commits, a clean
working tree, and `docs/index.html` is the built site. What follows is the part
that needs your GitHub account, which is why it is a paste rather than something
that could be done for you.

Run this from **Terminal on your Mac**, not from anywhere else, because that is
where your GitHub credentials live.

---

## One paste, if you have the GitHub CLI

Check first with `gh --version`. If it prints a version, this is the whole job:

```bash
cd ~/Documents/pylon-nfl

gh repo create pylon-nfl --public --source=. --remote=origin --push \
  --description "NFL projection engine: opponent-adjusted ridge model on nine seasons of play-by-play, out-of-sample calibrated shrinkage, 25,000-season Monte Carlo, a de-vigged market board, and a roster import that turns a Sleeper username into a set lineup."

gh api -X POST repos/Joshuapowers1/pylon-nfl/pages \
  -f 'source[branch]=main' -f 'source[path]=/docs'
```

The last command turns on GitHub Pages. Give it two or three minutes, then the
site is at **https://joshuapowers1.github.io/pylon-nfl/**

If you do not have `gh`, install it with `brew install gh` and run `gh auth login`
first, or use the manual route below.

---

## Manual route

**1. Make the empty repository.** Go to https://github.com/new

- Name: `pylon-nfl`
- Public
- Do **not** add a README, a .gitignore or a license. The repository already has
  all three, and an auto-created file would collide on the first push.

**2. Push.**

```bash
cd ~/Documents/pylon-nfl
git remote add origin https://github.com/Joshuapowers1/pylon-nfl.git
git branch -M main
git push -u origin main
```

**3. Turn on Pages.** Settings, then Pages in the left sidebar.

- Source: **Deploy from a branch**
- Branch: **main**, folder: **/docs**
- Save

Two or three minutes later the site is live at
**https://joshuapowers1.github.io/pylon-nfl/**

---

## Worth doing once it is up

**Add the link to the repository header.** On the repo home page, the gear icon
beside "About", tick "Use your GitHub Pages website".

**Add topics** so it turns up in search: `nfl`, `sports-analytics`,
`monte-carlo`, `fantasy-football`, `ridge-regression`, `python`,
`data-visualization`.

**Check that the roster import works on the live site.** Open the My team page,
type a Sleeper username, and confirm the lookup returns leagues. That path cannot
run inside the embedded preview, because its content policy blocks outside
requests, so the hosted copy is the first place it is real.

**The Actions tab will show one workflow already.** `tests` runs on every push.
`weekly in-season update` is on a Tuesday schedule and will not fire until then.
GitHub disables scheduled workflows on repositories with no activity for sixty
days, so if the site ever goes stale in the offseason, that is why.
