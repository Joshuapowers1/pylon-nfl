# Publishing pylon-nfl

The repository is built and committed here: 27 commits, a clean working tree,
and `docs/index.html` is the site. What is left needs your GitHub account,
which is the one part that cannot be done for you.

## The easy way

**Double-click `PUBLISH.command` in this folder.**

It opens a Terminal window and does the whole thing: installs the GitHub CLI if
you do not have it, signs you in if you are not, creates `pylon-nfl` as a public
repository, pushes, switches on GitHub Pages from `main` and `/docs`, and sets
the description and topics. It never deletes anything.

If macOS refuses to run it because it was downloaded, right-click the file and
choose Open, then Open again on the warning. Or run this once in Terminal:

```bash
chmod +x ~/Documents/pylon-nfl/PUBLISH.command
xattr -d com.apple.quarantine ~/Documents/pylon-nfl/PUBLISH.command 2>/dev/null
```

Two or three minutes after it finishes, the site is at
**https://joshuapowers1.github.io/pylon-nfl/**

---

## The manual way

**1. Make the empty repository** at https://github.com/new

- Name: `pylon-nfl`
- Public
- Do **not** add a README, a .gitignore or a license. This repository already
  has all three and an auto-created file would collide on the first push.

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

---

## Once it is up

**Point the repository at the site.** On the repo home page, the gear beside
"About", tick "Use your GitHub Pages website". `PUBLISH.command` does this for
you.

**Check the roster import on the live site.** Open My team, type a Sleeper
username, confirm it returns leagues. Then try a public ESPN league id. Neither
can run inside an embedded preview, because that blocks outside requests, so
the hosted copy is the first place they are real.

**Two workflows will appear under Actions.** `tests` runs on every push and
includes the contrast audit. `weekly in-season update` is on a Tuesday
schedule. GitHub disables scheduled workflows on repositories with no activity
for sixty days, so if the site ever goes stale in the offseason, that is why.
