# python-options (macOS)

This repo contains a simple options screener (options.py). These instructions are tailored for macOS.

## 1) Install Python 3 (macOS)

You can either use the helper script or install via Homebrew directly.

- Using the helper script (recommended):

  chmod +x install_python.sh
  ./install_python.sh

- Or manually with Homebrew (https://brew.sh):

  brew install python

After installation, verify:

  python3 --version
  pip3 --version

## 2) Create and activate a virtual environment

From the project folder:

  python3 -m venv .venv
  source .venv/bin/activate

## 3) Install project dependencies

With the virtual environment active:

  python3 -m pip install -r requirements.txt

## 4) Run the screener

Example:

  python options.py --tickers AAPL,MSFT,NVDA,SPY --min_oi 200 --min_vol 50

Outputs:
- screener_results.csv
- screener_results_backtest.csv
- plots/<TICKER>_support_resistance.png

## Quick start (one command)

If you prefer automation, run:

  bash setup_venv.sh

This will create a virtual environment and install requirements using `python -m pip`.

## Troubleshooting (macOS)

- zsh: command not found: python
  - Use `python3` instead. Example: `python3 -m pip install -r requirements.txt`
  - If `python3` is not found, install via `./install_python.sh` or `brew install python`.
  - Optional (zsh): add an alias so `python` maps to `python3`:
    - echo 'alias python=python3' >> ~/.zshrc && source ~/.zshrc

- zsh: command not found: pip
  - Use the module form: `python3 -m pip install -r requirements.txt`
  - Ensure your virtual environment is activated: `source .venv/bin/activate`
  - If you recently installed Homebrew/Python, open a new terminal so PATH updates, or run `hash -r` in zsh.

- Apple Silicon build tools
  - If you encounter build errors for scipy or numpy on Apple Silicon, ensure Command Line Tools are installed:

    xcode-select --install

Notes:
- These instructions focus on macOS. Linux/Windows setup has been omitted intentionally to keep this README mac-specific.

# python-options (macOS)

This repo contains a simple options screener (options.py). These instructions are tailored for macOS.

## 1) Install Python 3 (macOS)

You can either use the helper script or install via Homebrew directly.

- Using the helper script (recommended):

  chmod +x install_python.sh
  ./install_python.sh

- Or manually with Homebrew (https://brew.sh):

  brew install python

After installation, verify:

  python3 --version
  pip3 --version

## 2) Create and activate a virtual environment

From the project folder:

  python3 -m venv .venv
  source .venv/bin/activate

## 3) Install project dependencies

With the virtual environment active:

  python3 -m pip install -r requirements.txt

## 4) Run the screener

Example:

  python options.py --tickers AAPL,MSFT,NVDA,SPY --min_oi 200 --min_vol 50

Outputs:
- screener_results.csv
- screener_results_backtest.csv
- plots/<TICKER>_support_resistance.png

## Quick start (one command)

If you prefer automation, run:

  bash setup_venv.sh

This will create a virtual environment and install requirements using `python -m pip`.

## Troubleshooting (macOS)

- zsh: command not found: python
  - Use `python3` instead. Example: `python3 -m pip install -r requirements.txt`
  - If `python3` is not found, install via `./install_python.sh` or `brew install python`.
  - Optional (zsh): add an alias so `python` maps to `python3`:
    - echo 'alias python=python3' >> ~/.zshrc && source ~/.zshrc

- zsh: command not found: pip
  - Use the module form: `python3 -m pip install -r requirements.txt`
  - Ensure your virtual environment is activated: `source .venv/bin/activate`
  - If you recently installed Homebrew/Python, open a new terminal so PATH updates, or run `hash -r` in zsh.

- Apple Silicon build tools
  - If you encounter build errors for scipy or numpy on Apple Silicon, ensure Command Line Tools are installed:

    xcode-select --install

Notes:
- These instructions focus on macOS. Linux/Windows setup has been omitted intentionally to keep this README mac-specific.

---

## Git: Push this project to a new GitHub repository (macOS)

If you created a new, empty GitHub repo and push fails with commands like:

  git remote add origin git@github.com:<USER>/<REPO>.git
  git branch -M main
  git push -u origin main

use the checklist below.

### A) Initialize and make the first commit

From the project root:

  git init
  git add .
  git commit -m "Initial commit"
  git branch -M main

If you get "src refspec main does not match any", it means there are no commits yet or the branch doesn't exist. Make the commit first, then rename/create the branch as above.

### B) Connect the correct remote

Use SSH (recommended) or HTTPS.

- SSH (requires an SSH key added to your GitHub account):

  git remote add origin git@github.com:<USER>/<REPO>.git

  If "remote origin already exists":

  git remote set-url origin git@github.com:<USER>/<REPO>.git

- HTTPS (passwordless with a GitHub token):

  git remote add origin https://github.com/<USER>/<REPO>.git

  If it already exists:

  git remote set-url origin https://github.com/<USER>/<REPO>.git

### C) Verify your SSH setup (if using SSH)

Test your SSH connection:

  ssh -T git@github.com

If you see "Permission denied (publickey)", create and add an SSH key:

  ssh-keygen -t ed25519 -C "your_email@example.com"
  eval "$(ssh-agent -s)"
  ssh-add --apple-use-keychain ~/.ssh/id_ed25519
  pbcopy < ~/.ssh/id_ed25519.pub   # then add the copied key at https://github.com/settings/keys
  ssh -T git@github.com            # test again; you should see a success message

Tip (macOS Keychain persistence):

  git config --global credential.helper osxkeychain

### D) Push

  git push -u origin main

If the remote has an auto-created README or .gitignore and you see "Updates were rejected because the remote contains work that you do not have":

  git pull origin main --allow-unrelated-histories
  git push -u origin main

Alternatively, if you prefer a rebase:

  git fetch origin
  git rebase origin/main
  git push -u origin main

### E) Common errors and fixes

- src refspec main does not match any
  - Root cause: No commit or no branch named main.
  - Fix: Make an initial commit, run `git branch -M main`, then push.

- remote origin already exists
  - Fix: `git remote set-url origin git@github.com:<USER>/<REPO>.git` (or HTTPS URL), or `git remote remove origin && git remote add origin <URL>`.

- Permission denied (publickey)
  - Fix: Ensure your SSH key is added to ssh-agent and to GitHub (see step C). Then `ssh -T git@github.com` to verify.

- Updates were rejected / non-fast-forward
  - Fix: `git pull --rebase origin main` (or `git pull origin main --allow-unrelated-histories` if the remote is unrelated), then push again.

- Protected branch / required status checks
  - Root cause: Branch protection rules in the GitHub repo.
  - Fix: Temporarily relax protection or open a PR from a feature branch.

- Wrong default branch name on GitHub
  - If GitHub default branch is `main` but you're pushing `master` (or vice versa), rename locally: `git branch -M main` (or `master`) and set the same default on GitHub settings.

With these steps, the standard sequence should work:

  git add .
  git commit -m "Initial commit"
  git branch -M main
  git remote add origin git@github.com:<USER>/<REPO>.git
  git push -u origin main


---

## One-command push script

If you just want to automate the whole push process, use the included script:

  bash push_to_new_repo.sh git@github.com:<USER>/<REPO>.git

Or with HTTPS:

  bash push_to_new_repo.sh https://github.com/<USER>/<REPO>.git

What it does:
- Initializes the repo if needed and creates the first commit if none exists.
- Ensures your branch is named main.
- Adds or updates the origin remote to the URL you provide.
- Verifies SSH connectivity when using an SSH URL (informational check).
- Pushes to origin main and automatically handles common issues (unrelated histories, non-fast-forward) by retrying with safe steps.

Run it from the project root. On failure it prints actionable tips.
