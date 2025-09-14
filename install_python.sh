#!/usr/bin/env bash
# install_python.sh
# Installs Python 3 and pip on macOS and common Linux distributions.
# This script DOES NOT run on Windows; see README for Windows instructions.
# After installing Python, it prints the detected python3 and pip3 versions.

set -euo pipefail

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

OS="$(uname -s)"

if [[ "$OS" == "Darwin" ]]; then
  echo "Detected macOS"
  if ! need_cmd brew; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile" || true
    eval "$(/opt/homebrew/bin/brew shellenv)" || true
  fi
  echo "Installing Python 3 via Homebrew..."
  brew update
  # Use a specific modern Python; fallback to generic python if the formula changes
  if brew info python@3.11 >/dev/null 2>&1; then
    brew install python@3.11
    brew link python@3.11 --force || true
  else
    brew install python
  fi
  PY_BIN="$(command -v python3 || true)"
  PIP_BIN="$(command -v pip3 || true)"

elif [[ "$OS" == "Linux" ]]; then
  echo "Detected Linux"
  if need_cmd apt-get; then
    echo "Using apt-get (Debian/Ubuntu)..."
    sudo apt-get update -y
    sudo apt-get install -y python3 python3-pip
  elif need_cmd dnf; then
    echo "Using dnf (Fedora/RHEL 8+)..."
    sudo dnf install -y python3 python3-pip
  elif need_cmd yum; then
    echo "Using yum (RHEL/CentOS 7)..."
    sudo yum install -y python3 python3-pip
  elif need_cmd pacman; then
    echo "Using pacman (Arch/Manjaro)..."
    sudo pacman -Sy --noconfirm python python-pip
  elif need_cmd zypper; then
    echo "Using zypper (openSUSE)..."
    sudo zypper install -y python3 python3-pip
  else
    echo "Unsupported Linux package manager. Please install Python 3 and pip using your distro's tools."
    exit 1
  fi
  PY_BIN="$(command -v python3 || true)"
  PIP_BIN="$(command -v pip3 || true)"
else
  echo "Unsupported OS: $OS"
  echo "This script supports macOS and Linux. For Windows, install Python from https://www.python.org/downloads/ and ensure 'pip' is added to PATH."
  exit 1
fi

if [[ -n "${PY_BIN:-}" ]]; then
  echo "python3 found at: $PY_BIN"
  "$PY_BIN" --version || true
else
  echo "python3 not found after installation. Please check your PATH."
fi

if [[ -n "${PIP_BIN:-}" ]]; then
  echo "pip3 found at: $PIP_BIN"
  "$PIP_BIN" --version || true
else
  echo "pip3 not found after installation. Please check your PATH."
fi

# Helpful note if only python3 exists
if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  echo "Note: 'python' command not found, but 'python3' is available."
  echo "On macOS/Linux, use 'python3' instead of 'python'."
  echo "Optional (zsh): add an alias so 'python' maps to 'python3':"
  echo "  echo 'alias python=python3' >> ~/.zshrc && source ~/.zshrc"
fi

echo "Done."
