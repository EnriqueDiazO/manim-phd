#!/usr/bin/env bash
set -euo pipefail

sudo apt update

# Core runtime deps
sudo apt install -y \
  ffmpeg sox \
  pkg-config \
  git curl \
  libgl1-mesa-dev libglu1-mesa-dev \
  libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev

# Text / cairo / pango (muy típico para ManimGL)
sudo apt install -y \
  libcairo2-dev \
  libpango1.0-dev \
  libgirepository1.0-dev \
  gir1.2-pango-1.0 \
  fonts-dejavu fonts-freefont-ttf

# (Opcional) LaTeX si vas a usar TexMobject
sudo apt install -y texlive-latex-extra texlive-fonts-extra texlive-science
