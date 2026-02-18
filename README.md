<p align="center">
    <a href="https://github.com/3b1b/manim">
        <img src="https://raw.githubusercontent.com/3b1b/manim/master/logo/cropped.png">
    </a>
</p>

[![pypi version](https://img.shields.io/pypi/v/manimgl?logo=pypi)](https://pypi.org/project/manimgl/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](http://choosealicense.com/licenses/mit/)
[![Manim Subreddit](https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=ff4301&label=reddit&logo=reddit)](https://www.reddit.com/r/manim/)
[![Manim Discord](https://img.shields.io/discord/581738731934056449.svg?label=discord&logo=discord)](https://discord.com/invite/bYCyhM9Kz2)
[![docs](https://github.com/3b1b/manim/workflows/docs/badge.svg)](https://3b1b.github.io/manim/)

Manim is an engine for precise programmatic animations, designed for creating explanatory math videos.

Note, there are two versions of manim.  This repository began as a personal project by the author of [3Blue1Brown](https://www.3blue1brown.com/) for the purpose of animating those videos, with video-specific code available [here](https://github.com/3b1b/videos).  In 2020 a group of developers forked it into what is now the [community edition](https://github.com/ManimCommunity/manim/), with a goal of being more stable, better tested, quicker to respond to community contributions, and all around friendlier to get started with. See [this page](https://docs.manim.community/en/stable/faq/installation.html#different-versions) for more details.

## Installation
> [!Warning]
> **WARNING:** These instructions are for ManimGL _only_. Trying to use these instructions to install [Manim Community/manim](https://github.com/ManimCommunity/manim) or instructions there to install this version will cause problems. You should first decide which version you wish to install, then only follow the instructions for your desired version.

## Notes

## This is afork (EnriqueDiazO/manim-phd)

This repository is a **fork of ManimGL (3b1b/manim)** used as my working base to build and maintain **PhD / doctoral research animations** (scenes, experiments, and figure reconstructions).

The upstream project is here: https://github.com/3b1b/manim

### Install (Ubuntu + Poetry + pyenv)

> **Recommended:** use Python **3.10.x** for ManimGL v1.7.2 compatibility.

#### 1) System dependencies (Ubuntu)
Install OpenGL / FFmpeg / Pango / Cairo headers, etc.:

```bash
./scripts/install_ubuntu_deps.sh
```

#### 2) Create Poetry env using pyenv (Python 3.10)

```bash
pyenv install -s 3.10.14
pyenv shell 3.10.14

poetry env use "$(pyenv which python)"
poetry lock
poetry install

```

#### 3) Install this fork in editable mode (inside Poetry env)
```bash
poetry run pip install -e .
```

#### 4) Smoke test
```bash
poetry run manimgl --version
```

#### 5) Run a scene
```bash
poetry run manimgl example_scenes.py OpeningManimExample
```

### Run scenes interactively (menu / loop)

This fork includes a small helper script to **list all Scene subclasses in a given `.py` file**
and let you **run them repeatedly** by selecting a number or name.

#### Example: run the slides demo with a menu

```bash
poetry run python tools/runloop.py tools/slidesdemo.py
```


Look through the [example scenes](https://3b1b.github.io/manim/getting_started/example_scenes.html) to see examples of the library's syntax, animation types and object types. In the [3b1b/videos](https://github.com/3b1b/videos) repo, you can see all the code for 3blue1brown videos, though code from older videos may not be compatible with the most recent version of manim. The readme of that repo also outlines some details for how to set up a more interactive workflow, as shown in [this manim demo video](https://www.youtube.com/watch?v=rbu7Zu5X1zI) for example.

When running in the CLI, some useful flags include:
* `-w` to write the scene to a file
* `-o` to write the scene to a file and open the result
* `-s` to skip to the end and just show the final frame.
    * `-so` will save the final frame to an image and show it
* `-n <number>` to skip ahead to the `n`'th animation of a scene.
* `-f` to make the playback window fullscreen

Take a look at custom_config.yml for further configuration.  To add your customization, you can either edit this file, or add another file by the same name "custom_config.yml" to whatever directory you are running manim from.  For example [this is the one](https://github.com/3b1b/videos/blob/master/custom_config.yml) for 3blue1brown videos.  There you can specify where videos should be output to, where manim should look for image files and sounds you want to read in, and other defaults regarding style and video quality.

### Documentation
Documentation is in progress at [3b1b.github.io/manim](https://3b1b.github.io/manim/). And there is also a Chinese version maintained by [**@manim-kindergarten**](https://manim.org.cn): [docs.manim.org.cn](https://docs.manim.org.cn/) (in Chinese).

[manim-kindergarten](https://github.com/manim-kindergarten/) wrote and collected some useful extra classes and some codes of videos in [manim_sandbox repo](https://github.com/manim-kindergarten/manim_sandbox).


## Contributing
Is always welcome.  As mentioned above, the [community edition](https://github.com/ManimCommunity/manim) has the most active ecosystem for contributions, with testing and continuous integration, but pull requests are welcome here too.  Please explain the motivation for a given change and examples of its effect.


## License
This project falls under the MIT license.
