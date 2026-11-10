# What the slim Open Terminal image lacks for Chrome mode, the Desktop tab with its sound and
# notebook execution, added on top of it. The package lists follow the release's own Dockerfile.
#
#   docker build --build-arg BASE=owui-tests/open-terminal:v0.17.7 \
#     -t owui-tests/open-terminal-full:v0.17.7 - < harness/open_terminal_full.Dockerfile
ARG BASE
FROM ${BASE}

# Chrome mode: chromium, and xrandr for its headed display. Desktop: the TigerVNC X server, Xfce
# and the tools Open Terminal drives it with. Desktop sound: PulseAudio and ffmpeg.
RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium fonts-dejavu \
        tigervnc-standalone-server xfce4 dbus-x11 xauth xdotool xclip scrot x11-xserver-utils \
        pulseaudio pulseaudio-utils ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Debian's launcher resets CHROMIUM_FLAGS before reading this directory.
RUN printf '%s\n' \
    'case "${OPEN_TERMINAL_BROWSER_NO_SANDBOX:-false}" in' \
    '  [tT][rR][uU][eE]|1|[yY][eE][sS]) export CHROMIUM_FLAGS="$CHROMIUM_FLAGS --no-sandbox" ;;' \
    'esac' > /etc/chromium.d/open-terminal

# Notebooks: the packages of the `notebooks` extra. The slim image ships without pip.
RUN python -m ensurepip \
    && python -m pip install --no-cache-dir 'nbclient>=0.10.0' 'ipykernel>=6.0.0' \
    && python -m pip uninstall -y pip
