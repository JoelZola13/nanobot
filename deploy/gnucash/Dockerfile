FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install GnuCash from package manager + virtual display + noVNC
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnucash \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    dbus-x11 \
    at-spi2-core \
    libgtk-3-0 \
    adwaita-icon-theme \
    fonts-dejavu \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash gnucash

# noVNC setup — symlink so websockify can find it
RUN ln -sf /usr/share/novnc/vnc_lite.html /usr/share/novnc/index.html

# Supervisor config to run Xvfb, x11vnc, websockify, and GnuCash together
COPY supervisord.conf /etc/supervisor/conf.d/gnucash.conf

# GnuCash settings to suppress first-run wizard
RUN mkdir -p /home/gnucash/.config/gnucash && \
    chown -R gnucash:gnucash /home/gnucash

EXPOSE 6080

WORKDIR /home/gnucash

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/gnucash.conf"]
