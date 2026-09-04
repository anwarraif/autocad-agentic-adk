# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# cad-api — FastAPI + ezdxf.
#
# Multi-stage: wheels are built in `builder`, only the site-packages tree is
# copied into the runtime image. Keeps the runtime free of compilers.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LibreDWG: DWG -> DXF, compiled here so the pipeline accepts .dwg directly.
#
# This removes the last manual step in the pipeline. ODA File Converter is the
# usual answer but needs registration and an X server even when headless;
# LibreDWG needs neither. There is no apt package (checked on Debian trixie),
# so it is built from source.
#
# CFLAGS="-O2 -w": 0.13.3 predates GCC 14, which promotes several of its own
# warnings to errors. Relaxing the warning level is the upstream workaround and
# does not change the generated code.
#
# Measured on the whole sample set: 13 of 16 sample DWGs open straight away, a
# 14th (truetype) after the wrapped-line repair in app/convert.py, and
# Janadriyah converts in 324 s to a document identical in content to the one
# converted outside (20,328 modelspace entities, 292 layers, 261 audit errors).
# The two blocks_and_tables files still fail on a deeper LibreDWG defect --
# see docs/TECH-DEBT.md T-4.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS libredwg

RUN apt-get update  && apt-get install -y --no-install-recommends       build-essential autoconf automake libtool pkg-config texinfo       git ca-certificates perl  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch 0.13.3       https://github.com/LibreDWG/libredwg.git /src/libredwg  && cd /src/libredwg  && sh autogen.sh  && ./configure --disable-bindings --disable-shared --enable-static       CFLAGS="-O2 -w"  && make -j"$(nproc)"  && make install


FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

COPY --from=builder /install /usr/local

# Just the converter binaries; the build toolchain stays out of the runtime.
COPY --from=libredwg /usr/local/bin/dwg2dxf /usr/local/bin/dwg2dxf
COPY --from=libredwg /usr/local/bin/dwgread /usr/local/bin/dwgread

# ezdxf needs real font files to render TEXT/MTEXT. Without them it logs
# "no fonts available, not even fallback fonts" and silently drops every
# annotation from the SVG -- which on a CAD drawing is most of the meaning.
# DejaVu is ~2 MB and covers the Latin text in these drawings.
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# ODA File Converter: preferred DWG -> DXF converter; LibreDWG stays as the
# fallback so every file that converted before still converts. Why two:
# Autodesk's web-viewer extractor crashes (exit 0xC0000409) on LibreDWG's
# Janadriyah output while the original DWG uploads fine — the conversion is
# the difference — and ODA writes Autodesk-faithful DXF. Freeware; EULA
# accepted by the operator. The guestfiles URL 301s to a signed S3 link,
# hence curl -L. QT_QPA_PLATFORM=offscreen lets the Qt app run without an X
# server; xvfb covers ODA builds that reject the offscreen platform.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && curl -fsSL -o /tmp/oda.deb \
      "https://www.opendesign.com/guestfiles/get?filename=ODAFileConverter_QT6_lnxX64_8.3dll_27.1.deb" \
 && apt-get install -y --no-install-recommends /tmp/oda.deb \
      xvfb xauth \
      libxkbcommon0 libxkbcommon-x11-0 libgl1 libegl1 libopengl0 \
      libfontconfig1 libglib2.0-0 libdbus-1-3 \
      libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 \
      libxcb-xkb1 libxcb-image0 libxcb-render-util0 libxcb-randr0 \
 && rm /tmp/oda.deb \
 && apt-get purge -y curl \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# Non-root: the image never needs to write outside /tmp and the mounted volumes.
RUN useradd --create-home --uid 10001 cad

# /data/svg must exist in the image AND be owned by `cad`. Docker seeds a fresh
# named volume from the image directory, ownership included; if the path did
# not exist here the volume would be created root-owned and the non-root
# process could not write its render cache.
RUN mkdir -p /data/svg /data/dxf && chown -R cad:cad /data

WORKDIR /app
COPY --chown=cad:cad app ./app

USER cad

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
