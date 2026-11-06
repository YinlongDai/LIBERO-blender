#!/bin/bash
# Download the official Blender 4.4.0 Linux build into .cache/blender, the path configs/paths.yaml uses.
# The datasets were rendered with this build, so EEVEE renders match them exactly.
set -e
VERSION=4.4.0
NAME=blender-$VERSION-linux-x64
URL=https://download.blender.org/release/Blender${VERSION%.*}
cd "$(dirname "$0")/.."
mkdir -p .cache/blender
cd .cache/blender
if [ ! -x $NAME/blender ]; then
    curl -fLO $URL/$NAME.tar.xz
    curl -fsL $URL/blender-$VERSION.sha256 | grep " $NAME.tar.xz$" | sha256sum -c -
    tar -xf $NAME.tar.xz
    rm $NAME.tar.xz
fi
$NAME/blender --version | head -1
