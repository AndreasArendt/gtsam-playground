FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------
# 1. System dependencies
# -----------------------------------------------------------
RUN apt update && apt install -y \
    git cmake build-essential pkg-config \
    libboost-all-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    python3 python3-pip python3-dev python3-venv

# -----------------------------------------------------------
# 2. Python dependencies (GTSAM build + useful dev libs)
# -----------------------------------------------------------
RUN pip install --upgrade pip

# Core libs for GTSAM + plotting + debugging + Jupyter
RUN pip install \
    numpy \
    pyparsing packaging \
    matplotlib \
    debugpy \
    jupyterlab jupyter

# -----------------------------------------------------------
# 3. Clone GTSAM (shallow for smaller image)
# -----------------------------------------------------------
RUN git clone --depth 1 https://github.com/borglab/gtsam.git /opt/gtsam

# -----------------------------------------------------------
# 4. Build + Install GTSAM (C++ + Python)
# -----------------------------------------------------------
RUN mkdir -p /opt/gtsam/build && \
    cd /opt/gtsam/build && \
    cmake .. \
      -DGTSAM_BUILD_PYTHON=ON \
      -DGTSAM_FORCE_SHARED=ON \
      -DCMAKE_BUILD_TYPE=Release && \
    make -j2 && \
    make install && \
    pip install /opt/gtsam/build/python && \
    ldconfig

# -----------------------------------------------------------
# 5. Create VS Code user
# -----------------------------------------------------------
RUN useradd -m -s /bin/bash vscode
USER vscode

WORKDIR /workspace
