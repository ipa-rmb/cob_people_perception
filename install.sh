#!/usr/bin/env bash
set -ex

# Make it "0" if you have already OpenCV
INSTALL_OPENCV=0
# Make it "0" if you have already Dlib
INSTALL_DLIB=0
# Make it "0" if you have already Torch, but you need to install
# Torch packages manually
INSTALL_TORCH=0

apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    gfortran \
    git \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    libssl-dev \
    libffi-dev \
    pkg-config \
    python-dev \
    python-pip \
    python-numpy \
    python-nose \
    python-scipy \
    python-pandas \
    python-numpy \
    python-protobuf\
    python-openssl \
    software-properties-common \
    wget \
    zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


if [[ "$INSTALL_OPENCV" = 1 ]] ; then
    cd ~ && \
    mkdir -p ocv-tmp && \
    cd ocv-tmp && \
    curl -L https://github.com/Itseez/opencv/archive/2.4.11.zip -o ocv.zip && \
    unzip ocv.zip && \
    cd opencv-2.4.11 && \
    mkdir release && \
    cd release && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_PYTHON_SUPPORT=ON \
          .. && \
    make -j8 && \
    make install && \
    rm -rf ~/ocv-tmp
fi

if [[ "$INSTALL_TORCH" = 1 ]] ; then
    curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash -e
    git clone https://github.com/torch/distro.git ~/torch --recursive
    cd ~/torch && ./install.sh && \
    cd install/bin && \
    ./luarocks install nn && \
    ./luarocks install dpnn && \
    ./luarocks install image && \
    ./luarocks install optim && \
    ./luarocks install csvigo && \
    ./luarocks install torchx && \
    ./luarocks install tds
fi

if [[ "$INSTALL_DLIB" = 1 ]] ; then
     cd ~ && \
     mkdir -p dlib-tmp && \
     cd dlib-tmp && \
     curl -L \
         https://github.com/davisking/dlib/archive/v19.0.tar.gz \
         -o dlib.tar.bz2 && \
     tar xf dlib.tar.bz2 && \
     cd dlib-19.0/python_examples && \
     mkdir build && \
     cd build && \
     cmake ../../tools/python && \
     cmake --build . --config Release && \
     cp dlib.so /usr/local/lib/python2.7/dist-packages && \
     rm -rf ~/dlib-tmp
fi

git clone https://github.com/cmusatyalab/openface.git ~/openface --recursive
cd ~/openface && \
    ./models/get-models.sh && \
    pip2 install -r requirements.txt && \
    python2 setup.py install && \

echo Installation of dependencies of cob_people_detection completed
