FROM dorowu/ubuntu-desktop-lxde-vnc:xenial

# Install ROS Kinetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt update && apt install -y \
    ros-kinetic-desktop-full \
    python-rosinstall \
    python-rosinstall-generator \
    python-wstool \
    build-essential

# Install Python 3.7, pip, and packages
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt-get install -y python3.7 python3.7-dev python-pip python3-pip
RUN python3.7 -m pip install numpy