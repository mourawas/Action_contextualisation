# Handle ros distro
ARG ROS_DISTRO=noetic
FROM ghcr.io/aica-technology/ros-ws:${ROS_DISTRO}

# User provided arguments
ARG HOST_GID=1000
ARG GIT_NAME=""
ARG GIT_EMAIL=""
ARG USE_SIMD=OFF



# Tell docker we want to use bash instead of sh in general
SHELL ["/bin/bash", "-c"]

### Add the user to the current GID of the host to avoid permisson issues in volumes
# AICA uses the same name for user and user group
ENV USER_GROUP=${USER}
USER root
RUN if [ "HOST_GID" != "1000"] ; \
    then groupadd --gid ${HOST_GID} host_group && \
    usermod ${USER} -g ${HOST_GID} && \
    usermod ${USER} -a -G ${USER_GROUP}; fi
USER ${USER}

# Setup git identity
RUN git config --global user.name "${GIT_NAME}"
RUN git config --global user.email "${GIT_EMAIL}"

# Fix ROS GPG key issue
RUN sudo mkdir -p /usr/share/keyrings
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo tee /usr/share/keyrings/ros-archive-keyring.gpg > /dev/null
RUN sudo rm -f /etc/apt/sources.list.d/ros*.list
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros-latest.list > /dev/null

# Setup python version for noetic
RUN sudo apt update
RUN if [ "${ROS_DISTRO}" == "noetic" ] ; \
    then sudo apt install python-is-python3 ; fi


### Add a few tools
RUN sudo apt-get update && sudo apt-get install -y \
    bash-completion \
    silversearcher-ag \
    apt-transport-https \
    less \
    alsa-utils \
    ros-${ROS_DISTRO}-ros-control \
    ros-${ROS_DISTRO}-ros-controllers \
    net-tools \
    netbase \
    && sudo apt-get upgrade -y && sudo apt-get clean

# Install gazebo (9 or 11 depending on distro)
# WORKDIR /home/${USER}
# RUN sudo apt-get update
# RUN if [ "$ROS_DISTRO" = "noetic" ] ; \
#         then sudo apt-get install -y gazebo11 ; fi
# RUN if [ "$ROS_DISTRO" = "melodic" ] ; \
#         then sudo apt-get install -y gazebo9 ; fi

# # Install gaezbo ros packages
# RUN sudo apt install -y ros-${ROS_DISTRO}-gazebo-ros-pkgs \
#                         ros-${ROS_DISTRO}-gazebo-ros-control

# Handle SIMD option
RUN if [ "${USE_SIMD}" = "ON" ] ; \
    then export CMAKE_CXX_FLAGS="-march=native -faligned-new" ; fi

### Install all dependencies of IIWA ROS
# Clone KUKA FRI (need to be root to clone private repo)
#WORKDIR /tmp
#USER root
#RUN mkdir -p -m 0775 /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts

#RUN --mount=type=ssh git clone git@github.com:epfl-lasa/kuka_fri.git
#WORKDIR /tmp/kuka_fri
#RUN if [ "${USE_SMID}" != "ON" ] ; \
#    then wget https://gist.githubusercontent.com/matthias-mayr/0f947982474c1865aab825bd084e7a92/raw/244f1193bd30051ae625c8f29ed241855a59ee38/0001-Config-Disables-SIMD-march-native-by-default.patch \
#    ; fi
#RUN --mount=type=ssh  if [ "${USE_SMID}" != "ON" ] ; \
#    then git am 0001-Config-Disables-SIMD-march-native-by-default.patch ; fi
#
# Transfer repo back to original user after root clone
#WORKDIR /tmp
#RUN chown -R ${USER}:${HOST_GID} kuka_fri

# Install kuka Fri as USER
#USER ${USER}
#RUN cd kuka_fri && ./waf configure && ./waf && sudo ./waf install

# Install SpaceVecAlg
RUN git clone --recursive https://github.com/jrl-umi3218/SpaceVecAlg.git
RUN cd SpaceVecAlg && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_BINDING=OFF .. \
    && make -j && sudo make install

# Install RBDyn
RUN git clone --recursive https://github.com/jrl-umi3218/RBDyn.git
RUN cd RBDyn && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_BINDING=OFF .. \
    && make -j && sudo make install

# Install mc_rbdyn_urdf
RUN git clone --recursive -b v1.1.0 https://github.com/jrl-umi3218/mc_rbdyn_urdf.git
RUN cd mc_rbdyn_urdf && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_BINDING=OFF .. \
    && make -j && sudo make install

# Install corrade
RUN git clone https://github.com/mosra/corrade.git
RUN cd corrade && git checkout 0d149ee9f26a6e35c30b1b44f281b272397842f5 \
    && mkdir build && cd build && cmake .. && make -j && sudo make install

# Install robot_controller
RUN git clone https://github.com/epfl-lasa/robot_controllers.git
RUN cd robot_controllers && mkdir build && cd build \
    && cmake .. && make -j && sudo make install

# Remove temporary files
RUN sudo ldconfig
RUN sudo rm -rf /tmp/*

### Install IIWA ROS
# CHANGES STARTING FROM THE FIRST &&, REMOVE IF NEEDED
WORKDIR /home/${USER}/ros_ws/src
RUN git clone https://github.com/epfl-lasa/iiwa_ros.git && \
	cd iiwa_ros && \
	rm -rf iiwa_driver iiwa_moveit iiwa_gazebo

### Add environement variables to bashrc
WORKDIR /home/${USER}

# Give bashrc back to user
RUN sudo chown -R ${USER}:${HOST_GID} .bashrc

# Add cmake option to bash rc if needed
RUN if [ "${USE_SIMD}" = "ON" ] ; \
    then echo "export ENABLE_SIMD=ON" >> /home/${USER}/.bashrc ; fi

# Additional dependencies for the current package
RUN pip install mujoco \
                numpy-quaternion \
                sympy \
                scipy \
                torch==1.12.0 \
                omegaconf==2.2.3 \
                urdfpy==0.0.22 \
                pytorch-kinematics \
                open3d \
                scikit-image \
                pyembree \
                openai \
                pddl \
                pyVHACD \
                mistralai


RUN sudo apt install python3-rosdep

# Install Pinocchio
RUN sudo apt install -qqy lsb-release curl
RUN sudo mkdir -p /etc/apt/keyrings && curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc | sudo tee /etc/apt/keyrings/robotpkg.asc
RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
    | sudo tee /etc/apt/sources.list.d/robotpkg.list
RUN sudo apt update
RUN sudo apt install -qqy robotpkg-py3*-pinocchio
RUN source /home/${USER}/.bashrc && echo "export PATH=/opt/openrobots/bin:$PATH" >> /home/${USER}/.bashrc
RUN source /home/${USER}/.bashrc && echo "export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH" >> /home/${USER}/.bashrc
RUN source /home/${USER}/.bashrc && echo "export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH" >> /home/${USER}/.bashrc
RUN source /home/${USER}/.bashrc && echo "export PYTHONPATH=/opt/openrobots/lib/python3.8/site-packages:$PYTHONPATH" >> /home/${USER}/.bashrc
RUN source /home/${USER}/.bashrc &&  echo "export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH" >> /home/${USER}/.bashrc

# Robotics toolbox from personal fork
WORKDIR /home/${USER}
RUN git clone https://github.com/niederha/robotics-toolbox-python.git && pip install ./robotics-toolbox-python

### Build ros workspace
WORKDIR /home/${USER}/ros_ws
# Calling these 2 manually
# RUN source /home/${USER}/.bashrc && rosdep install --from-paths src --ignore-src -r -y
# RUN source /home/${USER}/.bashrc && catkin_make

# For some reason we need a more recencent version of numpy
RUN pip install --upgrade numpy==1.23

# Install nvidia drivers
RUN sudo apt update
# RUN sudo apt install software-properties-common -y
# RUN sudo add-apt-repository ppa:graphics-drivers/ppa
# RUN sudo apt-get install ubuntu-drivers-common -y
# RUN sudo apt install nvidia-driver-545 -y
# RUN sudo ubuntu-drivers install
# NVIDIA driver argument
ARG nvidia_binary_version="535.159.05"
ARG nvidia_binary="NVIDIA-Linux-x86_64-${nvidia_binary_version}.run"


# RUN wget -q https://us.download.nvidia.com/XFree86/Linux-x86_64/${nvidia_binary_version}/${nvidia_binary} \
#     && sudo chmod +x ${nvidia_binary} && \
#     sudo ./${nvidia_binary} --accept-license --ui=none --no-kernel-module --no-questions && \
#     sudo rm -rf ${nvidia_binary}

### Final apt clean
RUN sudo apt update -y
RUN sudo apt upgrade -y
RUN sudo apt clean -y
