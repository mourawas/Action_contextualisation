# LLM robotics with ROS integration

## Branches
This project has 2 branches. The sim branch on which you are if you can read this should only be used to run all experiments in simulation. If you want to do some real robot stuff please switch to the realrobot branch.

## Setup
This implementation uses a docker to run the simulator. This allows to port the environment to any computer without having to do extra setup.
The only thing needed is to build the docker. Note that this operation can take anywhere between 5 minutes and 1 hour depending on your computer and how the docker evolves in this project. That being said, you only need to build the docker once. Volumes are mounter on the docker for the ros_ws so you can develop the packages on your computer or on the docker and the changes are applied at both places. No need to rebuild the docker with each changes but depending on what you do you might still need to redo a `catkin_make`.

### Make sure you have a docker environment
Docker requires a few tools to be installed on your computer. You can find information and a script to install them for you [here](https://github.com/epfl-lasa/wiki/wiki/Docker-install-script)

### Build the docker
```bash
bash build_docker.sh    # Only do once. Operations are cached once it is completed
```

### Start the docker
```bash
bash start_docker.sh interactive   # Doing it this way starts the docker in your terminal. The docker will end once you close the terminal
```

```bash
bash start_docker.sh server   # Here we start the docker in the background. You see nothing in your terminal
bash start_docker.sh connect  # Here we connect to the docker and have terminal output. You can connect simultaneously from as many terminal as you want using this. Even if you kill the terminal, the container will always be running in the background
```

#### NVIDIA drivers
Once the docker has started if you plan on using nvidia drivers you should add
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-545

# Check that is worked
nvidia-smi
```

#### GPT API
If you plan on using the GPT api in the robot you will need to reference you api key. Please beware, with automated scripts money can go by fast, especially since we are using chat-gpt 4:
```bash
export GPT_API_KEY=YOU_GPT_API_KEY
```

#### Build and source ros packer
Classic, in the docker do:
```bash
catkin_make
source devel/setup.bash
```

### Starting simulator
Once you are in the docker, the files are arranged in a ros workspace. To start the simulator:
```bash
catkin_make
source devel/setup.bash
roslaunch llm_simulator simulator.launch
```

### Developing with docker
You can simply edit your files on your computer, the changes will synchronize with the docker. Alternatively you can use a docker tool in your favorite IDE and work directly in the docker environment. It doesn't matter if you edit files in the docker or on your computer they are synchronized in real time. VScode's docker extension works very well for example.

### Structure

The ropository is structured as follow:

- llm_common package: Bunch of common functions and definitions used throught the packages
- llm_simulator package: The simulator. It is based on Xiao's code. You can find some .xml files there describing objects, robots and placement. If you need help navigating those files, ask Xiao. There is also a `simulation.launch` file that lauches the `simulator.py` node which reproduces all topics and servers as they are on the real robot.
- primitive_library package containing the following:
     - controller_base which implements the basic communication and functions needed. It inherits from robotics_toolbox_pyhton which has a lot of builtin robotics stuff. You can inherit from this class to create your own controller and just implement the run_controller function.
     - js_lds (implementation of joint space linear dynamical system)
     - js_lds_orientation (inherits from js_lds, adds a parameter to maintain orientation)
     - js_lds_oa (inherits from js_lds_orientation, adds obstacle avoidance)
     - action_functions: Uses js_lds_oa and other stuff to implement action primitive
     - predicates: Uses js_lds_oa and other stuff to implement predicate checks
     - execute_task_plan: Give it action and task plan to execute them. Also responsible for logging the `task_plan_log_run_**.txt` in a cache folder.
     - In scripts: A bunch of nodes to test different aspect of the primitive. Start the simulator in an other terminal and run those to see how they work.
     - In launch: test_task_plan.lauch: A launch file that starts the simulator and runs the test_task_plan.py script in a single file.
- vision_serve: Some implementation of the vision server. Not up-to-date here. Switch to the realrobot branch if you want to use it.
- planner: Contains all the stuff to discuss with the llm as well as nodes for experiments 1, 2 and 4. Experiments 1, 2 and 4 also have corresponding launch files in launch allowing not to have to run the simulator in a separate terminal.

### Interfacing a controller using ros topics
There are 4 ros topics to control the allegro hand and IIWA. Those topics are the same as the ones that will be on the real robots. The topics are as follow:
- `/iiwa7/joint_states`  [**JointStates**](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/JointState.html): The joint state of the IIWA
- `/allegro_hand_left/joint_states` [**JointStates**](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/JointState.html): The joint states of the allegro hand
- `/iiwa7/TorqueController/command` [**Float64MultiArray**](http://docs.ros.org/en/melodic/api/std_msgs/html/msg/MultiArrayLayout.html): The torque command for the IIWA robot. We expect layout.dim to be filled with a single layout, have a stride of 1 and a size of 7. Check the parsing in `llm_simulator.py` in `_iiwa_torque_cmd_cb` if you don't know how to access this field. It is also better if the label of the command is `'joint'` but not mandatory.
- `/allegro_hand_left/TorqueController/command` [**Float64MultiArray**](http://docs.ros.org/en/melodic/api/std_msgs/html/msg/MultiArrayLayout.html): The torque command for the Allegro hand. We expect layout.dim to be filled with a single layout, have a stride of 1 and a size of 16. Check the parsing in `llm_simulator.py` in `_allegro_torque_cmd_cb` if you don't know how to access this field. It is also better if the label of the command is `'joint'` but not man