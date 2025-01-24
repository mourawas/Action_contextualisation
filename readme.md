# LLM robotics with ROS integration

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

### Starting simulator
Once you are in the docker, the files are arranged in a ros workspace. To start the simulator:
```bash
catkin_make
source devel/setup.bash
roslaunch llm_simulator simulator.launch
```


First of all you need to start you hand controller on the host computer:
```bash
roslaunch allegro_hand allegro_hand.launch HAND:=right CONTROLLER:=torque
```

Then you need to start the vision server on the docker:
```bash
roslaunch vision_server full_real_robot.launch 
```

Then you need to start the FRIOverlay_gripper on the IIWA tablet. Choose the gripper you have calibrated, pick position control and 
then on a second docker window start the contrller with:

```bash
rosrun primitive_library test_task_plan.py
```

Then allow the robot to move by selected a gain of 500. 

You can manually alter the test_task_plan.py script to test different scenarios.



### Developing with docker
You can simply edit your files on your computer, the changes will synchronize with the docker. Alternatively you can use a docker tool in your favorite IDE and work directly in the docker environment. It doesn't matter if you edit files in the docker or on your computer they are synchronized in real time. VScode's docker extension works very well for example.

### Interfacing a controller using ros topics
There are 4 ros topics to control the allegro hand and IIWA. Those topics are the same as the ones that will be on the real robots. The topics are as follow:
- `/iiwa7/joint_states`  [**JointStates**](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/JointState.html): The joint state of the IIWA
- `/allegro_hand_left/joint_states` [**JointStates**](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/JointState.html): The joint states of the allegro hand
- `/iiwa7/TorqueController/command` [**Float64MultiArray**](http://docs.ros.org/en/melodic/api/std_msgs/html/msg/MultiArrayLayout.html): The torque command for the IIWA robot. We expect layout.dim to be filled with a single layout, have a stride of 1 and a size of 7. Check the parsing in `llm_simulator.py` in `_iiwa_torque_cmd_cb` if you don't know how to access this field. It is also better if the label of the command is `'joint'` but not mandatory.
- `/allegro_hand_left/TorqueController/command` [**Float64MultiArray**](http://docs.ros.org/en/melodic/api/std_msgs/html/msg/MultiArrayLayout.html): The torque command for the Allegro hand. We expect layout.dim to be filled with a single layout, have a stride of 1 and a size of 16. Check the parsing in `llm_simulator.py` in `_allegro_torque_cmd_cb` if you don't know how to access this field. It is also better if the label of the command is `'joint'` but not mandatory.

## Notes (deprecated)
- If you want to update the GUI viewer step by step, use notebook and don't forget to add `view.sync()` after sending commands,
see `ycb_initial_grasp.ipynb` as an example.
- For LLM+robotics, run `llm_initial.ipynb` as an example. It loads `descriptions/iiwa7_allegro_llm.xml`
with 3 champagne glasses.
- Use Python scripts to send commands in a while loop and set `auto_sync=True`, see `sin_test.py` as an example.
- The inertial attributes are assigned by `Trimesh`. `.obj` files are loaded. Given a specific mass, scale the inertial matrix.
See `description/YCB_objects/ycb_gnerate_inertia.ipynb` for details. Please ask Xiao if you need to modify the inertial parameters.
- Since MuJoCo can only handle convex objects (otherwise a convex hull is generated), to add nonconvex objects,
we need to split it into some convex pieces and load them together as one body. Please ask Xiao for this.
- `*.jpg` is not supported as texture in MuJoCo

## Visualization
- To change the color and transparency of `champagne glass`, please change the `rgba` value at `description/llm_objects/champagne_glass/bodies.xml`

## VHACD for splitting nonconvex objects
 check [obj2mjcf](https://github.com/kevinzakka/obj2mjcf)

`obj2mjcf --obj-dir  . --save-mjcf --compile-model --verbose --vhacd-args.enable --vhacd-args.split-hull --vhacd-args.no-disable-shrink-wrap --vhacd-args.volume-error-percent 0.002 --vhacd-args.max-hull-vert-count 800 --vhacd-args.max-recursion-depth 30 --vhacd-args.max-output-convex-hulls 64`
