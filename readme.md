# LLM robotics with ROS integration

## Setup issues and fixes

### Docker Build Issues
- **ROS GPG key expiration**: Added key update commands to Dockerfile because ROS signing keys had expired in the base image
- **KUKA FRI private repository**: Commented out KUKA FRI installation since the repository requires private access and anyways isn't needed for simulation

### Container management  
- **Missing AICA docker tools**: Installed AICA's optional custom Docker tools from their GitHub repository since they're not available via pip

### GPU Support
- **NVIDIA Container Toolkit missing**: Installed NVIDIA Container Toolkit following official docs to enable GPU acceleration for MuJoCo

### ROS Workspace Build
- **Missing Gazebo packages**: Installed needed Gazebo and ROS packages inside container since they were commented out in Dockerfile
- **KUKA FRI dependency in build**: Deleted iiwa_driver, iiwa_moveit and iiwa_gazebo in the dockerfile because they need the KUKA FRI stuff from the private repo. To revert, remove the added lines in dockerfile and build again

### KUKA FRI Installation (Real Robot Support)
- **KUKA FRI from public repo**: Installed kuka_fri from public GitHub repo (https://github.com/mourawas/kuka_fri) instead of private epfl-lasa repo
- **Real robot packages restored**: Kept iiwa_driver and iiwa_moveit for real robot control, only removed iiwa_gazebo (not needed - using MuJoCo for simulation)
- **waf permissions**: Added `chmod +x waf` before building kuka_fri to fix permission issues

### CMake Compatibility Issues
- **CMake upgraded to 3.20+**: Required for RBDyn, but broke compatibility with older packages
- **CMake policy workaround**: Added `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` flag to:
  - mc_rbdyn_urdf
  - corrade
  - robot_controllers
  - catkin_make (workspace build)
- **Building workspace**: Must use `catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5` instead of plain `catkin_make`

### Python Package Versions
- **OpenAI compatibility**: Pinned `openai==1.12.0` for Python 3.8 compatibility (newer versions require jiter>=0.10.0 which doesn't exist for Python 3.8)

### LLM Integration
- **LLM framework**: Built the planner package with LLM code, adapted code to use Mistral AI in chatbots.py and adapted some prompts in prompt_generator.py
- **API Key Setup**: For Mistral (Free limited API), get your key here https://admin.mistral.ai/organization/api-keys. After building and sourcing the workspace, set your Mistral API key with `export MISTRAL_API_KEY=YOUR_API_KEY`. For Openai keys, `export GPT_API_KEY=YOUR_API_KEY`.

### Launch Commands

#### Simulation with the LLM with a launch file:
```bash
cd Action_contextualisation/
bash start_docker.sh interactive
catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash
export MISTRAL_API_KEY=YOUR_API_KEY  # Set API key after sourcing
roslaunch planner experiment_1.launch
```

#### If you want to run a specific script, in a first terminal:
```bash
cd Action_contextualisation/
bash start_docker.sh interactive
catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash
export MISTRAL_API_KEY=YOUR_API_KEY  # Set API key after sourcing
roslaunch llm_simulator simulator.launch 
```

#### And in a second terminal:
```bash
cd Action_contextualisation/
bash start_docker.sh connect
catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash
rosrun primitive_library test_beam.py
```

### Modifying Objects on Table

Edit `ros_ws/src/llm_simulator/src/description/llm_objects/llm_object_bodies.xml`:

Comment out objects:
```xml
<!-- <body name="apple" pos="0.5 -0.45 1.07" quat="1 1 0 0">
    <inertial pos="-0.0016480202957852835 -0.002256664593323946 -0.0007355532922551801" mass="0.002" fullinertia="0.00020654858167238498 0.0002167076279555137 0.000209544910164081 1.011183250711457e-06 -6.115403648378643e-07 1.948864069749578e-06" />
    <geom mesh="apple" class="obj" type="mesh"/>
    <geom mesh="apple" material="apple"  class="visual" type="mesh"/>
    <freejoint/>
</body> -->
```

Change object positions:
```xml
<body name="champagne_2" pos="0.6 -0.3 1.015">
```
- x: forward/backward
- y: left/right  
- z: height (1.07 is table height)

Then update dictionaries in `llm_common/utils.py`:
- `SIM_OBJECT_LIST`
- `ROUND_OBJECTS`

Rebuild and restart after changes.