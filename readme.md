# Beam Assembly with LLM Robot Control

Transparent LLM-based planning framework for robotic beam assembly. Achieves 85% success rate on L-shape and U-shape assembly tasks through iterative refinement in simulation.

## System Overview

Uses LLMs to generate task plans for a KUKA iiwa14 arm with Allegro hand. The system combines:
- LLM-based task and evaluation planning with physically grounded, interpretable action primitives
- Dynamical systems control with collision avoidance
- Simulation-based parameter tuning and replanning
- MuJoCo physics simulation integrated with ROS

## ROS Packages

- **llm_common**: Shared utilities and base classes for LLM integration
- **llm_simulator**: MuJoCo-based simulation environment and services
- **primitive_library**: Action primitives (approach, pick, place) with physical parameters
- **planner**: LLM-based task planner with iterative refinement
- **vision_server**: Vision services for real robot deployment with OptiTrack

## Prerequisites

- ROS (tested with ROS Noetic)
- Docker
- Python 3
- MuJoCo physics engine
- LLM API access (GPT-4.1-mini recommended)

## Installation

1. Clone the repository
2. Build the Docker image:
```bash
./start_docker.sh build
```

3. Set up your LLM API key:
```bash
export GPT_API_KEY=your_api_key_here
```

Note: While Mistral (free) is supported, a smarter (paid) model such as GPT-4.1-mini is strongly recommended.

## Usage

1. Start the Docker container:
```bash
./start_docker.sh interactive
```

2. Inside the container, source the workspace:
```bash
source devel/setup.bash
```

3. Run the full pipeline:
```bash
roslaunch planner experiment_1.launch
```

This launches:
- MuJoCo simulator with KUKA iiwa14 and Allegro hand
- Vision server for object pose tracking
- LLM planner with iterative refinement
- Action primitives library

## Task Examples

The specific task given to the robot should be changed in ```ros_ws/src/planner/scripts/experiment_1.py```

**L-Shape Assembly**: Places one beam vertically at the end of a horizontal base beam

**U-Shape Assembly**: Places two beams vertically at both ends of a horizontal base beam

## Action Primitives

All primitives use physically meaningful parameters:

- **approach()**: Move to beam vicinity
- **pick()**: Grasp and lift up the beam
- **place()**: Place beam at target location with target orientation

## Performance

Experimental results (20 runs, GPT-4.1-mini):
- Overall success rate: 85%
- L-shape: 90% (9/10)
- U-shape: 80% (8/10)

Primitive success rates:
- approach: 98.8%
- pick: 92.1%
- place: 84.8%

## Project Structure

```
ros_ws/src/
├── llm_common/          # Shared LLM utilities
├── llm_simulator/       # MuJoCo simulation environment
├── primitive_library/   # Action primitives implementation
├── planner/            # LLM-based task planner
└── vision_server/      # Vision services for real robot
```

## Related Work

This work is a fork of [Action_contextualisation](https://github.com/epfl-lasa/Action_contextualisation) from Sthithpragya Gupta (LASA EPFL) who's work focused on table clearing using LLMs (Gupta & al. "Action Contextualization: Adaptive Task Planning and Action Tuning using Large Language Models", in IEEE Robotics and Automation Letters, 2024), in this work extended to construction tasks.

## Author

Mouhamad Rawas, EPFL Robotics MSc student.