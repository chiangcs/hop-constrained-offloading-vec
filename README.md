# Vehicular Edge Computing Task Offloading Simulation

## Overview

The system models a realistic VEC scenario with the following components:

- **Vehicles** with V2V and V2MEC wireless communication
- **MEC servers** arranged in a grid, connected via high-speed backhaul
- **Subtasks** organized as a tree-structured dependency graph
- **Multi-hop communication** with a configurable hop limit

Three offloading algorithms are compared:

| Method | Description |
|--------|-------------|
| **Ours (BFS-based)** | BFS-guided greedy offloading that respects communication reachability constraints |
| **RANDOM** | Random device selection within reachable neighbors |
| **PBTSA** | Priority-based scheduling using a maximum clique of the communication graph |

## Requirements

Install dependencies via pip:

```bash
pip install numpy networkx tqdm
```

## Usage

Run all experiments:

```bash
python main.py
```

This will sequentially execute the following experiments and print results to stdout:

- **Coverage Range vs Latency** — area sizes from 500m to 2000m
- **Number of Subtasks vs Latency** — subtask counts from 100 to 600
- **Number of Vehicles vs Latency** — vehicle counts from 10 to 100
- **Number of MEC Servers vs Latency** — MEC server counts from 1 to 20
- **Max Hops vs Latency** — hop limits from 1 to 6

Each experiment runs 20 independent trials and reports averaged latency for all three methods.

## System Parameters

| Parameter | Default Value |
|-----------|--------------|
| Area size | 1500 m |
| V2V range | 150 m |
| V2MEC range | 200 m |
| Max hops | 4 |
| Vehicle CPU | 2 GHz |
| MEC server CPU | 10 GHz |
| V2V bandwidth | 25 MHz |
| V2MEC bandwidth | 30 MHz |
| M2M rate | 1 Gbps |

## Project Important Structure

```
main.py
├── VehicularEdgeComputingSystem      # Main simulation class
│   ├── _generate_positions           # Random vehicle + grid MEC placement
│   ├── _generate_physical_graph      # Single-hop communication graph
│   ├── _compute_multihop_graph       # Multi-hop equivalent rate computation
│   ├── _generate_tree_task_graph     # Random tree-structured subtask
│   ├── compute_execution_priority    # Algorithm 1: Execution Priority
|   ├── reassign                      # Algorithm 3: Reassignment
│   ├── bfs_based_offloading          # Algorithm 2: proposed BFS offloading
│   ├── random_offloading             # Baseline: random assignment
│   └── pbtsa_baseline                # Baseline: PBTSA with max clique
└── Experiment functions
    ├── experiment_coverage_range
    ├── experiment_num_subtasks
    ├── experiment_num_vehicles
    ├── experiment_num_mec_servers
    └── experiment_max_hop
```