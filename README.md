# MPC-CBF for Overtaking

A Python-based simulator for Model Predictive Control (MPC) with Control Barrier Functions (CBF) for vehicle overtaking maneuvers, with Acados and optional RL (PASTA) tuning.

## Dependencies

- **CasADi**: 3.6.7 – symbolic framework for nonlinear optimization
- **NumPy**: ≥1.22.4, <2.0
- **Pandas**: 2.2.3
- **Matplotlib**: 3.5.1
- **SciPy**: 1.8.0
- **PyYAML**: config parsing
- **GurobiPy**: MIQP solver
- **PyTorch**: RL (PASTA) and controller
- **acados**: OCP solver (see Installation)

## Project Structure

### Core

- **`helpers/car_models.py`**: Vehicle models (kinematic)
- **`helpers/controllers.py`**: Control algorithms (Pure Pursuit, PID)
- **`helpers/sim_helpers.py`**: Simulation helpers (obstacles, splines, collision/boundary checks)

### Acados MPC

- **`acados_controllers/mpc_cbf_ff_setup.py`**: Acados OCP setup for Frenet-frame kinematic model with CBF
- **`acados_controllers/mpc_cbf_ff_rl.py`**: Acados-based MPC-CBF in Frenet frame with RL-tunable CBF (gamma)

### RL

- **`rl_pasta/pasta.py`**: PASTA (preference-based) RL for multi-objective tuning

### Simulator

- **`simulators/FrenetFrameSimulatorMIQP-MPC_acados_with_rl.py`**: Frenet-frame MIQP-MPC simulator using Acados, with optional RL

### Config and data

- **`configs/params.yaml`**: Simulator, vehicle, and Acados MIQP-MPC parameters
- **`racelines/*.csv`**: Reference racelines (e.g. raceline_KS.csv, raceline_Monza.csv, raceline_LS.csv, raceline_IMS.csv)
- **`boundaries/*.csv`**: Track boundaries (e.g. boundaries_KS.csv, boundaries_Monza.csv, boundaries_LS.csv, boundaries_IMS.csv)

## Usage

### Run the simulator

```bash
python3 simulators/FrenetFrameSimulatorMIQP-MPC_acados_with_rl.py
```

Tune behavior in `configs/params.yaml` under:

- **`simulator:`** – raceline/boundary paths, dt, obstacles, etc.
- **`vehicle_models:`** – initial state, wheelbase
- **`acados_miqp_mpc_params:`** – MIQP/CBF and Frenet MPC (horizon, costs, gamma, safety margins)

### Installation

1. Python deps:

   ```bash
   pip install -r requirements.txt
   ```

2. CasADi version:

   ```bash
   pip install casadi==3.6.7
   ```

3. Optional: `future-fstrings` if needed:

   ```bash
   pip install future-fstrings
   ```

4. **Acados**: clone in home directory.

   ```bash
   git clone https://github.com/acados/acados.git
   cd acados
   git checkout v0.4.3
   git submodule update --recursive --init
   mkdir -p build && cd build
   cmake -DACADOS_WITH_QPOASES=ON ..
   make install -j4
   ```

5. Add to `~/.bashrc` and run `source ~/.bashrc`:

   ```bash
   export ACADOS_SOURCE_DIR=~/acados
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib
   export PYTHONPATH=$PYTHONPATH:$ACADOS_SOURCE_DIR/interfaces/acados_template
   ```

6. Compile generated code in `c_generated_code/` if your setup uses it.

7. For MIQP: install and license Gurobi.

## Notes

- CBF constraints are used for obstacle avoidance; gamma can be tuned by RL (PASTA).
- MIQP enables discrete overtaking decisions.
- Simulator supports static and dynamic obstacles and multiple raceline/boundary sets.
