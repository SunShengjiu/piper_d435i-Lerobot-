# piper_d435i

Scripts and recorded LeRobot datasets for the Piper arm with two Intel D435i
cameras.

## Scripts

- `stage1_hardware_check.py` checks the Piper SDK and both cameras.
- `d435i_param_depth_test.py` configures camera parameters and validates RGB-D
  capture.
- `lerobot_collect.py` records and resumes LeRobot episodes.

## Dataset

The recorded datasets are under `lerobot_data/`. MP4 and Parquet files are
stored with Git LFS because of their size. Install Git LFS before cloning or
pulling the dataset:

```bash
git lfs install
git clone git@github.com:SunShengjiu/piper_d435i.git
cd piper_d435i
git lfs pull
```

The local `.venv/` and Python cache files are intentionally excluded from the
repository; install the dependencies described in the scripts on each machine.
