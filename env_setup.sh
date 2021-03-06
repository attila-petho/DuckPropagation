#! /bin/bash
set -e

echo ""
echo "--------------------------------------------"
echo "|      Creating virtual environment        |"
echo "--------------------------------------------"
conda env create -f env.yml

echo "--------------------------------------------"
echo "|         Cloning gym-duckietown           |"
echo "--------------------------------------------"
git clone https://github.com/duckietown/gym-duckietown.git

echo ""
echo "--------------------------------------------"
echo "|        Installing gym-duckietown         |"
echo "--------------------------------------------"
conda run -vn dtgym pip3 install --upgrade pip
conda run -vn dtgym pip3 install -e gym-duckietown

echo "logger.setLevel('WARNING')" >> ./gym-duckietown/src/gym_duckietown/__init__.py

echo "--------------------------------------------"
echo "
| To activate this environment, use        |
|                                          | 
|     $ conda activate dtgym               |
|                                          |
| To deactivate an active environment, use |
|                                          |
|     $ conda deactivate                   |
"
echo "--------------------------------------------"
echo "|            Setup successful              |"
echo "--------------------------------------------"
