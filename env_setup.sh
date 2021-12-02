#! /bin/bash
set -e

echo ""
echo "--------------------------------------------"
echo "|      Creating virtual environment        |"
echo "--------------------------------------------"
python3 -m venv .dt_gym
source .dt_gym/bin/activate

echo "--------------------------------------------"
echo "|         Cloning gym-duckietown           |"
echo "--------------------------------------------"
git clone https://github.com/duckietown/gym-duckietown.git

echo ""
echo "--------------------------------------------"
echo "|        Installing gym-duckietown         |"
echo "--------------------------------------------"
pip3 install --upgrade pip
pip3 install -e gym-duckietown 

echo "--------------------------------------------"
echo "|         Installing dependencies          |"
echo "--------------------------------------------"
pip3 install stable-baselines3[extra]
pip3 install seaborn
# TODO

echo "--------------------------------------------"
echo "
| To activate this environment, use        |
|                                          | 
|     $ source .dt_gym/bin/activate        |
|                                          |
| To deactivate an active environment, use |
|                                          |
|     $ deactivate                         |
"
echo "--------------------------------------------"
echo "|            Setup successful              |"
echo "--------------------------------------------"