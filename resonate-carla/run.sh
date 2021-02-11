export CARLA_ROOT=/opt/CARLA_0.9.9.4      # change to where you installed CARLA
export PORT=2001                                            # change to port that CARLA is running on
export ROUTES=leaderboard/data/routes/route_19.xml          # change to desired route
export TEAM_AGENT=auto_pilot.py                             # no need to change
export TEAM_CONFIG=sample_data                              # change path to save data

./run_agent.sh
