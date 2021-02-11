#!/bin/bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT
k=0
for j in {0..1}
  do
    export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
    export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg           # 0.9.8
    export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg           # 0.9.8
    export PYTHONPATH=$PYTHONPATH:leaderboard
    export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
    export PYTHONPATH=$PYTHONPATH:scenario_runner

    if [ -d "$TEAM_CONFIG" ]; then
        CHECKPOINT_ENDPOINT="$TEAM_CONFIG/$(basename $ROUTES .xml).txt"
    else
        CHECKPOINT_ENDPOINT="$(dirname $TEAM_CONFIG)/$(basename $ROUTES .xml).txt"
    fi
    python leaderboard/leaderboard/leaderboard_evaluator.py \
    --challenge-mode \
    --track=dev_track_3 \
    --scenarios=leaderboard/data/all_towns_traffic_scenarios_public.json  \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --routes=${ROUTES} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --port=${PORT} \
    --record="/home/scope/Carla/ICCPS_CARLA_challenge/"\
    --simulation_number=${k}\
    --scene_number=${j}
    echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."
    k=$((k+=1))


done
