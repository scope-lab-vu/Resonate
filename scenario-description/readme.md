# TextX Based Scene Generation in CARLA

We use a scenario description DSML written in [textX](https://textx.github.io/textX/stable/) to generate different scenes with different weather patterns. 

[Scenrio.entity](https://github.com/Shreyasramakrishna90/Carla-Safety-Evaluation/blob/master/textx-scenario-description/demo/scenario.entity) -- Has the entities of a CARLA scenarion. The entities are town name, weather, ego_agent, other_agent, global_route, and hazard_list.

[entity.tx](https://github.com/Shreyasramakrishna90/Carla-Safety-Evaluation/blob/master/textx-scenario-description/demo/entity.tx) -- Has the grammer for the scenario description language. 

[scenario-parser.py](https://github.com/Shreyasramakrishna90/Carla-Safety-Evaluation/blob/master/demo/textx-scenario-description/scenario-parser.py) -- Parses the scenario language as python object and fills in the required values for each entity. Then an XML is generated that can be used in CARLA.

[sample-output.xml](https://github.com/Shreyasramakrishna90/Carla-Safety-Evaluation/blob/master/demo/textx-scenario-description/sample-output.xml) -- The sample xml file generated for CARLA. 

Scenario.dot, entity.dot -- metamodel figures of the scenario description and the textual language. Read the [docs](https://textx.github.io/textX/stable/) to convert it to png.

To generate scenes with different weather patterns, activate the virtual environment using demo/bin/activate. Then run the following commands to generate different simulation scenarios.

```
textx generate entity.tx --target dot
textx generate scenario.entity --grammar entity.tx --target dot
python3 scenario-generator.py 
```
The scene generator takes a route xml file from the Carla AD challenge, and generates one simulation run by randomly samples the route with different weather patterns to generate different scenes. A sample simulation xml files are shown in [folder](https://github.com/Shreyasramakrishna90/Resonate-Dynamic-Risk/tree/main/resonate-carla/leaderboard/data/my_routes)
