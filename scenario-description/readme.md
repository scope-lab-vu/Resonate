# TextX Based Scene Generation in CARLA

We use a scenario description DSML written in [textX](https://textx.github.io/textX/stable/) to generate different scenes with different weather patterns. 

[Scenrio.entity](https://github.com/scope-lab-vu/Resonate/blob/main/scenario-description/scenario.entity) -- Has the entities of a CARLA scenarion. The entities are town name, weather, ego_agent, other_agent, global_route, and hazard_list.

[entity.tx](https://github.com/scope-lab-vu/Resonate/blob/main/scenario-description/entity.tx) -- Has the grammer for the scenario description language. 

[scenario-generator.py](https://github.com/scope-lab-vu/Resonate/blob/main/scenario-description/scenario-generator.py) -- Parses the scenario language as python object and fills in the required values for each entity. Then an XML is generated that can be used in CARLA.

[sample xml files](https://github.com/scope-lab-vu/Resonate/tree/main/scenario-description/Scenario-example/simulation1) -- The sample xml files generated for CARLA. 

Scenario.dot, entity.dot -- metamodel figures of the scenario description and the textual language. Read the [docs](https://textx.github.io/textX/stable/) to convert it to png.

To generate scenes with different weather patterns, activate the virtual environment using demo/bin/activate. Then run the following commands to generate different simulation scenarios.

```
textx generate entity.tx --target dot
textx generate scenario.entity --grammar entity.tx --target dot
python3 scenario-generator.py 
```
The scene generator takes a route xml file from the Carla AD challenge, and generates one simulation run by randomly samples the route with different weather patterns to generate different scenes. A sample simulation xml files are shown in [folder](https://github.com/Shreyasramakrishna90/Resonate-Dynamic-Risk/tree/main/resonate-carla/leaderboard/data/my_routes)
