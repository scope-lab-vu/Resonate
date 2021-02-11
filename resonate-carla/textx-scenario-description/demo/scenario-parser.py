#!/usr/bin/python3
import textx
import numpy as np
import lxml.etree
import lxml.builder
import sys

from textx.metamodel import metamodel_from_file

def scenario_town_description(scenarios,id):
    scenarios.entities[0].properties[0] = id
    scenarios.entities[0].properties[1] = "Town03"

    #print(scenarios.entities[0].properties)

    #return scenarios

def scenario_weather(scenarios):
    for i in range(len(scenarios.entities[1].properties)):
        print(str(scenarios.entities[1].properties[i].name))
        # if(str(scenarios.entities[1].properties[i].name) == "sun_altitude_angle"):
        #          scenarios.entities[1].properties[i] = "70.0"
        if (scenarios.entities[1].properties[i].type.name == "uniform"):
                scenarios.entities[1].properties[i] = str(np.random.uniform(0,100))
        elif(scenarios.entities[1].properties[i].type.name == "string"):
                scenarios.entities[1].properties[i] = "0.0"

    sun_altitude_angle = "70.0"
    scenarios.entities[1].properties[5] = sun_altitude_angle

    #print(scenarios.entities[1].properties)

    #return scenarios

def scenario_agent_route(scenarios,global_route):
    #for i in range(len(global_route)):
            for i in range(len(scenarios.entities[2].properties)):
                for j in range(5):
                    scenarios.entities[2].properties[i] = global_route[i]

            return scenarios

def ego_agent_info(scenarios,global_route):
        scenarios.entities[3].properties[0] = "vehicle.lincoln.mkz2017"
        scenarios.entities[3].properties[1] = "black"
        scenarios.entities[3].properties[2] = "hero"
        scenarios.entities[3].properties[3] = global_route[0]

        #print(scenarios.entities[3].properties)
        return scenarios

def other_agent_info(scenarios,other_agent_numbers,other_agent_route):
        scenarios.entities[4].properties[0] = "vehicle.lincoln.mkz2017"
        scenarios.entities[4].properties[1] = "blue"
        scenarios.entities[4].properties[2] = "other_agent"
        scenarios.entities[4].properties[3] = other_agent_route

        #print(scenarios.entities[4].properties)

def XML_generator(scenarios):

    E = lxml.builder.ElementMaker()
    ROOT = E.routes
    #DOC = E.doc
    route = E.route
    weather = E.weather
    waypoint1 = E.waypoint
    waypoint2 = E.waypoint
    waypoint3 = E.waypoint
    waypoint4 = E.waypoint
    waypoint5 = E.waypoint
    waypoint6 = E.waypoint
    waypoint7 = E.waypoint
    waypoint8 = E.waypoint
    waypoint9 = E.waypoint
    waypoint10 = E.waypoint
    waypoint11 = E.waypoint
    the_doc = ROOT(
                route(id = scenarios.entities[0].properties[0], map=scenarios.entities[0].properties[1]),
                weather(cloudiness = scenarios.entities[1].properties[0], precipitation = scenarios.entities[1].properties[1],
                precipitation_deposits = scenarios.entities[1].properties[2], wind_intensity = scenarios.entities[1].properties[3],
                sun_azimuth_angle = scenarios.entities[1].properties[4], sun_altitude_angle = scenarios.entities[1].properties[5],
                wetness = scenarios.entities[1].properties[6], fog_distance = scenarios.entities[1].properties[7], fog_density = scenarios.entities[1].properties[8]),
                waypoint1(pitch = scenarios.entities[2].properties[0][0], roll = scenarios.entities[2].properties[0][1],
                x = scenarios.entities[2].properties[0][2], y = scenarios.entities[2].properties[0][3], yaw = scenarios.entities[2].properties[0][4],
                z = scenarios.entities[2].properties[0][5]),
                waypoint2(pitch = scenarios.entities[2].properties[1][0], roll = scenarios.entities[2].properties[1][1],
                x = scenarios.entities[2].properties[1][2], y = scenarios.entities[2].properties[1][3], yaw = scenarios.entities[2].properties[1][4],
                z = scenarios.entities[2].properties[1][5]),
                waypoint3(pitch = scenarios.entities[2].properties[2][0], roll = scenarios.entities[2].properties[2][1],
                x = scenarios.entities[2].properties[2][2], y = scenarios.entities[2].properties[2][3], yaw = scenarios.entities[2].properties[2][4],
                z = scenarios.entities[2].properties[2][5]),
                waypoint4(pitch = scenarios.entities[2].properties[3][0], roll = scenarios.entities[2].properties[3][1],
                x = scenarios.entities[2].properties[3][2], y = scenarios.entities[2].properties[3][3], yaw = scenarios.entities[2].properties[3][4],
                z = scenarios.entities[2].properties[0][5]),
                waypoint5(pitch = scenarios.entities[2].properties[4][0], roll = scenarios.entities[2].properties[4][1],
                x = scenarios.entities[2].properties[4][2], y = scenarios.entities[2].properties[4][3], yaw = scenarios.entities[2].properties[4][4],
                z = scenarios.entities[2].properties[4][5]),
                waypoint6(pitch = scenarios.entities[2].properties[5][0], roll = scenarios.entities[2].properties[5][1],
                x = scenarios.entities[2].properties[5][2], y = scenarios.entities[2].properties[5][3], yaw = scenarios.entities[2].properties[5][4],
                z = scenarios.entities[2].properties[5][5]),
                waypoint7(pitch = scenarios.entities[2].properties[6][0], roll = scenarios.entities[2].properties[6][1],
                x = scenarios.entities[2].properties[6][2], y = scenarios.entities[2].properties[6][3], yaw = scenarios.entities[2].properties[6][4],
                z = scenarios.entities[2].properties[6][5]),
                waypoint8(pitch = scenarios.entities[2].properties[7][0], roll = scenarios.entities[2].properties[7][1],
                x = scenarios.entities[2].properties[7][2], y = scenarios.entities[2].properties[7][3], yaw = scenarios.entities[2].properties[7][4],
                z = scenarios.entities[2].properties[7][5]),
                waypoint9(pitch = scenarios.entities[2].properties[8][0], roll = scenarios.entities[2].properties[8][1],
                x = scenarios.entities[2].properties[8][2], y = scenarios.entities[2].properties[8][3], yaw = scenarios.entities[2].properties[8][4],
                z = scenarios.entities[2].properties[8][5]),
                waypoint10(pitch = scenarios.entities[2].properties[9][0], roll = scenarios.entities[2].properties[9][1],
                x = scenarios.entities[2].properties[9][2], y = scenarios.entities[2].properties[9][3], yaw = scenarios.entities[2].properties[9][4],
                z = scenarios.entities[2].properties[9][5]),
                waypoint11(pitch = scenarios.entities[2].properties[10][0], roll = scenarios.entities[2].properties[10][1],
                x = scenarios.entities[2].properties[10][2], y = scenarios.entities[2].properties[10][3], yaw = scenarios.entities[2].properties[10][4],
                z = scenarios.entities[2].properties[10][5]),
            )

    print(lxml.etree.tostring(the_doc, pretty_print=True))
    tree = lxml.etree.ElementTree(the_doc)
    tree.write('output.xml', pretty_print=True, xml_declaration=True, encoding="utf-8")



if __name__ == '__main__':
        id = "17"
        global_route = [["360.0","0.0","338.7027893066406","226.75003051757812","269.9790954589844","0.0"],["360.0","0.0","321.98931884765625","194.67242431640625","179.83230590820312","0.0"],
        ["360.0","0.0","283.6903991699219","194.78451538085938","179.83230590820312","0.0"],
        ["360.0","0.0","108.0505142211914","195.29856872558594","179.83230590820312","0.0"],["0.0","0.0","88.40200805664062","210.57827758789062","89.99128723144531","0.0"],
        ["0.0","0.0","88.41706848144531","309.6344299316406","89.99128723144531","0.0"],
        ["360.0","0.0","75.58748626708984","326.3004455566406","180.0352020263672","0.0"],["360.0","0.0","14.334035873413086","326.2628173828125","180.0352020263672","0.0"],
        ["360.0","0.0","1.8717632293701172","299.4347229003906","269.8846435546875","0.0"],
        ["360.0","0.0","1.612621784210205","170.71238708496094","269.8846435546875","0.0"],["360.0","0.0","1.3654530048370361","47.93744659423828","269.8846435546875","0.0"]]
        other_agent_route = ["360.0","0.0","321.98931884765625","194.67242431640625","179.83230590820312","0.0"]
        other_agent_numbers = 1
        scenario_meta = metamodel_from_file('entity.tx')
        scenarios = scenario_meta.model_from_file('scenario.entity')
        for i in range(4):
            scenario_town_description(scenarios,id)
            scenario_weather(scenarios)
            scenario_agent_route(scenarios,global_route)
            ego_agent_info(scenarios,global_route)
            other_agent_info(scenarios, other_agent_numbers,other_agent_route)
            XML_generator(scenarios)
