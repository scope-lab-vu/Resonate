#!/usr/bin/python3
import textx
import numpy as np
import lxml.etree
import lxml.builder
import sys
import glob
from xml.etree import ElementTree
import xml.etree.ElementTree as ET

from textx.metamodel import metamodel_from_file


def parse_routes_file(route_filename, single_route=None):
        """
        Returns a list of route elements that is where the challenge is going to happen.
        :param route_filename: the path to a set of routes.
        :param single_route: If set, only this route shall be returned
        :return:  List of dicts containing the waypoints, id and town of the routes
        """

        list_route_descriptions = []
        tree = ET.parse(route_filename)
        for route in tree.iter("route"):
            route_town = route.attrib['map']
            route_id = route.attrib['id']
            #route_weather = RouteParser.parse_weather(route)
            if single_route and route_id != single_route:
                continue

            waypoint_list = []  # the list of waypoints that can be found on this route
            for waypoint in route.iter('waypoint'):
                waypoint_list.append([waypoint.attrib['pitch'],waypoint.attrib['roll'],waypoint.attrib['x'],waypoint.attrib['y'],waypoint.attrib['yaw'],waypoint.attrib['z']])

                # Waypoints is basically a list of XML nodes

            # list_route_descriptions.append({
            #     'id': route_id,
            #     'town_name': route_town,
            #     'trajectory': waypoint_list,
            #     #'weather': route_weather
            # })

        #print(waypoint_list)

        return waypoint_list

def scenario_town_description(scenarios,id):
    scenarios.entities[0].properties[0] = id
    scenarios.entities[0].properties[1] = "Town03"

    #print(scenarios.entities[0].properties)

    #return scenarios

def scenario_weather(scenarios):
    # for i in range(len(scenarios.entities[1].properties)):
    #     print(str(scenarios.entities[1].properties[i].name))
    #     if(str(scenarios.entities[1].properties[i].name) == "sun_altitude_angle"):
    #              scenarios.entities[1].properties[i] = "70.0"
    #     if (scenarios.entities[1].properties[i].type.name == "uniform"):
    #             scenarios.entities[1].properties[i] = str(np.random.uniform(0,100))
    #     elif(scenarios.entities[1].properties[i].type.name == "string"):
    #             scenarios.entities[1].properties[i] = "0.0"


    scenarios.entities[1].properties[0] = str(np.random.uniform(0,100))
    scenarios.entities[1].properties[1] = str(np.random.uniform(0,100))
    scenarios.entities[1].properties[2] = str(np.random.uniform(0,100))
    scenarios.entities[1].properties[3] = "0.0"
    scenarios.entities[1].properties[4] = "0.0"
    scenarios.entities[1].properties[5] = "70.0"
    scenarios.entities[1].properties[6] =  "0.0"
    scenarios.entities[1].properties[7] =  "0.0"
    scenarios.entities[1].properties[8] =  "0.0"

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

def XML_generator(scenarios,i):

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
    waypoint12 = E.waypoint
    waypoint13 = E.waypoint
    waypoint14 = E.waypoint
    waypoint15 = E.waypoint
    waypoint16 = E.waypoint
    waypoint17 = E.waypoint
    waypoint18 = E.waypoint
    waypoint19 = E.waypoint
    waypoint20 = E.waypoint
    waypoint21 = E.waypoint
    waypoint22 = E.waypoint
    waypoint23 = E.waypoint
    waypoint24 = E.waypoint
    waypoint25 = E.waypoint
    waypoint26 = E.waypoint
    waypoint27 = E.waypoint
    waypoint28 = E.waypoint
    waypoint29 = E.waypoint
    waypoint30 = E.waypoint
    waypoint31 = E.waypoint
    waypoint32 = E.waypoint
    waypoint33 = E.waypoint
    waypoint34 = E.waypoint
    waypoint35 = E.waypoint
    waypoint36 = E.waypoint
    waypoint37 = E.waypoint
    waypoint38 = E.waypoint
    waypoint39 = E.waypoint
    waypoint40 = E.waypoint
    waypoint41 = E.waypoint

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
                waypoint12(pitch = scenarios.entities[2].properties[11][0], roll = scenarios.entities[2].properties[11][1],
                x = scenarios.entities[2].properties[11][2], y = scenarios.entities[2].properties[11][3], yaw = scenarios.entities[2].properties[11][4],
                z = scenarios.entities[2].properties[11][5]),
                waypoint13(pitch = scenarios.entities[2].properties[12][0], roll = scenarios.entities[2].properties[12][1],
                x = scenarios.entities[2].properties[12][2], y = scenarios.entities[2].properties[12][3], yaw = scenarios.entities[2].properties[12][4],
                z = scenarios.entities[2].properties[12][5]),
                waypoint14(pitch = scenarios.entities[2].properties[13][0], roll = scenarios.entities[2].properties[13][1],
                x = scenarios.entities[2].properties[13][2], y = scenarios.entities[2].properties[13][3], yaw = scenarios.entities[2].properties[13][4],
                z = scenarios.entities[2].properties[13][5]),
                waypoint15(pitch = scenarios.entities[2].properties[14][0], roll = scenarios.entities[2].properties[14][1],
                x = scenarios.entities[2].properties[14][2], y = scenarios.entities[2].properties[14][3], yaw = scenarios.entities[2].properties[14][4],
                z = scenarios.entities[2].properties[14][5]),
                # waypoint16(pitch = scenarios.entities[2].properties[15][0], roll = scenarios.entities[2].properties[15][1],
                # x = scenarios.entities[2].properties[15][2], y = scenarios.entities[2].properties[15][3], yaw = scenarios.entities[2].properties[15][4],
                # z = scenarios.entities[2].properties[15][5]),
                # waypoint17(pitch = scenarios.entities[2].properties[16][0], roll = scenarios.entities[2].properties[16][1],
                # x = scenarios.entities[2].properties[16][2], y = scenarios.entities[2].properties[16][3], yaw = scenarios.entities[2].properties[16][4],
                # z = scenarios.entities[2].properties[16][5]),
                # waypoint18(pitch = scenarios.entities[2].properties[17][0], roll = scenarios.entities[2].properties[17][1],
                # x = scenarios.entities[2].properties[17][2], y = scenarios.entities[2].properties[17][3], yaw = scenarios.entities[2].properties[17][4],
                # z = scenarios.entities[2].properties[17][5]),
                # waypoint19(pitch = scenarios.entities[2].properties[18][0], roll = scenarios.entities[2].properties[18][1],
                # x = scenarios.entities[2].properties[18][2], y = scenarios.entities[2].properties[18][3], yaw = scenarios.entities[2].properties[18][4],
                # z = scenarios.entities[2].properties[18][5]),
                # waypoint20(pitch = scenarios.entities[2].properties[19][0], roll = scenarios.entities[2].properties[19][1],
                # x = scenarios.entities[2].properties[19][2], y = scenarios.entities[2].properties[19][3], yaw = scenarios.entities[2].properties[19][4],
                # z = scenarios.entities[2].properties[19][5]),
                # waypoint21(pitch = scenarios.entities[2].properties[20][0], roll = scenarios.entities[2].properties[20][1],
                # x = scenarios.entities[2].properties[20][2], y = scenarios.entities[2].properties[20][3], yaw = scenarios.entities[2].properties[20][4],
                # z = scenarios.entities[2].properties[20][5]),
                # waypoint22(pitch = scenarios.entities[2].properties[21][0], roll = scenarios.entities[2].properties[21][1],
                # x = scenarios.entities[2].properties[21][2], y = scenarios.entities[2].properties[21][3], yaw = scenarios.entities[2].properties[21][4],
                # z = scenarios.entities[2].properties[21][5]),
                # waypoint23(pitch = scenarios.entities[2].properties[22][0], roll = scenarios.entities[2].properties[22][1],
                # x = scenarios.entities[2].properties[22][2], y = scenarios.entities[2].properties[22][3], yaw = scenarios.entities[2].properties[22][4],
                # z = scenarios.entities[2].properties[22][5]),
                # waypoint24(pitch = scenarios.entities[2].properties[23][0], roll = scenarios.entities[2].properties[23][1],
                # x = scenarios.entities[2].properties[23][2], y = scenarios.entities[2].properties[23][3], yaw = scenarios.entities[2].properties[23][4],
                # z = scenarios.entities[2].properties[23][5]),
                # waypoint25(pitch = scenarios.entities[2].properties[24][0], roll = scenarios.entities[2].properties[24][1],
                # x = scenarios.entities[2].properties[24][2], y = scenarios.entities[2].properties[24][3], yaw = scenarios.entities[2].properties[24][4],
                # z = scenarios.entities[2].properties[24][5]),
                # waypoint26(pitch = scenarios.entities[2].properties[25][0], roll = scenarios.entities[2].properties[25][1],
                # x = scenarios.entities[2].properties[25][2], y = scenarios.entities[2].properties[25][3], yaw = scenarios.entities[2].properties[25][4],
                # z = scenarios.entities[2].properties[25][5]),
                # waypoint27(pitch = scenarios.entities[2].properties[26][0], roll = scenarios.entities[2].properties[26][1],
                # x = scenarios.entities[2].properties[26][2], y = scenarios.entities[2].properties[26][3], yaw = scenarios.entities[2].properties[26][4],
                # z = scenarios.entities[2].properties[26][5]),
                # waypoint28(pitch = scenarios.entities[2].properties[27][0], roll = scenarios.entities[2].properties[27][1],
                # x = scenarios.entities[2].properties[27][2], y = scenarios.entities[2].properties[27][3], yaw = scenarios.entities[2].properties[27][4],
                # z = scenarios.entities[2].properties[27][5]),
                # waypoint29(pitch = scenarios.entities[2].properties[28][0], roll = scenarios.entities[2].properties[28][1],
                # x = scenarios.entities[2].properties[28][2], y = scenarios.entities[2].properties[28][3], yaw = scenarios.entities[2].properties[28][4],
                # z = scenarios.entities[2].properties[28][5]),
                # waypoint30(pitch = scenarios.entities[2].properties[29][0], roll = scenarios.entities[2].properties[29][1],
                # x = scenarios.entities[2].properties[29][2], y = scenarios.entities[2].properties[29][3], yaw = scenarios.entities[2].properties[29][4],
                # z = scenarios.entities[2].properties[29][5]),
                # waypoint31(pitch = scenarios.entities[2].properties[30][0], roll = scenarios.entities[2].properties[30][1],
                # x = scenarios.entities[2].properties[30][2], y = scenarios.entities[2].properties[30][3], yaw = scenarios.entities[2].properties[30][4],
                # z = scenarios.entities[2].properties[30][5]),
                # waypoint32(pitch = scenarios.entities[2].properties[31][0], roll = scenarios.entities[2].properties[31][1],
                # x = scenarios.entities[2].properties[31][2], y = scenarios.entities[2].properties[31][3], yaw = scenarios.entities[2].properties[31][4],
                # z = scenarios.entities[2].properties[31][5]),
                # waypoint33(pitch = scenarios.entities[2].properties[32][0], roll = scenarios.entities[2].properties[32][1],
                # x = scenarios.entities[2].properties[32][2], y = scenarios.entities[2].properties[32][3], yaw = scenarios.entities[2].properties[32][4],
                # z = scenarios.entities[2].properties[32][5]),
                # waypoint34(pitch = scenarios.entities[2].properties[33][0], roll = scenarios.entities[2].properties[33][1],
                # x = scenarios.entities[2].properties[33][2], y = scenarios.entities[2].properties[33][3], yaw = scenarios.entities[2].properties[33][4],
                # z = scenarios.entities[2].properties[33][5]),
                # waypoint35(pitch = scenarios.entities[2].properties[34][0], roll = scenarios.entities[2].properties[34][1],
                # x = scenarios.entities[2].properties[34][2], y = scenarios.entities[2].properties[34][3], yaw = scenarios.entities[2].properties[34][4],
                # z = scenarios.entities[2].properties[34][5]),
                # waypoint36(pitch = scenarios.entities[2].properties[35][0], roll = scenarios.entities[2].properties[35][1],
                # x = scenarios.entities[2].properties[35][2], y = scenarios.entities[2].properties[35][3], yaw = scenarios.entities[2].properties[35][4],
                # z = scenarios.entities[2].properties[35][5]),
                # waypoint37(pitch = scenarios.entities[2].properties[36][0], roll = scenarios.entities[2].properties[36][1],
                # x = scenarios.entities[2].properties[36][2], y = scenarios.entities[2].properties[36][3], yaw = scenarios.entities[2].properties[36][4],
                # z = scenarios.entities[2].properties[36][5]),
                # waypoint38(pitch = scenarios.entities[2].properties[37][0], roll = scenarios.entities[2].properties[37][1],
                # x = scenarios.entities[2].properties[37][2], y = scenarios.entities[2].properties[37][3], yaw = scenarios.entities[2].properties[37][4],
                # z = scenarios.entities[2].properties[37][5]),
                # waypoint39(pitch = scenarios.entities[2].properties[38][0], roll = scenarios.entities[2].properties[38][1],
                # x = scenarios.entities[2].properties[38][2], y = scenarios.entities[2].properties[38][3], yaw = scenarios.entities[2].properties[38][4],
                # z = scenarios.entities[2].properties[38][5]),
                # waypoint40(pitch = scenarios.entities[2].properties[39][0], roll = scenarios.entities[2].properties[39][1],
                # x = scenarios.entities[2].properties[39][2], y = scenarios.entities[2].properties[39][3], yaw = scenarios.entities[2].properties[39][4],
                # z = scenarios.entities[2].properties[39][5]),
                # waypoint41(pitch = scenarios.entities[2].properties[40][0], roll = scenarios.entities[2].properties[40][1],
                # x = scenarios.entities[2].properties[40][2], y = scenarios.entities[2].properties[40][3], yaw = scenarios.entities[2].properties[40][4],
                # z = scenarios.entities[2].properties[40][5]),
            )

    print(lxml.etree.tostring(the_doc, pretty_print=True))
    tree = lxml.etree.ElementTree(the_doc)
    tree.write('/home/scope/Carla/textx-trial/demo/XML_files/short-17%d.xml'%i, pretty_print=True, xml_declaration=True, encoding="utf-8")


if __name__ == '__main__':
        id = "17"
        # global_route = [["0.0","0.0","236.03851318359375","-37.81154251098633","91.3932113647461","0.0"],
        # ["0.0","0.0","234.3028564453125","33.55348205566406","91.3932113647461","0.0"],
        # ["0.5513595342636108","0.0","233.07542419433594","84.020751953125","91.3932113647461","0.11305883526802063"],
        # ["1.1732286214828491","0.0","232.1027374267578","124.01483917236328","91.3932113647461","0.7884235978126526"],
        # ["1.1732286214828491","0.0","231.76669311523438","137.8319549560547","91.3932113647461","1.0714359283447266"],
        # ["1.1732286214828491","0.0","224.5847930908203","163.8441162109375","120.6165542602539","1.685321569442749"],
        # ["1.1732286214828491","0.0","207.194580078125","181.5075225830078","148.15965270996094","2.249598741531372"],
        # ["0.48729410767555237","0.0","181.1305694580078","192.95262145996094","164.4263153076172","2.8146581649780273"],
        # ["-1.274809718132019","0.0","141.6644744873047","196.67758178710938","179.85704040527344","2.2296857833862305"],
        # ["-1.274809718132019","0.0","81.40699005126953","196.82794189453125","179.85704040527344","0.8889751434326172"],
        # ["0.0","0.0","15.399579048156738","196.99264526367188","179.85704040527344","0.0"],
        # ["0.0","0.0","-35.3316764831543","197.1192169189453","179.85704040527344","0.0"],
        # ["0.0","0.0","-67.58390045166016","185.36460876464844","229.0654296875","0.0"],
        # ["0.0","0.0","-76.3777847290039","169.23289489746094","253.7422637939453","0.0"],
        # ["0.0","0.0","-78.1178970336914","151.60719299316406","-90.21229553222656","0.0"],
        # ["360.5830993652344","0.0","-77.54217529296875","118.35594177246094","269.84375","0.08791226893663406"],
        # ["361.01251220703125","0.0","-77.60916900634766","93.79161834716797","269.84375","0.47433051466941833"],
        # ["359.6525573730469","0.0","-77.6905288696289","63.95880889892578","269.84375","0.8224835395812988"],
        # ["359.4696960449219","0.0","-77.9049072265625","-14.645129203796387","269.84375","-0.0876927524805069"],
        # ["359.160400390625","0.0","-78.02450561523438","-58.49807357788086","269.84375","-0.7004829049110413"],
        # ["360.79266357421875","0.0","-78.19097900390625","-119.54064178466797","269.84375","-0.12565355002880096"],
        # ["1.0470895767211914","0.0","-100.9601058959961","-140.4352264404297","-178.77273559570312","0.214286670088768"],
        # ["1.1378107070922852","0.0","-123.3655776977539","-135.28465270996094","-207.3763885498047","0.7149972319602966"],
        # ["-0.3563445210456848","0.0","-140.6134796142578","-119.8656005859375","-235.88018798828125","0.868480920791626"],
        # ["-0.6521514654159546","0.0","-148.46490478515625","-99.57848358154297","-261.8055114746094","0.6392691731452942"],
        # ["-0.6521514654159546","0.0","-148.96641540527344","-80.2991714477539","90.02985382080078","0.4223979115486145"],
        # ["-0.23242473602294922","0.0","-148.9879913330078","-38.88064956665039","90.02985382080078","0.021686267107725143"],
        # ["0.0","0.0","-148.9995880126953","-16.62619972229004","90.02985382080078","0.0"],
        # ["0.0","0.0","-149.01351928710938","10.073049545288086","90.02985382080078","0.0"],
        # ["0.0","0.0","-149.05345153808594","86.74226379394531","90.02985382080078","0.0"],
        # ["0.0","0.0","-138.78024291992188","130.7232666015625","40.636573791503906","0.0"],
        # ["0.0","0.0","-107.38884735107422","136.4002227783203","-1.296802282333374","0.0"],
        # ["0.0","0.0","-61.1401252746582","135.35328674316406","-1.296802282333374","0.0"],
        # ["0.0","0.0","-26.570899963378906","134.5707244873047","-1.296802282333374","0.0"],
        # ["0.0","0.0","-9.406842231750488","147.98507690429688","89.63746643066406","0.0"],
        # ["0.0","0.0","-9.188850402832031","182.43663024902344","89.63746643066406","0.0"],
        # ["0.0","0.0","-32.6129264831543","193.6124267578125","179.85704040527344","0.0"],
        # ["0.0","0.0","-47.37412643432617","193.12583923339844","190.4706268310547","0.0"],
        # ["0.0","0.0","-62.72514343261719","185.37353515625","-136.88282775878906","0.0"],
        # ["0.0","0.0","-74.6160888671875","152.08935546875","-90.21229553222656","0.0"],
        # ["360.59765625","0.0","-74.04336547851562","117.91499328613281","269.84375","0.09235748648643494"]]

        other_agent_route = ["360.0","0.0","321.98931884765625","194.67242431640625","179.83230590820312","0.0"]
        other_agent_numbers = 1
        folder = "/home/scope/Carla/textx-trial/demo/XML_files"
        scenario_meta = metamodel_from_file('entity.tx')
        scenarios = scenario_meta.model_from_file('scenario.entity')
        global_route = parse_routes_file('/home/scope/Carla/ICCPS_CARLA_challenge/leaderboard/data/routes/route_17.xml',False)
        for i in range(10):
            scenario_town_description(scenarios,id)
            scenario_weather(scenarios)
            scenario_agent_route(scenarios,global_route)
            ego_agent_info(scenarios,global_route)
            other_agent_info(scenarios, other_agent_numbers,other_agent_route)
            print("Run%d complete"%i)
            XML_generator(scenarios,i)
        #XML_Merger(folder)
