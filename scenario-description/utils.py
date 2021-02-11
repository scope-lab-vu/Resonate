from xml.etree import ElementTree
import xml.etree.ElementTree as ET
import lxml.etree
import lxml.builder


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


        return waypoint_list, route_town


def XML_generator(scenarios,i,folder):

    E = lxml.builder.ElementMaker()
    ROOT = E.routes
    ROUTE = E.route
    route = E.map
    route1 = E.id
    route2 = E.weather
    route3 = E.waypoint
    route4 = E.waypoint
    route5 = E.waypoint
    route6 = E.waypoint
    route7 = E.waypoint
    route8 = E.waypoint
    route9 = E.waypoint
    route10 = E.waypoint
    route11 = E.waypoint
    route12 = E.waypoint
    route13 = E.waypoint
    route14 = E.waypoint
    route15 = E.waypoint
    route16 = E.waypoint
    route17 = E.waypoint

    the_doc = ROOT(
                    ROUTE(
                route(map=str(scenarios.entities[0].properties[1])),
                route1(id=str(scenarios.entities[0].properties[0])),
                route2(cloudiness = scenarios.entities[1].properties[0], precipitation = scenarios.entities[1].properties[1],
                precipitation_deposits = scenarios.entities[1].properties[2], wind_intensity = scenarios.entities[1].properties[3],
                sun_azimuth_angle = scenarios.entities[1].properties[4], sun_altitude_angle = scenarios.entities[1].properties[5],
                wetness = scenarios.entities[1].properties[6], fog_distance = scenarios.entities[1].properties[7], fog_density = scenarios.entities[1].properties[8]),
                route3(pitch = scenarios.entities[2].properties[0][0], roll = scenarios.entities[2].properties[0][1],
                x = scenarios.entities[2].properties[0][2], y = scenarios.entities[2].properties[0][3], yaw = scenarios.entities[2].properties[0][4],
                z = scenarios.entities[2].properties[0][5]),
                route4(pitch = scenarios.entities[2].properties[1][0], roll = scenarios.entities[2].properties[1][1],
                x = scenarios.entities[2].properties[1][2], y = scenarios.entities[2].properties[1][3], yaw = scenarios.entities[2].properties[1][4],
                z = scenarios.entities[2].properties[1][5]),
                route5(pitch = scenarios.entities[2].properties[2][0], roll = scenarios.entities[2].properties[2][1],
                x = scenarios.entities[2].properties[2][2], y = scenarios.entities[2].properties[2][3], yaw = scenarios.entities[2].properties[2][4],
                z = scenarios.entities[2].properties[2][5]),
                #route6(pitch = scenarios.entities[2].properties[3][0], roll = scenarios.entities[2].properties[3][1],
                #x = scenarios.entities[2].properties[3][2], y = scenarios.entities[2].properties[3][3], yaw = scenarios.entities[2].properties[3][4],
                #z = scenarios.entities[2].properties[0][5]),
                # route7(pitch = scenarios.entities[2].properties[4][0], roll = scenarios.entities[2].properties[4][1],
                # x = scenarios.entities[2].properties[4][2], y = scenarios.entities[2].properties[4][3], yaw = scenarios.entities[2].properties[4][4],
                # z = scenarios.entities[2].properties[4][5]),
                # route8(pitch = scenarios.entities[2].properties[5][0], roll = scenarios.entities[2].properties[5][1],
                # x = scenarios.entities[2].properties[5][2], y = scenarios.entities[2].properties[5][3], yaw = scenarios.entities[2].properties[5][4],
                # z = scenarios.entities[2].properties[5][5]),
                # route9(pitch = scenarios.entities[2].properties[6][0], roll = scenarios.entities[2].properties[6][1],
                # x = scenarios.entities[2].properties[6][2], y = scenarios.entities[2].properties[6][3], yaw = scenarios.entities[2].properties[6][4],
                # z = scenarios.entities[2].properties[6][5]),
                # route10(pitch = scenarios.entities[2].properties[7][0], roll = scenarios.entities[2].properties[7][1],
                # x = scenarios.entities[2].properties[7][2], y = scenarios.entities[2].properties[7][3], yaw = scenarios.entities[2].properties[7][4],
                # z = scenarios.entities[2].properties[7][5]),
                # route11(pitch = scenarios.entities[2].properties[8][0], roll = scenarios.entities[2].properties[8][1],
                # x = scenarios.entities[2].properties[8][2], y = scenarios.entities[2].properties[8][3], yaw = scenarios.entities[2].properties[8][4],
                # z = scenarios.entities[2].properties[8][5]),
                # route12(pitch = scenarios.entities[2].properties[9][0], roll = scenarios.entities[2].properties[9][1],
                # x = scenarios.entities[2].properties[9][2], y = scenarios.entities[2].properties[9][3], yaw = scenarios.entities[2].properties[9][4],
                # z = scenarios.entities[2].properties[9][5]),
                # route13(pitch = scenarios.entities[2].properties[10][0], roll = scenarios.entities[2].properties[10][1],
                # x = scenarios.entities[2].properties[10][2], y = scenarios.entities[2].properties[10][3], yaw = scenarios.entities[2].properties[10][4],
                # z = scenarios.entities[2].properties[10][5]),
                # route14(pitch = scenarios.entities[2].properties[11][0], roll = scenarios.entities[2].properties[11][1],
                # x = scenarios.entities[2].properties[11][2], y = scenarios.entities[2].properties[11][3], yaw = scenarios.entities[2].properties[11][4],
                # z = scenarios.entities[2].properties[11][5]),
                # route15(pitch = scenarios.entities[2].properties[12][0], roll = scenarios.entities[2].properties[12][1],
                # x = scenarios.entities[2].properties[12][2], y = scenarios.entities[2].properties[12][3], yaw = scenarios.entities[2].properties[12][4],
                # z = scenarios.entities[2].properties[12][5]),
                # route16(pitch = scenarios.entities[2].properties[13][0], roll = scenarios.entities[2].properties[13][1],
                # x = scenarios.entities[2].properties[13][2], y = scenarios.entities[2].properties[13][3], yaw = scenarios.entities[2].properties[13][4],
                # z = scenarios.entities[2].properties[13][5]),
                # route17(pitch = scenarios.entities[2].properties[14][0], roll = scenarios.entities[2].properties[14][1],
                # x = scenarios.entities[2].properties[14][2], y = scenarios.entities[2].properties[14][3], yaw = scenarios.entities[2].properties[14][4],
                # z = scenarios.entities[2].properties[14][5]),
                )
                )
    print(lxml.etree.tostring(the_doc, pretty_print=True))
    tree = lxml.etree.ElementTree(the_doc)
    tree.write(folder+'/%d.xml'%i, pretty_print=True, xml_declaration=True, encoding="utf-8")
