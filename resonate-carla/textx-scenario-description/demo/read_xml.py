import xml.etree.ElementTree as ET



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

if __name__ == '__main__':
    parse_routes_file('/home/scope/Carla/ICCPS_CARLA_challenge/leaderboard/data/routes/route_17.xml',False)
