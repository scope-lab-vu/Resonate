#!/usr/bin/env python2
""" Print total cumulative serialized msg size per topic in a ROS Bag file """
import rosbag
import sys

# Calculate size of messages in bag file
topic_size_dict = {}
for topic, msg, time in rosbag.Bag(sys.argv[1], 'r').read_messages(raw=True):
  topic_size_dict[topic] = topic_size_dict.get(topic, 0) + len(msg[1])
topic_size = list(topic_size_dict.items())


# Print results from smallest to largest
topic_size.sort(key=lambda x: x[1])
print("TOPIC\t\t\tSIZE")
for topic, size in topic_size:
  # For more human readable results, determine scale of each value (Kilo, Mega, Giga, Tera)
  order = 0
  while size > 1000:
    size /= 1000
    order += 1
  order_str = ['', 'K', 'M', 'G', 'T'][order]
  print("%s  %d%s" % (topic, size, order_str))
