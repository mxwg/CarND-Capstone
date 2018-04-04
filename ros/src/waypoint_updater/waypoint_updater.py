#!/usr/bin/env python

import rospy
import numpy as np
from scipy.spatial import KDTree
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from visualization_msgs.msg import Marker, MarkerArray

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100  # Number of waypoints we will publish. You can change this number


def waypointToMarker(waypoint, frame_id, ts=rospy.Time(0), idx=0, color=[0.0, 1.0, 0.0]):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = ts
    marker.id = idx
    marker.pose.position = waypoint.pose.pose.position
    marker.pose.position.z = 2  # show above other markers...
    marker.pose.orientation = waypoint.pose.pose.orientation
    marker.scale.x = 5
    marker.scale.y = 1
    marker.scale.z = 1
    marker.color.a = 1
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    return marker


def stopMarker(waypoint, frame_id, ts=rospy.Time(0), idx=0):
    scale = 25
    color = [1.0, 0.0, 0.0]

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = ts
    marker.id = idx
    marker.type = Marker.CUBE
    marker.pose.position = waypoint.pose.pose.position
    marker.pose.orientation = waypoint.pose.pose.orientation
    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale
    marker.color.a = 1
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    return marker


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.get_all_waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        # Publishers
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.final_waypoints_marker_pub = rospy.Publisher('final_waypoints_markers', MarkerArray, queue_size=1)

        # Current state
        self.car_pose_x = None
        self.car_pose_y = None
        self.waypoints = None
        self.n_wp = 0
        self.n_lookahead_wp = LOOKAHEAD_WPS
        self.nearest_idx = None
        self.wp_tree = None
        self.next_traffic_light_idx = -1

        rospy.spin()

    def pose_cb(self, msg):
        """Update the next waypoints using the current car pose."""
        self.car_pose_x = msg.pose.position.x
        self.car_pose_y = msg.pose.position.y

        if not self.waypoints:
            return  # not yet ready

        self.update_nearest_waypoint()
        waypoints = self.set_velocities()
        self.publish_final_waypoints(waypoints)

    def set_velocities(self):
        """Set the velocities of the next waypoints."""
        waypoints = []
        for i in range(self.n_lookahead_wp):
            wp_orig = self.nearest(offset=i)
            wp = Waypoint()
            wp.pose = wp_orig.pose
            wp.twist.twist.linear.x = wp_orig.twist.twist.linear.x
            if self.next_traffic_light_idx != -1 and self.next_traffic_light_idx - self.nearest_idx <= self.n_lookahead_wp*2:
                dist_to_stoplight = self.distance(self.waypoints, self.nearest_idx + i, self.next_traffic_light_idx)
                vel = math.sqrt(2 * 0.5 * dist_to_stoplight)
                if vel < 2.0:
                    vel = 0.0
                wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)  # don't set a vel higher than it was
            waypoints.append(wp)
        return waypoints

    def publish_final_waypoints(self, waypoints, visualize=True):
        """Publish the next waypoints."""
        lane = Lane()
        lane.header.frame_id = 'world'
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = waypoints

        self.final_waypoints_pub.publish(lane)
        if visualize:
            self.publish_markers()

    def publish_markers(self):
        """Publish markers for the next waypoints."""
        green = [0.0, 1.0, 0.0]
        yellow = [1.0, 1.0, 0.0]
        array = MarkerArray()

        for i in range(self.n_lookahead_wp):
            if i % (self.n_lookahead_wp / 10) == 0:  # only publish a few waypoints as a marker
                array.markers.append(
                    waypointToMarker(self.nearest(i), 'world', ts=rospy.Time.now(), idx=i, color=yellow))

        if self.next_traffic_light_idx != -1:
            array.markers.append(stopMarker(self.waypoints[self.next_traffic_light_idx],
                                            'world', ts=rospy.Time.now(), idx=len(array.markers)))
        self.final_waypoints_marker_pub.publish(array)

    def nearest(self, offset=0):
        """Get current nearest waypoint."""
        return self.waypoints[(self.nearest_idx + offset) % self.n_wp]

    def update_nearest_waypoint(self):
        """Find the index of the currently nearest waypoint."""
        if not self.wp_tree:
            self.nearest_idx = 0
            return
        nearest_idx = self.wp_tree.query([self.car_pose_x, self.car_pose_y], 1)[1]

        # check if the closest waypoint is behind the car
        nearest_wp = self.waypoints[nearest_idx].pose.pose.position
        prev_wp = self.waypoints[nearest_idx - 1].pose.pose.position

        nearest = np.array([nearest_wp.x, nearest_wp.y])
        prev = np.array([prev_wp.x, prev_wp.y])
        car = np.array([self.car_pose_x, self.car_pose_y])

        if np.dot(nearest - prev, car - nearest) > 0:  # waypoint is behind car
            self.nearest_idx = (nearest_idx + 1) % self.n_wp

        # rospy.logdebug("Nearest waypoint: {}, dist: {}".format(self.nearest_waypoint_idx, self.dist_to(self.nearest())))

    def dist_to(self, waypoint):
        """Compute distance from current car position to a waypoint."""
        wp = waypoint.pose.pose.position
        return math.sqrt((self.car_pose_x - wp.x) ** 2 + (self.car_pose_y - wp.y) ** 2)

    def get_all_waypoints_cb(self, lane):
        """Subscribe to the list of all waypoints."""
        self.waypoints = lane.waypoints
        self.n_wp = len(self.waypoints)
        if self.n_wp < self.n_lookahead_wp:
            self.n_lookahead_wp = self.n_wp / 4  # only look 25% of all waypoints ahead

        self.wp_tree = KDTree([[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in self.waypoints])
        rospy.loginfo("Received {} waypoints...".format(self.n_wp))

    def traffic_cb(self, msg):
        self.next_traffic_light_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
