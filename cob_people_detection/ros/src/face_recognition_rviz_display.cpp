#include <ros/ros.h>

#include <people_msgs/PositionMeasurementArray.h>
#include <geometry_msgs/Point.h>
#include <people_msgs/PositionMeasurement.h>
#include <cob_perception_msgs/DetectionArray.h>
#include <cob_perception_msgs/ColorDepthImageArray.h>
#include <pcl_ros/point_cloud.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <boost/bind.hpp>
#include <boost/thread/mutex.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <vector>
#include <iostream>

class FaceRecognitionRvizDisplay{
protected:
  ros::NodeHandle nh_;
  ros::Publisher face_rviz_circle_pub;
  ros::Publisher face_rviz_text_pub;
  boost::mutex mutex_;
  message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<cob_perception_msgs::ColorDepthImageArray, sensor_msgs::PointCloud2, cob_perception_msgs::DetectionArray> >* sync_input_3_;
  message_filters::Subscriber<cob_perception_msgs::ColorDepthImageArray> face_position_sub;
  message_filters::Subscriber<sensor_msgs::PointCloud2> pointcloud_sub;
  message_filters::Subscriber<cob_perception_msgs::DetectionArray> face_label_sub;

public :
  FaceRecognitionRvizDisplay()
  {
    //subscribers
    face_position_sub.subscribe(nh_, "/cob_people_detection/face_detector/face_positions",1);
    pointcloud_sub.subscribe(nh_,"/cob_people_detection/sensor_message_gateway/pointcloud_rgb_out",1);
    face_label_sub.subscribe(nh_,"/cob_people_detection/detection_tracker/face_position_array", 1); 
    sync_input_3_ = new message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<cob_perception_msgs::ColorDepthImageArray, sensor_msgs::PointCloud2, cob_perception_msgs::DetectionArray> >(10);
    sync_input_3_->connectInput(face_position_sub, pointcloud_sub, face_label_sub);
    sync_input_3_->registerCallback(boost::bind(&FaceRecognitionRvizDisplay::inputCallback, this, _1, _2, _3));

    //publish
    face_rviz_circle_pub = nh_.advertise<people_msgs::PositionMeasurementArray>("/cob_people_detection/people_detection_rviz_display/face_positions_for_rviz",1);
    face_rviz_text_pub = nh_.advertise<visualization_msgs::MarkerArray>("/cob_people_detection/people_detection_rviz_display/face_recognition_for_rviz", 1);
  
  }

  ~FaceRecognitionRvizDisplay()
  {
    if(sync_input_3_ != 0)
      delete sync_input_3_;
  }
  
  void inputCallback(const cob_perception_msgs::ColorDepthImageArray::ConstPtr& face_position_msgs, const sensor_msgs::PointCloud2::ConstPtr& points, const cob_perception_msgs::DetectionArray::ConstPtr& face_label_msgs)
  {
    pcl::PointCloud < pcl::PointXYZRGB > depth_cloud;                                                                      
    pcl::fromROSMsg(*points, depth_cloud);

    people_msgs::PositionMeasurementArray rviz_msgs;
    std::vector<people_msgs::PositionMeasurement> PeopleArray;
    people_msgs::PositionMeasurement people;
    
    visualization_msgs::MarkerArray marker_msg;
    visualization_msgs::Marker marker;
    std::vector<visualization_msgs::Marker> MarkerArray;
    marker.color.a = 1.0; 
    marker.color.r = 76.0/255;
    marker.color.g = 235.0/255;
    marker.color.b = 235.0/255;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;
    marker.ns = "cob_face_recognition";
    marker.header = points->header;    
    marker.scale.x=0.3;
    marker.scale.y=0.3;
    marker.scale.z=0.3;
    marker.lifetime=ros::Duration(1.0);
    rviz_msgs.header = points->header;



    
    for(int i=0; i< face_position_msgs->head_detections.size(); i++)
      {
	int x = face_position_msgs->head_detections[i].head_detection.x+face_position_msgs->head_detections[i].head_detection.width/2;
	int y = face_position_msgs->head_detections[i].head_detection.y+face_position_msgs->head_detections[i].head_detection.height/2;
	pcl::PointXYZRGB p;
	p = depth_cloud.points[depth_cloud.width * y + x] ;
	if ( !isnan (p.x) && ((p.x != 0.0) || (p.y != 0.0) || (p.z == 0.0)) )
	  {
	    people.pos.z=p.z;
	    people.pos.x=p.x;
	    people.pos.y=p.y;
	    
	    PeopleArray.push_back(people);
	    rviz_msgs.people=PeopleArray;
	  }

	  face_rviz_circle_pub.publish(rviz_msgs);
      }
    for(int i=0; i < face_label_msgs->detections.size(); i++)
      {
	int x = face_label_msgs->detections[i].mask.roi.x+face_label_msgs->detections[i].mask.roi.width/2;
	int y = face_label_msgs->detections[i].mask.roi.y+face_label_msgs->detections[i].mask.roi.height/2;
	pcl::PointXYZRGB p;
	
	p = depth_cloud.points[depth_cloud.width * y + x] ;
	if ( !isnan (p.x) && ((p.x != 0.0) || (p.y != 0.0) || (p.z == 0.0)) )
	  {

	    marker.pose.position.x = p.x;
	    marker.pose.position.y = p.y-0.5;
	    marker.pose.position.z = p.z;//above the circle mark
	    marker.text = face_label_msgs->detections[i].label.c_str();
	    marker.id=i;
	    MarkerArray.push_back(marker);
	    marker_msg.markers = MarkerArray;
	  }
	
	face_rviz_text_pub.publish(marker_msg);
      }
  }
};


int main(int argc, char** argv)
{
  ros::init(argc,argv,"msg_test");
  FaceRecognitionRvizDisplay face_recognition_rviz_display;
  ros::spin();
  return 0;
}
