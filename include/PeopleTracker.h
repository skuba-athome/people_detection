//
// Created by kandithws on 7/1/2559.
//

#ifndef PEOPLE_DETECTION_PEOPLE_TRACKER_H
#define PEOPLE_DETECTION_PEOPLE_TRACKER_H

#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>


#define DEFAULT_OUT_OF_TRACK_CONDITION 12
#define DEFAULT_GET_IN_TRACK_CONDITION 3
#define DEFAULT_GET_IN_TRACK_CHECK_FRAME 5

#define SINGLE_NEAREST_NEIGHBOR_TRACKER 0
#define MULTI_NEAREST_NEIGHBOR_TRACKER 1
#define KALMAN_TRACKER 2

#define UPDATE_NORMAL 0
#define UPDATE_WITH_FRAME_COUNT 1

typedef struct{
    Eigen::Vector3f points;
    Eigen::Vector3f color;
    int id;
    int framesage;
    int incount;
    int outcount;
    bool istrack;
}person;

class PeopleTracker
{
    public:
        PeopleTracker();
        //Use this before track: if wish to use list_update_method = UPDATE_WITH_FRAME_COUNT
        void setListUpdateConstraints(int getin, int getincheck, int getout);
        void setTrackThreshold(float distTH);
        void trackPeople(std::vector<person> &global_track_list, std::vector<Eigen::Vector3f> new_center_list,
                                    int algorithm = SINGLE_NEAREST_NEIGHBOR_TRACKER, int list_update_method = UPDATE_NORMAL);
        void addTrackerBall(pcl::visualization::PCLVisualizer::Ptr viewer_obj, std::vector<person> world_track_list);

    private:
        bool track_usingSingleNN(std::vector<person>& world, std::vector<Eigen::Vector3f>& pp_newcenter_list, float disTH = 0.35);
        void track_usingMultiNN(std::vector<person> &world, std::vector<Eigen::Vector3f> &pp_newcenter_list, std::vector<int>& lost_track_id, float disTH); //return lost tracked ids
        void findMinInNearestNeighborTable(std::vector<Eigen::Vector3f> row, std::vector<Eigen::Vector3f> col, float& min, std::vector<int>& index);
        void updateMatchedNearestNeighbor(float min, std::vector<int>& index,float distance_threshold, std::vector<person> &world,
                                            std::vector<person>& world_temp, std::vector<Eigen::Vector3f> &pp_new_center_list);

        void changeAllTrackTrue(std::vector<person>& world);
        void penaltyLostTrackPerson(std::vector<person>& world, std::vector<int> lost_found_id);
        void checkTrackList(std::vector<person> &track_list);
        person createNewPerson(Eigen::Vector3f center_points,  bool id_increment = true);
        Eigen::Vector3f generateTrackerColor();
        float compute_norm3(Eigen::Vector3f A, Eigen::Vector3f B);

        int last_available_id;
        std::vector<person> world_track_list;
        int person_out_of_track_condition;
        int person_get_in_track_condition;
        int get_in_track_check_frame;
        float track_distance_threshold;



};


#endif //PEOPLE_DETECTION_PEOPLE_TRACKER_H
