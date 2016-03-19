//
// Created by kandithws on 7/1/2559.
//


#include <PeopleTracker.h>


//Public Function
PeopleTracker::PeopleTracker()
{
    this->last_available_id = 1;
    this->track_distance_threshold = 0.3;
    this->single_track_reset_distance = 1.0;
}

void PeopleTracker::setListUpdateConstraints(int getin, int getincheck, int getout)
{
    this->person_get_in_track_condition = getin;
    this->get_in_track_check_frame = getincheck;
    this->person_out_of_track_condition = getout;
}

void PeopleTracker::setTrackThreshold(float distTH)
{
    this->track_distance_threshold = distTH;
}

void PeopleTracker::setSingleTrackResetDistance(float distTH)
{
    this->single_track_reset_distance = distTH;
}

void PeopleTracker::resetTrackID(void)
{
    this->last_available_id = 1;
}



void PeopleTracker::trackPeople(std::vector<person> &global_track_list, std::vector<Eigen::Vector3f> new_center_list,
                                    int algorithm, int list_update_method)
{
    std::vector<int> lost_track_id;
    if(algorithm == SINGLE_NEAREST_NEIGHBOR_TRACKER)
    {
        bool track_status = this->track_usingSingleNN(global_track_list, new_center_list, this->track_distance_threshold);
        if(track_status)
        {
            lost_track_id.push_back(1);
        }
        //this->changeAllTrackTrue(global_track_list);

    }
    else if(algorithm == MULTI_NEAREST_NEIGHBOR_TRACKER)
    {
        this->track_usingMultiNN(global_track_list, new_center_list, lost_track_id ,this->track_distance_threshold);
        std::cout << "Track Multi NN ********" << std::endl;
    }
    else if(algorithm == KALMAN_TRACKER)
    {
        //TODO -- Implement Kalman Tracker
    }
    else
    {
        ROS_WARN("No Specified Algorithm: Abort");
        return;
    }

    if(list_update_method == UPDATE_WITH_FRAME_COUNT)
    {
        if(!lost_track_id.empty())
        {
            this->penaltyLostTrackPerson( global_track_list, lost_track_id);
            std::cout << "Penalty Lost Track ********" << std::endl;
        }
        this->checkTrackList(global_track_list);
        std::cout << "Check Track List********" << std::endl;
    }
    else if(list_update_method == UPDATE_NORMAL)
    {
        this->updateFrameWithNormalMethod(global_track_list, lost_track_id);
    }
    else
    {
        ROS_WARN("No Specified UPDATE METHOD: Abort");
    }

}

void PeopleTracker::addTrackerBall(pcl::visualization::PCLVisualizer::Ptr viewer_obj, std::vector<person> world_track_list)
{
    static std::vector<int> last_world_track_id;

    if(world_track_list.empty())
    {
        viewer_obj->removeAllShapes();
        last_world_track_id.clear();
        return;
    }

    //check lost found id & remove ball from UI
    for(int i =0 ; i<last_world_track_id.size(); i++)
        for(int j = 0; j< world_track_list.size(); j++)
        {
            if(last_world_track_id[i] == world_track_list[j].id)
                last_world_track_id.erase(last_world_track_id.begin() + i);
        }

    if(!last_world_track_id.empty())
    {
        for(int i =0 ; i<last_world_track_id.size(); i++)
        {
            std::string name = "sphere" + last_world_track_id[i];
            viewer_obj->removeShape(name.c_str());
        }
        std::cout << "-------------------------------" <<std::endl;
    }

    last_world_track_id.clear();

    for(int i=0; i< world_track_list.size();i++)
    {
        if(world_track_list[i].istrack == true)
        {
            Eigen::Vector3f out;
            out = world_track_list[i].points;
            std::string name = "sphere" + world_track_list[i].id;
            pcl::PointXYZRGBA pts;
            pts.x = out(0); pts.y = out(1); pts.z = out(2);
            viewer_obj->removeShape(name.c_str());
            viewer_obj->addSphere (pts, 0.1, world_track_list[i].color(0), world_track_list[i].color(1), world_track_list[i].color(2), name.c_str());
            last_world_track_id.push_back(world_track_list[i].id);
        }
    }

}



//Private Function---------------------------------------------------------

person PeopleTracker::createNewPerson(Eigen::Vector3f center_points, bool id_increment)
{
    person temp;
    temp.points = center_points;
    if(id_increment)
        temp.id = this->last_available_id++;
    else if(!id_increment)
    {
        //this->last_available_id = 1; //Refresh ID to have only 1 value
        temp.id = this->last_available_id;
    }

    temp.framesage = 0;
    temp.outcount = this->person_out_of_track_condition;
    temp.incount = this->person_get_in_track_condition;
    temp.istrack = false;
    temp.color = this->generateTrackerColor();
    return temp;
}




void PeopleTracker::findMinInNearestNeighborTable(std::vector<Eigen::Vector3f> row,std::vector<Eigen::Vector3f> col, float& min, std::vector<int>& index)
{
    //Brute Force Euclidean Distance Calculation between row and col, find minimum value and its index as an output
    if(row.empty())
        std::cout << "row is empty in NN table____" << std::endl;
    if(col.empty())
        std::cout << "col is empty in NN table____" << std::endl;

    Eigen::MatrixXf nn_matching_table(row.size(),col.size());
        min = this->compute_norm3(row[0], col[0]);
        for(int i=0; i< row.size() ; i++)
        {
            for(int j=0; j < col.size();j++ )
            {
                nn_matching_table(i,j) = this->compute_norm3(row[i], col[j]);
                if(nn_matching_table(i,j) < min )
                {
                    //update min finding nearest neighbour
                    index[0] = i;
                    index[1] = j;
                    min = nn_matching_table(i,j);
                }
            }
        }

        //Debugging: Print Out Matching Table
       /* std::cout <<"--Matiching Table--" << std::endl;
        for(int i =0 ; i < row.size() ; i++)
        {
            for(int j =0 ; j < col.size() ; j++)
            {
                std::cout << nn_matching_table(i,j) << '\t';
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;*/

}


void PeopleTracker::updateMatchedNearestNeighbor(float min, std::vector<int>& index,float distance_threshold, std::vector<person> &world,
                                                    std::vector<person>& world_temp, std::vector<Eigen::Vector3f> &pp_new_center_list)
{
    if((min < distance_threshold) && (!world_temp.empty())) //still track
    {
        //Match calculated minimum with going-to-publish list
        for(int k = 0; k < world.size() ; k++)
        {
            if(world[k].id == world_temp[index[0]].id )
            {
                world[k].points = pp_new_center_list[index[1]];
                world[k].id = world_temp[index[0]].id;
                //Refresh outcount condition
                world[k].outcount = this->person_out_of_track_condition;

                std::cout << "Updated Track id --> " <<  world[k].id << std::endl;
            }
        }
    }
    else //min above track distance threshold: This is new person
    {
        world.push_back(this->createNewPerson(pp_new_center_list[index[1]]));
    }

    //clear row and column inwhich contain min in NN Table.
    if(!world_temp.empty())
    {
        world_temp.erase(world_temp.begin() + index[0]);
    }
    pp_new_center_list.erase(pp_new_center_list.begin() + index[1]);
}




bool PeopleTracker::track_usingSingleNN(std::vector<person>& world, std::vector<Eigen::Vector3f>& pp_newcenter_list, float disTH)
{
    if(!world.empty())
    {
        Eigen::Vector3f last_point(world[0].points);
        int index;
        float min = 9999.0f;
        for(int i = 0 ; i < pp_newcenter_list.size();i++)
        {
            float tmp = this->compute_norm3(last_point, pp_newcenter_list[i]);
            if((tmp < min) && (tmp< disTH))
            {
                min = tmp;
                index = i;
            }
        }
        if (min < 9999.0f) //it is track
        {
            //world.clear();
            //world.push_back(this->createNewPerson(pp_newcenter_list[index], false));
            world[0].points = pp_newcenter_list[index];
            return true;
        }
        else
            return false;
    }
    else
    {
        //No one is tracked from the last frame; select minimum and add the closest one (Relative to Camera front) to the list
        Eigen::Vector3f origin;
        origin << 1.0, 0.0, 1.0;
        int index;
        float min = 9999.0f;
        for(int i = 0 ; i < pp_newcenter_list.size();i++)
        {
            float tmp = this->compute_norm3(origin, pp_newcenter_list[i]);
            if((tmp < min) && (tmp <= this->single_track_reset_distance))
            {
                min = tmp;
                index = i;
            }
        }
        if (min < 9999.0f) //it is track
        {
            world.push_back(this->createNewPerson(pp_newcenter_list[index], false));
            return true;
        }
        else
            return false;
    }
}



void PeopleTracker::track_usingMultiNN(std::vector<person>& world, std::vector<Eigen::Vector3f>& pp_newcenter_list, std::vector<int>& lost_track_id, float disTH)
{
    std::vector<person> world_temp(world);
    if(!world_temp.empty())
    {
        std::cout << "*******kuy 1" << std::endl;
        if(!pp_newcenter_list.empty())
        {
            int j=1;
            while( (!pp_newcenter_list.empty()) && (!world_temp.empty()) )
            {
                std::vector<Eigen::Vector3f> world_temp_points(world_temp.size());
                if(world_temp_points.empty())
                    break;
                if(pp_newcenter_list.empty())
                    break;
                float min;
                //int index[2];
                std::vector<int> index(2);
                for(int i=0;i<world_temp.size();i++)
                    world_temp_points[i] = world_temp[i].points;
                std::cout << "*******kuy 2 : iteration = " << j << std::endl;
                this->findMinInNearestNeighborTable(world_temp_points, pp_newcenter_list, min, index);
                std::cout << "*******kuy 3 : iteration = "<< j << std::endl;
                this->updateMatchedNearestNeighbor(min, index, disTH, world, world_temp, pp_newcenter_list);
                std::cout << "*******kuy 4 : iteration = "<< j++ << std::endl;
            }
        }
        else //No one Detected in this Frame, penalty all
        {
            std::cout << "*******kuy 5" << std::endl;
            for(int i=0; i < world_temp.size(); i++)
                    lost_track_id.push_back(world_temp[i].id);
            std::cout << "*******kuy 6" << std::endl;
        }

        //(Lost Track IDs)
        if(!world_temp.empty())
        {
            std::cout << "*******wtf 1" << std::endl;
            for(int i=0; i < world_temp.size(); i++)
                lost_track_id.push_back(world_temp[i].id);
            std::cout << "*******wtf 2" << std::endl;
        }
    }
    else
    {
        //No one is tracked from the last frame re_init tracker list
        std::cout << "*******ei 1" << std::endl;
        if(!pp_newcenter_list.empty())
        for(int i = 0 ; i < pp_newcenter_list.size();i++)
            world.push_back(this->createNewPerson(pp_newcenter_list[i]));
        std::cout << "*******ei 2" << std::endl;

    }
}

void PeopleTracker::updateFrameWithNormalMethod(std::vector<person>& world, std::vector<int> lost_found_id)
{
    //Forced Every single person to be tracked
    for(int i=0; i < world.size(); i++)
    {
        world[i].istrack = true;
    }

    //Delete all lost track frame
    if(!lost_found_id.empty())
    {
        for(int i=0; i < lost_found_id.size(); i++)
            for(int j=0; j< world.size(); j++)
            {
                if(world[j].id == lost_found_id[i])
                    world.erase(world.begin() + j);
            }
    }
}


void PeopleTracker::penaltyLostTrackPerson(std::vector<person>& world, std::vector<int> lost_found_id)
{
    //Decrease Frame lifetime/in_track_condition: if lost track in that frame
    if(!lost_found_id.empty())
    {
        for (int i=0; i < lost_found_id.size();i++)
        {
            for(int j=0 ; j < world.size();j++)
            {
                if(world[j].istrack == true)
                {
                    if(world[j].id == lost_found_id[i]) --world[j].outcount;
                }
                else if(world[j].istrack == false)
                {
                    if(world[j].id == lost_found_id[i]) --world[j].incount;
                }

            }
        }
    }
}

void PeopleTracker::checkTrackList(std::vector<person> &track_list)
{
    //Remove or Move in Person
    for(int i=0 ; i < track_list.size() ; i++ )
    {
        //Lost track -> Remove This Person
        if(track_list[i].outcount <= 0)
        {
            track_list.erase(track_list.begin() + i);
        }
        else
        {
            //New Person check Condition
            if(track_list[i].istrack == false)
            {

                if( (track_list[i].framesage >= this->get_in_track_check_frame))
                {
                    //Check this new entry person whether he/she is qualified to be tracked
                    if(track_list[i].outcount <= 0)
                    {
                        track_list.erase(track_list.begin() + i);
                    }
                    else
                    {
                        track_list[i].istrack = true;
                        track_list[i].outcount = this->person_out_of_track_condition;
                    }
                }
                else
                {
                    track_list[i].framesage++;
                }
            }
            else
            {
                track_list[i].framesage++;
            }
        }
    }
}




float PeopleTracker::compute_norm3(Eigen::Vector3f A, Eigen::Vector3f B)
{
    float delx2 = (A(0)-B(0))*(A(0)-B(0));
    float dely2 = (A(1)-B(1))*(A(1)-B(1));
    float delz2 = (A(2)-B(2))*(A(2)-B(2));
    return sqrt(delx2+dely2+delz2);
}


Eigen::Vector3f PeopleTracker::generateTrackerColor()
{
    Eigen::Vector3f color;
    float r = ((double) rand() / (RAND_MAX));
    float g = ((double) rand() / (RAND_MAX));
    float b = ((double) rand() / (RAND_MAX));
    color(0) = r;
    color(1) = g;
    color(2) = b;
    return color;
}
