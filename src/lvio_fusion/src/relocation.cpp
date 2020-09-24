#include "lvio_fusion/loop/relocation.h"
#include <DBoW3/QueryResults.h>

namespace lvio_fusion
{

Relocation::Relocation(std::string voc_path)
{
    thread_ = std::thread(std::bind(&Relocation::RelocationLoop, this));
    detector_ = cv::ORB::create();
    voc_ = DBoW3::Vocabulary(voc_path);
    db_ = DBoW3::Database(voc_, false, 0);
    head_ = 0;
}

void Relocation::RelocationLoop()
{
    while (true)
    {
        while (head_ < frames_.size())
        {
            AddKeyFrameIntoVoc(frames_[head_]);
            if(DetectLoop(frames_[head_]))
            {
                Associate(frames_[head_]);
            }
            head_++;
        }
        std::chrono::milliseconds dura(100);
        std::this_thread::sleep_for(dura);
    }
}

void Relocation::AddKeyFrameIntoVoc(Frame::Ptr frame)
{
    static int thershold = 20;
    // compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(frame->image_left, keypoints, 20, true);
    detector_->compute(frame->image_left, keypoints, frame->descriptors);
    db_.add(frame->descriptors);
}

bool Relocation::DetectLoop(Frame::Ptr frame)
{
    //first query; then add this frame into database!
    DBoW3::QueryResults ret;
    db_.query(frame->descriptors, ret, 4, frame->id - 20);
    // ret[0] is the nearest neighbour's score. threshold change with neighour score
    bool find_loop = false;
    cv::Mat loop_result;
    // a good match with its nerghbour
    if (ret.size() >= 1 && ret[0].Score > 0.05)
        for (unsigned int i = 1; i < ret.size(); i++)
        {
            if (ret[i].Score > 0.015)
            {
                find_loop = true;
            }
        }
    
    if (find_loop && frame->id > 20)
    {
        int min_index = -1;
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            if (min_index == -1 || (ret[i].Id < min_index && ret[i].Score > 0.015))
                min_index = ret[i].Id;
        }
        frame->loop = frames_[min_index];
        return true;
    }
    return false;
}

void Relocation::Associate(Frame::Ptr frame)
{
    // Frame::Ptr base_frame = frame->loop;
    
	// Eigen::Vector3d PnP_T_old;
	// Eigen::Matrix3d PnP_R_old;
	// Eigen::Vector3d relative_t;
	// Quaterniond relative_q;
	// double relative_yaw;
	// if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	// {
	// 	status.clear();
	//     PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
	// }

	// if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	// {
	//     relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
	//     relative_q = PnP_R_old.transpose() * origin_vio_R;
	//     relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
	//     //printf("PNP relative\n");
	//     //cout << "pnp relative_t " << relative_t.transpose() << endl;
	//     //cout << "pnp relative_yaw " << relative_yaw << endl;
	//     if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)
	//     {

	//     	has_loop = true;
	//     	loop_index = old_kf->index;
	//     	loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
	//     	             relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
	//     	             relative_yaw;
	//         return true;
	//     }
	// }
	// //printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(), t_match.toc());
	// return false;


}

// void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
//                          const std::vector<cv::Point3f> &matched_3d,
//                          std::vector<uchar> &status,
//                          Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
// {
// 	//for (int i = 0; i < matched_3d.size(); i++)
// 	//	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
// 	//printf("match size %d \n", matched_3d.size());
//     cv::Mat r, rvec, t, D, tmp_r;
//     cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
//     Matrix3d R_inital;
//     Vector3d P_inital;
//     Matrix3d R_w_c = origin_vio_R * qic;
//     Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

//     R_inital = R_w_c.inverse();
//     P_inital = -(R_inital * T_w_c);

//     cv::eigen2cv(R_inital, tmp_r);
//     cv::Rodrigues(tmp_r, rvec);
//     cv::eigen2cv(P_inital, t);

//     cv::Mat inliers;
//     TicToc t_pnp_ransac;

//     if (CV_MAJOR_VERSION < 3)
//         solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
//     else
//     {
//         if (CV_MINOR_VERSION < 2)
//             solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
//         else
//             solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);

//     }

//     for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
//         status.push_back(0);

//     for( int i = 0; i < inliers.rows; i++)
//     {
//         int n = inliers.at<int>(i);
//         status[n] = 1;
//     }

//     cv::Rodrigues(rvec, r);
//     Matrix3d R_pnp, R_w_c_old;
//     cv::cv2eigen(r, R_pnp);
//     R_w_c_old = R_pnp.transpose();
//     Vector3d T_pnp, T_w_c_old;
//     cv::cv2eigen(t, T_pnp);
//     T_w_c_old = R_w_c_old * (-T_pnp);

//     PnP_R_old = R_w_c_old * qic.transpose();
//     PnP_T_old = T_w_c_old - PnP_R_old * tic;

// }

// void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
// 								std::vector<cv::Point2f> &matched_2d_old_norm,
//                                 std::vector<uchar> &status,
//                                 const std::vector<BRIEF::bitset> &descriptors_old,
//                                 const std::vector<cv::KeyPoint> &keypoints_old,
//                                 const std::vector<cv::KeyPoint> &keypoints_old_norm)
// {
//     for(int i = 0; i < (int)window_brief_descriptors.size(); i++)
//     {
//         cv::Point2f pt(0.f, 0.f);
//         cv::Point2f pt_norm(0.f, 0.f);
//         if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
//           status.push_back(1);
//         else
//           status.push_back(0);
//         matched_2d_old.push_back(pt);
//         matched_2d_old_norm.push_back(pt_norm);
//     }

// }

// bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
//                             const std::vector<BRIEF::bitset> &descriptors_old,
//                             const std::vector<cv::KeyPoint> &keypoints_old,
//                             const std::vector<cv::KeyPoint> &keypoints_old_norm,
//                             cv::Point2f &best_match,
//                             cv::Point2f &best_match_norm)
// {
//     cv::Point2f best_pt;
//     int bestDist = 128;
//     int bestIndex = -1;
//     for(int i = 0; i < (int)descriptors_old.size(); i++)
//     {

//         int dis = HammingDis(window_descriptor, descriptors_old[i]);
//         if(dis < bestDist)
//         {
//             bestDist = dis;
//             bestIndex = i;
//         }
//     }
//     //printf("best dist %d", bestDist);
//     if (bestIndex != -1 && bestDist < 80)
//     {
//       best_match = keypoints_old[bestIndex].pt;
//       best_match_norm = keypoints_old_norm[bestIndex].pt;
//       return true;
//     }
//     else
//       return false;
// }

// int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
// {
//     BRIEF::bitset xor_of_bitset = a ^ b;
//     int dis = xor_of_bitset.count();
//     return dis;
// }

} // namespace lvio_fusion