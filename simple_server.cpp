#include <ecal/ecal.h>
#include <ecal/msg/protobuf/subscriber.h>
#include <ecal/msg/protobuf/publisher.h>
#include <google/protobuf/util/time_util.h>
#include <opencv2/core/mat.hpp>
#include <opencv/cv.hpp>
#include <math.h>

#include <mutex>
#include <condition_variable>
#include <eigen3/Eigen/Dense>
#include <functional> 
#include <iostream>
#include <thread>
#include <chrono>

#include "kimera-vio/pipeline/Pipeline.h"
#include "vio_input.pb.h"
#include "vio_output.pb.h"

#define NANOSECONDS_PER_SECOND 1000000000


VIO::FrameId kimera_current_frame_id = 0;
VIO::VioParams kimera_pipeline_params(std::string("../params/ILLIXR"));
VIO::Pipeline kimera_pipeline(kimera_pipeline_params);
eCAL::protobuf::CSubscriber<vio_input_proto::IMUCamVec> subscriber;
eCAL::protobuf::CPublisher<vio_output_proto::VIOOutput> publisher;

std::mutex mtx;
std::condition_variable cond_var; 
std::vector<vio_input_proto::IMUCamVec> input_buffer;
vio_input_proto::IMUCamData cam_buffer;


// define a subscriber callback function
void ReceiveVIOInput(const vio_input_proto::IMUCamVec& vio_input) {
	input_buffer.push_back(vio_input);

	if (mtx.try_lock() == false) {
		std::cout << "CONSUMER BUSY" << std::endl;
		return;
	}

	auto curr_vio_input = input_buffer.front();
	std::cout << "CONSUMING INPUT: " << curr_vio_input.num() << std::endl;

	for (int i = 0; i < vio_input.imu_cam_data_size(); i++) {
		vio_input_proto::IMUCamData curr_data = curr_vio_input.imu_cam_data(i);
		cam_buffer = curr_data;

		VIO::Vector6 imu_raw_vals;
		imu_raw_vals << curr_data.linear_accel().x(), curr_data.linear_accel().y(), curr_data.linear_accel().z(), curr_data.angular_vel().x(), curr_data.angular_vel().y(), curr_data.angular_vel().z();

		// std::cout << curr_data.timestamp() << ", " << curr_data.linear_accel().x() << ", " << curr_data.linear_accel().y() << ", " << curr_data.linear_accel().z() << ", " << curr_data.angular_vel().x() << ", " << curr_data.angular_vel().y() << ", " << curr_data.angular_vel().z() << std::endl;
		
		kimera_pipeline.fillSingleImuQueue(VIO::ImuMeasurement(
			curr_data.timestamp(), 
			imu_raw_vals
		));

		// std::cout << "Received IMU data: " << count << std::endl;
		if (curr_data.rows() != -1 && curr_data.cols() != -1) {
			// std::vector<uint8_t> img0_vec(curr_data.img0_data().begin(),curr_data.img0_data().end());
			// std::vector<uint8_t> img1_vec(curr_data.img1_data().begin(),curr_data.img1_data().end());

			unsigned char* img0_data = (unsigned char*) curr_data.img0_data().c_str();
			unsigned char* img1_data = (unsigned char*) curr_data.img1_data().c_str();

			cv::Mat img0(curr_data.rows(), curr_data.cols(), 0, img0_data);
			cv::Mat img1(curr_data.rows(), curr_data.cols(), 0, img1_data);

			VIO::CameraParams left_cam_info = kimera_pipeline_params.camera_params_.at(0);
			VIO::CameraParams right_cam_info = kimera_pipeline_params.camera_params_.at(1);

			kimera_pipeline.fillLeftFrameQueue(VIO::make_unique<VIO::Frame>
				(kimera_current_frame_id, 
				curr_data.timestamp(), 
				left_cam_info, 
				img0)
			);
			kimera_pipeline.fillRightFrameQueue(VIO::make_unique<VIO::Frame>
				(kimera_current_frame_id, 
				curr_data.timestamp(), 
				right_cam_info, 
				img1)
			);
		}
	}

	input_buffer.erase(input_buffer.begin());
	mtx.unlock();

	auto f = []() {
		mtx.lock();

		std::cout << "RUNNING CAM" << std::endl;
		kimera_pipeline.spin();

		mtx.unlock();
	};

	std::thread kimeara_thread(f);
	kimeara_thread.detach();
}


void pose_callback(const std::shared_ptr<VIO::BackendOutput>& vio_output) {
	const auto& cached_state = vio_output->W_State_Blkf_;
	const auto& w_pose_blkf_trans = cached_state.pose_.translation().transpose();
	const auto& w_pose_blkf_rot = cached_state.pose_.rotation().quaternion();
	const auto& w_vel_blkf = cached_state.velocity_.transpose();
	const auto& imu_bias_gyro = cached_state.imu_bias_.gyroscope().transpose();
	const auto& imu_bias_acc = cached_state.imu_bias_.accelerometer().transpose();

	// std::cout << "x: " << w_pose_blkf_trans[0] << " y: " << w_pose_blkf_trans[1] << " z: " << w_pose_blkf_trans[2] << std::endl;

	// Construct slow pose for output
	vio_output_proto::SlowPose* slow_pose = new vio_output_proto::SlowPose();
	slow_pose->set_timestamp(cam_buffer.timestamp());

	vio_output_proto::Vec3* position = new vio_output_proto::Vec3();
	position->set_x(w_pose_blkf_trans.x());
	position->set_y(w_pose_blkf_trans.y());
	position->set_z(w_pose_blkf_trans.z());
	slow_pose->set_allocated_position(position);

	vio_output_proto::Quat* rotation = new vio_output_proto::Quat();
	rotation->set_w(w_pose_blkf_rot[0]);
	rotation->set_x(w_pose_blkf_rot[1]);
	rotation->set_y(w_pose_blkf_rot[2]);
	rotation->set_z(w_pose_blkf_rot[3]);
	slow_pose->set_allocated_rotation(rotation);

	// Construct IMU integrator input for output
	vio_output_proto::IMUIntInput* imu_int_input = new vio_output_proto::IMUIntInput();
	imu_int_input->set_t_offset(0.05);
	imu_int_input->set_last_cam_integration_time(cam_buffer.timestamp() / NANOSECONDS_PER_SECOND);

	vio_output_proto::IMUParams* imu_params = new vio_output_proto::IMUParams();
	imu_params->set_gyro_noise(kimera_pipeline_params.imu_params_.gyro_noise_);
    imu_params->set_acc_noise(kimera_pipeline_params.imu_params_.acc_noise_);
    imu_params->set_gyro_walk(kimera_pipeline_params.imu_params_.gyro_walk_);
    imu_params->set_acc_walk(kimera_pipeline_params.imu_params_.acc_walk_);
	vio_output_proto::Vec3* n_gravity = new vio_output_proto::Vec3();
	n_gravity->set_x(kimera_pipeline_params.imu_params_.n_gravity_.x());
	n_gravity->set_y(kimera_pipeline_params.imu_params_.n_gravity_.y());
	n_gravity->set_z(kimera_pipeline_params.imu_params_.n_gravity_.z());
	imu_params->set_allocated_n_gravity(n_gravity);
    imu_params->set_imu_integration_sigma(kimera_pipeline_params.imu_params_.imu_integration_sigma_);
    imu_params->set_nominal_rate(kimera_pipeline_params.imu_params_.nominal_rate_);
	imu_int_input->set_allocated_imu_params(imu_params);

	vio_output_proto::Vec3* biasAcc = new vio_output_proto::Vec3();
	biasAcc->set_x(imu_bias_acc.x());
	biasAcc->set_y(imu_bias_acc.y());
	biasAcc->set_z(imu_bias_acc.z());
	imu_int_input->set_allocated_biasacc(biasAcc);

	vio_output_proto::Vec3* biasGyro = new vio_output_proto::Vec3();
	biasGyro->set_x(imu_bias_gyro.x());
	biasGyro->set_y(imu_bias_gyro.y());
	biasGyro->set_z(imu_bias_gyro.z());
	imu_int_input->set_allocated_biasgyro(biasGyro);

	vio_output_proto::Vec3* position_int = new vio_output_proto::Vec3();
	position_int->set_x(w_pose_blkf_trans.x());
	position_int->set_y(w_pose_blkf_trans.y());
	position_int->set_z(w_pose_blkf_trans.z());
	imu_int_input->set_allocated_position(position_int);

	vio_output_proto::Quat* rotation_int = new vio_output_proto::Quat();
	rotation_int->set_w(w_pose_blkf_rot[0]);
	rotation_int->set_x(w_pose_blkf_rot[1]);
	rotation_int->set_y(w_pose_blkf_rot[2]);
	rotation_int->set_z(w_pose_blkf_rot[3]);
	imu_int_input->set_allocated_rotation(rotation_int);

	vio_output_proto::Vec3* velocity = new vio_output_proto::Vec3();
	velocity->set_x(w_vel_blkf.x());
	velocity->set_y(w_vel_blkf.y());
	velocity->set_z(w_vel_blkf.z());
	imu_int_input->set_allocated_velocity(velocity);

	vio_output_proto::VIOOutput* vio_output_params = new vio_output_proto::VIOOutput();
	vio_output_params->set_allocated_slow_pose(slow_pose);
	vio_output_params->set_allocated_imu_int_input(imu_int_input);

	publisher.Send(*vio_output_params);
	delete vio_output_params;
}

int main(int argc, char **argv)
{
	// Verified that params are loaded correctly
	// std::cout << kimera_pipeline_params.imu_params_.gyro_noise_ << " " <<  kimera_pipeline_params.imu_params_.gyro_walk_ << std::endl;

	kimera_pipeline.registerBackendOutputCallback(
		std::bind(
			&pose_callback,
			std::placeholders::_1
		)
	);

	// create a subscriber (topic name "shape")
	eCAL::Initialize(0, NULL, "VIO Offloading Sensor Data Reader");

	subscriber = eCAL::protobuf::CSubscriber
		<vio_input_proto::IMUCamVec>("vio_input");
	subscriber.AddReceiveCallback(
		std::bind(&ReceiveVIOInput, std::placeholders::_2));

	publisher = eCAL::protobuf::CPublisher
		<vio_output_proto::VIOOutput>("vio_output");

	while (eCAL::Ok()) {
	}

	eCAL::Finalize();
}
