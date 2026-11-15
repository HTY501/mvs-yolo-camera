#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "MvCameraControl.h"

// 用于保存检测结果的结构体
struct Detection {
    cv::Rect box;
    float conf;
    int class_id;
    std::vector<cv::Point2f> keypoints; 
};

int main() {
    // ==========================================
    // 1. 初始化相机 (MVS SDK)
    // ==========================================
    if (MV_CC_Initialize() != MV_OK) return -1;
    MV_CC_DEVICE_INFO_LIST deviceList;
    if (MV_CC_EnumDevices(MV_USB_DEVICE, &deviceList) != MV_OK || deviceList.nDeviceNum == 0) return -1;

    void* handle = nullptr; 
    if (MV_CC_CreateHandle(&handle, deviceList.pDeviceInfo[0]) != MV_OK) return -1;
    if (MV_CC_OpenDevice(handle) != MV_OK) {
        std::cerr << "打开相机失败，可能被占用！" << std::endl;
        return -1;
    }

    MV_CC_SetEnumValueByString(handle, "ExposureAuto", "Off");
    MV_CC_SetFloatValue(handle, "ExposureTime", 5000.0f); 
    MV_CC_SetEnumValueByString(handle, "GainAuto", "Off");
    MV_CC_SetFloatValue(handle, "Gain", 6.0f);

    if (MV_CC_StartGrabbing(handle) != MV_OK) return -1;

    // ==========================================
    // 2. 初始化 OpenVINO 与 读取相机内参
    // ==========================================
    std::cout << "正在加载 OpenVINO 模型..." << std::endl;
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model("best.xml"); 
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // 读取相机内参
    cv::Mat camera_matrix, dist_coeffs;
    cv::FileStorage fs("camera_params.yml", cv::FileStorage::READ);
    if (fs.isOpened()) {
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> dist_coeffs;
        std::cout << "相机内参读取成功！" << std::endl;
    } else {
        std::cerr << "⚠️ 警告：未能读取 camera_params.yml，PnP将无法运行！" << std::endl;
    }

    MV_FRAME_OUT frame = {0};
    const int INPUT_SIZE = 640;
    const float CONF_THRESHOLD = 0.5f;
    const float NMS_THRESHOLD = 0.45f;

    // 装甲板物理尺寸 (小装甲板：宽125mm，高55mm，单位:mm)
    float HALF_W = 62.5f; 
    float HALF_H = 27.5f; 
    std::vector<cv::Point3f> object_points = {
        {-HALF_W, -HALF_H, 0.0f}, // 左上
        { HALF_W, -HALF_H, 0.0f}, // 右上
        { HALF_W,  HALF_H, 0.0f}, // 右下
        {-HALF_W,  HALF_H, 0.0f}  // 左下
    };

    std::cout << "开始推理，按 'q' 退出..." << std::endl;
    // ==========================================
    // 初始化卡尔曼滤波器 (6维状态: x, y, z, vx, vy, vz; 3维测量: x, y, z)
    // ==========================================
    cv::KalmanFilter kf(6, 3, 0);
    cv::Mat state(6, 1, CV_32F);  // [x, y, z, vx, vy, vz]
    cv::Mat meas(3, 1, CV_32F);   // [x, y, z]

    // 状态转移矩阵 F (假设两帧之间的时间间隔 dt = 1)
    kf.transitionMatrix = (cv::Mat_<float>(6, 6) << 
        1, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 1,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);

    // 测量矩阵 H (只测量位置 x, y, z)
    cv::setIdentity(kf.measurementMatrix);

    // 过程噪声协方差 Q (稍微调大，适应运动)
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));
    // 测量噪声协方差 R (越小越相信 PnP 的原始数据)
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2));
    // 后验误差协方差 P
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
    // ==========================================
    // 3. 主循环
    // ==========================================
    while (true) {
        int ret = MV_CC_GetImageBuffer(handle, &frame, 1000);
        
        if (ret == MV_OK) {
            cv::Mat bayer_img(frame.stFrameInfo.nHeight, frame.stFrameInfo.nWidth, CV_8UC1, frame.pBufAddr);
            cv::Mat bgr_img;
            
            // 颜色转换
            cv::cvtColor(bayer_img, bgr_img, cv::COLOR_BayerRG2BGR);
            std::vector<cv::Mat> channels;
            cv::split(bgr_img, channels);
            std::swap(channels[0], channels[2]); 
            cv::merge(channels, bgr_img);

            int orig_w = bgr_img.cols;
            int orig_h = bgr_img.rows;

            // 预处理
            cv::Mat resized_img;
            cv::resize(bgr_img, resized_img, cv::Size(INPUT_SIZE, INPUT_SIZE));
            cv::Mat blob;
            cv::dnn::blobFromImage(resized_img, blob, 1.0 / 255.0, cv::Size(INPUT_SIZE, INPUT_SIZE), cv::Scalar(), true, false);
            
            ov::Tensor input_tensor = infer_request.get_input_tensor();
            std::memcpy(input_tensor.data<float>(), blob.ptr<float>(), blob.total() * sizeof(float));
            infer_request.infer();

            // --- 后处理解析 ---
            auto output_tensor = infer_request.get_output_tensor(0);
            float* output_data = output_tensor.data<float>();
            
            int num_anchors = 8400;
            int num_classes = 6; 
            int num_kpts = 4;    

            std::vector<cv::Rect> boxes;
            std::vector<float> confidences;
            std::vector<int> class_ids;
            std::vector<std::vector<cv::Point2f>> all_kpts;

            for (int i = 0; i < num_anchors; ++i) {
                float max_class_score = 0;
                int class_id = -1;
                for (int c = 0; c < num_classes; ++c) {
                    float score = output_data[(4 + c) * num_anchors + i];
                    if (score > max_class_score) {
                        max_class_score = score;
                        class_id = c;
                    }
                }

                if (max_class_score > CONF_THRESHOLD) {
                    float cx = output_data[0 * num_anchors + i];
                    float cy = output_data[1 * num_anchors + i];
                    float w  = output_data[2 * num_anchors + i];
                    float h  = output_data[3 * num_anchors + i];

                    float x = cx - w / 2.0;
                    float y = cy - h / 2.0;

                    x = x * orig_w / INPUT_SIZE;
                    y = y * orig_h / INPUT_SIZE;
                    w = w * orig_w / INPUT_SIZE;
                    h = h * orig_h / INPUT_SIZE;

                    boxes.push_back(cv::Rect(x, y, w, h));
                    confidences.push_back(max_class_score);
                    class_ids.push_back(class_id);

                    std::vector<cv::Point2f> kpts;
                    for (int k = 0; k < num_kpts; ++k) {
                        float kx = output_data[(10 + k * 2) * num_anchors + i];
                        float ky = output_data[(10 + k * 2 + 1) * num_anchors + i];
                        kx = kx * orig_w / INPUT_SIZE;
                        ky = ky * orig_h / INPUT_SIZE;
                        kpts.push_back(cv::Point2f(kx, ky));
                    }
                    all_kpts.push_back(kpts);
                }
            }

            // NMS
            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);

            // --- 遍历结果画图并解算 PnP ---
            for (int idx : indices) {
                cv::Rect box = boxes[idx];
                int class_id = class_ids[idx];
                float conf = confidences[idx];

                // 1. 画框和文字
                cv::rectangle(bgr_img, box, cv::Scalar(0, 255, 0), 2);
                char text[256];
                sprintf(text, "Class %d: %.2f", class_id, conf);
                cv::putText(bgr_img, text, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

                // 2. 画关键点
                for (const auto& kpt : all_kpts[idx]) {
                    cv::circle(bgr_img, kpt, 4, cv::Scalar(0, 0, 255), -1); 
                }

                // 3. 【核心】PnP 位姿解算
                std::vector<cv::Point2f> image_points = all_kpts[idx];
                if (image_points.size() == 4 && !camera_matrix.empty()) {
                    cv::Mat rvec, tvec;
                    // 使用 SOLVEPNP_IPPE 对平面四点解算更稳定
                    bool pnp_success = cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE);
                    
                    if (pnp_success) {
                        // 1. 获取原始测量值
                        float x = tvec.at<double>(0);
                        float y = tvec.at<double>(1);
                        float z = tvec.at<double>(2);
                        float distance = std::sqrt(x*x + y*y + z*z);

                        // 2. 填入测量矩阵
                        meas.at<float>(0) = x;
                        meas.at<float>(1) = y;
                        meas.at<float>(2) = z;

                        // 3. 卡尔曼预测（这一步可以获取下一帧的预测位置）
                        cv::Mat prediction = kf.predict();

                        // 4. 卡尔曼更新（结合测量值纠正预测）
                        cv::Mat estimated = kf.correct(meas);

                        // 获取滤波后的数据和速度
                        float est_x = estimated.at<float>(0);
                        float est_y = estimated.at<float>(1);
                        float est_z = estimated.at<float>(2);
                        
                        float vx = estimated.at<float>(3);
                        float vy = estimated.at<float>(4);
                        float vz = estimated.at<float>(5);

                        // 5. 预测未来 (假设预测未来 5 帧后的位置)
                        float pred_x = est_x + vx * 5;
                        float pred_y = est_y + vy * 5;
                        float pred_z = est_z + vz * 5;

                        // --- 屏幕显示区 ---
                        // 1. 画原始测量数据(黄色)
                        char pnp_text[256];
                        sprintf(pnp_text, "Raw Dist: %.0f mm", distance);
                        cv::putText(bgr_img, pnp_text, cv::Point(box.x, box.y + box.height + 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

                        // 2. 画滤波后数据(绿色)
                        sprintf(pnp_text, "Kalman Dist: %.0f mm", std::sqrt(est_x*est_x + est_y*est_y + est_z*est_z));
                        cv::putText(bgr_img, pnp_text, cv::Point(box.x, box.y + box.height + 45), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

                        // 3. 画预测数据(红色)
                        sprintf(pnp_text, "Pred Dist: %.0f mm", std::sqrt(pred_x*pred_x + pred_y*pred_y + pred_z*pred_z));
                        cv::putText(bgr_img, pnp_text, cv::Point(box.x, box.y + box.height + 70), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
                    }                }
            }

            cv::imshow("MVS YOLO + PnP", bgr_img);
            MV_CC_FreeImageBuffer(handle, &frame);

            if (cv::waitKey(1) == 'q') break;
        } else {
            std::cerr << "获取图像超时或失败，错误码: " << std::hex << ret << std::endl;
        }
    }

    // 退出清理
    MV_CC_StopGrabbing(handle);
    MV_CC_CloseDevice(handle);
    MV_CC_DestroyHandle(handle);
    MV_CC_Finalize();
    cv::destroyAllWindows();
    return 0;
}
