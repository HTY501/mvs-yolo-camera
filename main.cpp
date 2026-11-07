#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "MvCameraControl.h"

struct Detection {
    cv::Rect box;
    float conf;
    int class_id;
    std::vector<cv::Point2f> keypoints; 
};

int main() {
    // 1. 初始化相机
    if (MV_CC_Initialize() != MV_OK) return -1;
    MV_CC_DEVICE_INFO_LIST deviceList;
    if (MV_CC_EnumDevices(MV_USB_DEVICE, &deviceList) != MV_OK || deviceList.nDeviceNum == 0) return -1;

    void* handle = nullptr; 
    if (MV_CC_CreateHandle(&handle, deviceList.pDeviceInfo[0]) != MV_OK) return -1;
    if (MV_CC_OpenDevice(handle) != MV_OK) return -1;

    // 【调亮画面】：把曝光调大，关掉自动
    MV_CC_SetEnumValueByString(handle, "ExposureAuto", "Off");
    MV_CC_SetFloatValue(handle, "ExposureTime", 5000.0f); // 改大一点
    MV_CC_SetEnumValueByString(handle, "GainAuto", "Off");
    MV_CC_SetFloatValue(handle, "Gain", 6.0f); // 稍微加点增益

    if (MV_CC_StartGrabbing(handle) != MV_OK) return -1;

    // 2. 初始化 OpenVINO
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model("best.xml"); 
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    MV_FRAME_OUT frame = {0};
    const int INPUT_SIZE = 640;
    const float CONF_THRESHOLD = 0.5f;
    const float NMS_THRESHOLD = 0.45f;

    while (true) {
	auto start = std::chrono::high_resolution_clock::now();
        int ret = MV_CC_GetImageBuffer(handle, &frame, 1000);
        
        if (ret == MV_OK) {
            // 提取原图并转为 BGR
            cv::Mat bayer_img(frame.stFrameInfo.nHeight, frame.stFrameInfo.nWidth, CV_8UC1, frame.pBufAddr);
            cv::Mat bgr_img;
            
            // 【修复颜色】：如果红蓝反了，换成 cv::COLOR_BayerBG2BGR
	                // 1. 先转换成 BGR
            cv::cvtColor(bayer_img, bgr_img, cv::COLOR_BayerRG2BGR);

            // 2. 【核心修复】：分离通道，互换 R 和 B，再合并
            std::vector<cv::Mat> channels;
            cv::split(bgr_img, channels);
            std::swap(channels[0], channels[2]); // channels[0]是B，channels[2]是R
            cv::merge(channels, bgr_img);

            int orig_w = bgr_img.cols;
            int orig_h = bgr_img.rows;
	    // // 预处理
            cv::Mat resized_img;
            cv::resize(bgr_img, resized_img, cv::Size(INPUT_SIZE, INPUT_SIZE));
            cv::Mat blob;
            cv::dnn::blobFromImage(resized_img, blob, 1.0 / 255.0, cv::Size(INPUT_SIZE, INPUT_SIZE), cv::Scalar(), true, false);
            
            ov::Tensor input_tensor = infer_request.get_input_tensor();
            std::memcpy(input_tensor.data<float>(), blob.ptr<float>(), blob.total() * sizeof(float));
            infer_request.infer();

            // --- 后处理 ---
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
                    // 1, 18, 8400 格式中，类别的起始索引是 4
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

                    // 映射回原图
                    x = x * orig_w / INPUT_SIZE;
                    y = y * orig_h / INPUT_SIZE;
                    w = w * orig_w / INPUT_SIZE;
                    h = h * orig_h / INPUT_SIZE;

                    boxes.push_back(cv::Rect(x, y, w, h));
                    confidences.push_back(max_class_score);
                    class_ids.push_back(class_id);

                    // 【关键修复】：提取 4 个关键点 (每个点只占 x, y 两个位置，没有置信度)
                    std::vector<cv::Point2f> kpts;
                    for (int k = 0; k < num_kpts; ++k) {
                        // 关键点起始索引是 4(bbox) + 6(cls) = 10
                        // 每个关键点占 2 个位置
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

            // 画图
            for (int idx : indices) {
                cv::Rect box = boxes[idx];
                int class_id = class_ids[idx];
                float conf = confidences[idx];

                cv::rectangle(bgr_img, box, cv::Scalar(0, 255, 0), 2);
                
                char text[256];
                sprintf(text, "Class %d: %.2f", class_id, conf);
                cv::putText(bgr_img, text, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

                // 画红色的关键点
                for (const auto& kpt : all_kpts[idx]) {
                    cv::circle(bgr_img, kpt, 5, cv::Scalar(0, 0, 255), -1); 
                }
            }

            cv::imshow("MVS YOLO", bgr_img);
            MV_CC_FreeImageBuffer(handle, &frame);

            if (cv::waitKey(1) == 'q') break;
        } else {
            // 如果是插拔导致的断开，打印一次即可，别刷屏
            static bool err_printed = false;
            if (!err_printed) {
                std::cerr << "相机断开或获取超时，错误码: " << std::hex << ret << std::endl;
                err_printed = true;
            }
        }
    }

    MV_CC_StopGrabbing(handle);
    MV_CC_CloseDevice(handle);
    MV_CC_DestroyHandle(handle);
    MV_CC_Finalize();
    return 0;
}
