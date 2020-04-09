#pragma once

#include "ObjectListener.h"

#include <utility>
#include "PlottingListener.h"

using namespace affdex::vision;

class PlottingObjectListener : public ObjectListener, public PlottingListener<Object> {

public:

    PlottingObjectListener(std::ofstream& csv, bool draw_display, bool enable_logging, bool draw_object_id,
        std::map<Feature, Duration> callback_intervals, std::vector<CabinRegion> cabin_regions) :
        PlottingListener(csv, draw_display, enable_logging), callback_intervals_(std::move(callback_intervals)),
        cabin_regions_(std::move(cabin_regions)), draw_object_id_(draw_object_id), frames_with_objects_(0) {
        out_stream_ << "TimeStamp, objectId, confidence, upperLeftX, upperLeftY, lowerRightX, lowerRightY, ObjectType";

        for (const auto& cr :cabin_regions_) {
            out_stream_ << "," << "Region " << cr.id;
        }

        out_stream_ << std::endl;
        out_stream_.precision(2);
        out_stream_ << std::fixed;

        // set the timeout to the max callback interval
        for (const auto& pair : callback_intervals) {
            if (pair.second > timeout_) {
                timeout_ = pair.second;
            }
        }
    }

    std::map<Feature, Duration> getCallbackIntervals() const override {
        return callback_intervals_;
    }

    void onObjectResults(const std::map<ObjectId, Object>& objects, vision::Frame frame) override {
        std::lock_guard<std::mutex> lg(mtx);
        results_.emplace_back(frame, objects);
        process_last_ts_ = frame.getTimestamp();

        processed_frames_++;
        if (!objects.empty()) {
            frames_with_objects_++;
        }
    };

    static std::string typeToString(const Object::Type& type) {
        switch (type) {
            case Object::Type::UNKNOWN: {
                return "UNKNOWN";
            }
            case Object::Type::PHONE: {
                return "PHONE";
            }
            case Object::Type::CHILD_SEAT: {
                return "CHILD_SEAT";
            }
            default: {
                throw std::runtime_error{
                    std::string("Object::typeToString encountered unrecognized Type: ")
                        + std::to_string(static_cast<int>(type))};
            }
        }
    }

    void outputToFile(const std::map<ObjectId, Object>& objects, double time_stamp) override {
        if (objects.empty()) {
            // TimeStamp objectId confidence upperLeftX upperLeftY lowerRightX lowerRightY ObjectType"
            out_stream_ << time_stamp << ",nan,nan,nan,nan,nan,nan,nan";
            for (const auto& cr :cabin_regions_) {
                out_stream_ << ",nan";
            }
            out_stream_ << std::endl;
        }

        for (const auto& id_obj_pair : objects) {
            Object obj = id_obj_pair.second;
            std::vector<Point> bbox({obj.boundingBox.getTopLeft(), obj.boundingBox.getBottomRight()});

            out_stream_ << time_stamp << ","
                        << id_obj_pair.first << ","
                        << obj.confidence << ","
                        << std::setprecision(0) << bbox[0].x << "," << bbox[0].y << "," << bbox[1].x << "," << bbox[1].y
                        << "," << std::setprecision(4)
                        << typeToString(obj.type);

            for (const auto& cr :cabin_regions_) {
                bool found = false;
                for (const auto& o : obj.matchedRegions) {
                    if (cr.id == o.cabinRegion.id) {
                        out_stream_ << "," << o.matchConfidence;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    out_stream_ << "," << 0;
                }
            }
            out_stream_ << std::endl;
        }
    }

    void draw(const std::map<ObjectId, Object>& objects, const vision::Frame& image) override {
        std::shared_ptr<unsigned char> img_data = image.getBGRByteArray();
        const cv::Mat img = cv::Mat(image.getHeight(), image.getWidth(), CV_8UC3, img_data.get());
        viz_.updateImage(img);

        for (const auto& id_object_pair : objects) {
            const auto obj = id_object_pair.second;
            viz_.drawObjectMetrics(obj);
            //add object region detected
            for(const auto& o : obj.matchedRegions){
                const auto id = o.cabinRegion.id;
                if(std::find(object_regions_.begin(), object_regions_.end(), id) == object_regions_.end()) {
                    object_regions_.emplace_back(id);
                }
            }

            //add object type detected
            if(std::find(object_types_.begin(), object_types_.end(), obj.type) == object_types_.end()) {
                object_types_.emplace_back(obj.type);
            }
        }

        viz_.showImage();
        image_data_ = viz_.getImageData();
    }

    using PlottingListener::processResults;  // make the overload taking a Frame arg visible
    void processResults() override {
        while (getDataSize() > 0) {
            latest_data_ = getData();
            drawRecentFrame();
            vision::Frame old_frame = latest_data_.first;
            const auto objects = latest_data_.second;
            outputToFile(objects, old_frame.getTimestamp());
        }
    }

    int getSamplesWithObjectsPercent() const {
        return (static_cast<float>(frames_with_objects_) / processed_frames_) * 100;
    }

    std::string getObjectTypesDetected() const {
        std::string obj_types;
        for(int i = 0; i<object_types_.size(); ++i){
            if(i>0){
                obj_types += ", ";
            }
            obj_types += typeToString(object_types_[i]);
        }
        return obj_types;
    }

    std::string getObjectRegionsDetected() const {
        std::string object_regions;
        for(int i = 0; i<object_regions_.size(); ++i){
            if(i>0){
                object_regions += ", ";
            }
            object_regions += std::to_string(object_regions_[i]);
        }
        return object_regions;
    }

    std::string getCallBackInterval() {
        if(object_types_.empty()) {
            return {};
        }
        else {
            return std::to_string(callback_intervals_.begin()->second)+"ms";
        }
    }

    void reset() override {
        std::lock_guard<std::mutex> lg(mtx);
        process_last_ts_ = 0;
        start_ = std::chrono::system_clock::now();
        processed_frames_ = 0;
        frames_with_objects_ = 0;
        results_.clear();
    }

private:
    std::map<Feature, Duration> callback_intervals_;
    std::vector<CabinRegion> cabin_regions_;
    std::vector<Object::Type> object_types_;
    std::vector<int> object_regions_;
    bool draw_object_id_;
    int frames_with_objects_;
};
