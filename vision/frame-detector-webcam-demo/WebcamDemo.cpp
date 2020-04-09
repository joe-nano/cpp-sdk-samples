#include "AFaceListener.h"
#include "PlottingImageListener.h"
#include "PlottingObjectListener.h"
#include "PlottingOccupantListener.h"
#include "StatusListener.h"
#include "FileUtils.h"

#include <Core.h>
#include <FrameDetector.h>
#include <SyncFrameDetector.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace affdex;
namespace po = boost::program_options; // abbreviate namespace


static const std::string DISPLAY_DATA_DIR_ENV_VAR = "AFFECTIVA_VISION_DATA_DIR";
static const affdex::Str DATA_DIR_ENV_VAR = STR(DISPLAY_DATA_DIR_ENV_VAR);

struct ProgramOptions {

    enum DetectionType {
        FACE,
        OBJECT,
        OCCUPANT
    };
    // cmd line args
    affdex::Path data_dir;
    affdex::Path output_file_path;
    affdex::Path output_video_path;
    std::vector<int> resolution;
    int process_framerate;
    int camera_framerate;
    int camera_id;
    unsigned int num_faces;
    bool draw_display = true;
    bool sync = false;
    bool draw_id = true;
    bool disable_logging = false;
    bool write_video = false;
    cv::VideoWriter output_video;
    DetectionType detection_type = FACE;
};

void assembleProgramOptions(po::options_description& description, ProgramOptions& program_options) {
    const std::vector<int> DEFAULT_RESOLUTION{1280, 720};

    description.add_options()
        ("help,h", po::bool_switch()->default_value(false), "Display this help message.")
#ifdef _WIN32
    ("data,d", po::wvalue<affdex::Path>(&program_options.data_dir),
            std::string("Path to the data folder. Alternatively, specify the path via the environment variable "
                + DISPLAY_DATA_DIR_ENV_VAR + R"(=\path\to\data)").c_str())
    ("output,o", po::wvalue<affdex::Path>(&program_options.output_video_path), "Output video path.")

#else //  _WIN32
        ("data,d", po::value<affdex::Path>(&program_options.data_dir),
         (std::string("Path to the data folder. Alternatively, specify the path via the environment variable ")
             + DATA_DIR_ENV_VAR + "=/path/to/data").c_str())
        ("output,o", po::value<affdex::Path>(&program_options.output_video_path), "Output video path.")

#endif // _WIN32
        ("resolution,r",
         po::value<std::vector<int>
         >(&program_options.resolution)->default_value(DEFAULT_RESOLUTION, "1280 720")->multitoken(),
         "Resolution in pixels (2-values): width height")
        ("pfps", po::value<int>(&program_options.process_framerate)->default_value(30), "Processing framerate.")
        ("cfps", po::value<int>(&program_options.camera_framerate)->default_value(30), "Camera capture framerate.")
        ("cid", po::value<int>(&program_options.camera_id)->default_value(0), "Camera ID.")
        ("numFaces",
         po::value<unsigned int>(&program_options.num_faces)->default_value(1),
         "Number of faces to be tracked.")
        ("draw", po::value<bool>(&program_options.draw_display)->default_value(true), "Draw metrics on screen.")
        ("sync",
         po::bool_switch(&program_options.sync)->default_value(false),
         "Process frames synchronously. Note this will process all frames captured by the camera and will ignore the value in --pfps")
        ("quiet,q",
         po::bool_switch(&program_options.disable_logging)->default_value(false),
         "Disable logging to console")
        ("face_id",
         po::value<bool>(&program_options.draw_id)->default_value(true),
         "Draw face id on screen. Note: Drawing to screen must be enabled.")
        ("file,f", po::value<affdex::Path>(&program_options.output_file_path), "Name of the output CSV file.")
        ("object", "Enable object detection")
        ("occupant", "Enable occupant detection");
}

bool processFrameFromWebcam(unique_ptr<vision::Detector>& frame_detector, ProgramOptions& program_options,
                            cv::VideoCapture& webcam, const std::chrono::system_clock::time_point& start_time,
                            vision::Frame& frame) {

    cv::Mat img;
    if (!webcam.read(img)) {   //Capture an image from the camera
        std::cerr << "Failed to read frame from webcam\n";
        return false;
    }

    const Timestamp ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now() - start_time).count();

    // Create a Frame from the webcam image and process it with the Detector
    frame = vision::Frame(img.size().width, img.size().height, img.data, vision::Frame::ColorFormat::BGR, ts);
    if (program_options.sync) {
        dynamic_cast<vision::SyncFrameDetector*>(frame_detector.get())->process(frame);
    }
    else {
        dynamic_cast<vision::FrameDetector*>(frame_detector.get())->process(frame);
    }

    return true;
}

void processFaceStream(unique_ptr<vision::Detector>& frame_detector, std::ofstream& csv_file_stream,
                       ProgramOptions& program_options, StatusListener& status_listener, cv::VideoCapture& webcam) {

    // prepare listeners
    PlottingImageListener image_listener(csv_file_stream, program_options.draw_display, !program_options
        .disable_logging, program_options.draw_id);
    AFaceListener face_listener;

    // configure the Detector by enabling features and assigning listeners
    frame_detector->enable({vision::Feature::EMOTIONS, vision::Feature::EXPRESSIONS, vision::Feature::IDENTITY,
                            vision::Feature::APPEARANCES});
    frame_detector->setImageListener(&image_listener);
    frame_detector->setFaceListener(&face_listener);
    frame_detector->setProcessStatusListener(&status_listener);
    const auto start_time = std::chrono::system_clock::now();

    //Start the frame detector thread.
    frame_detector->start();

    vision::Frame frame;
    do {
        if (!processFrameFromWebcam(frame_detector, program_options, webcam, start_time, frame)) {
            break;
        }

        image_listener.processResults();
        //To save output video file
        if (program_options.write_video) {
            program_options.output_video << image_listener.getImageData();
        }
    }
#ifdef _WIN32
        while (!GetAsyncKeyState(VK_ESCAPE) && status_listener.isRunning());
#else //  _WIN32
    while (status_listener.isRunning() && (cv::waitKey(20) != 27)); // ascii for ESC
#endif
}

void processObjectStream(unique_ptr<vision::Detector>& frame_detector, std::ofstream& csv_file_stream,
                         ProgramOptions& program_options, StatusListener& status_listener, cv::VideoCapture& webcam) {

    // prepare listeners
    PlottingObjectListener object_listener(csv_file_stream,
                                           program_options.draw_display,
                                           !program_options.disable_logging,
                                           program_options.draw_id,
                                           {{vision::Feature::CHILD_SEATS, 1000}, {vision::Feature::PHONES, 1000}},
                                           frame_detector->getCabinRegionConfig().getRegions());


    // configure the Detector by enabling features and assigning listeners
    frame_detector->enable({vision::Feature::CHILD_SEATS, vision::Feature::PHONES});
    frame_detector->setObjectListener(&object_listener);
    frame_detector->setProcessStatusListener(&status_listener);

    const auto start_time = std::chrono::system_clock::now();

    //Start the frame detector thread.
    frame_detector->start();

    vision::Frame frame;
    do {
        if (!processFrameFromWebcam(frame_detector, program_options, webcam, start_time, frame)) {
            break;
        }
        object_listener.processResults(frame);
        //To save output video file
        if (program_options.write_video) {
            program_options.output_video << object_listener.getImageData();
        }
    }
#ifdef _WIN32
        while (!GetAsyncKeyState(VK_ESCAPE) && status_listener.isRunning());
#else //  _WIN32
    while (status_listener.isRunning() && (cv::waitKey(20) != 27)); // ascii for ESC
#endif
}

void processOccupantStream(unique_ptr<vision::Detector>& frame_detector,
                           std::ofstream& csv_file_stream,
                           ProgramOptions& program_options,
                           StatusListener& status_listener,
                           cv::VideoCapture& webcam) {

    // prepare listeners
    PlottingOccupantListener occupant_listener(csv_file_stream, program_options.draw_display, !program_options
        .disable_logging, program_options.draw_id, 500, frame_detector->getCabinRegionConfig().getRegions());


    // configure the Detector by enabling features and assigning listeners
    frame_detector->enable(vision::Feature::OCCUPANTS);
    frame_detector->setOccupantListener(&occupant_listener);
    frame_detector->setProcessStatusListener(&status_listener);

    const auto start_time = std::chrono::system_clock::now();

    //Start the frame detector thread.
    frame_detector->start();

    vision::Frame frame;
    do {
        if (!processFrameFromWebcam(frame_detector, program_options, webcam, start_time, frame)) {
            break;
        }

        occupant_listener.processResults(frame);
        //To save output video file
        if (program_options.write_video) {
            program_options.output_video << occupant_listener.getImageData();
        }
    }
#ifdef _WIN32
        while (!GetAsyncKeyState(VK_ESCAPE) && status_listener.isRunning());
#else //  _WIN32
    while (status_listener.isRunning() && (cv::waitKey(20) != 27)); // ascii for ESC
#endif
}

int main(int argsc, char** argsv) {

    std::cout << "Hit ESCAPE key to exit app.." << endl;
    unique_ptr<vision::Detector> frame_detector;

    try {
        //setting up output precision
        const int precision = 2;
        std::cerr << std::fixed << std::setprecision(precision);
        std::cout << std::fixed << std::setprecision(precision);

        //Gathering program options
        ProgramOptions program_options;

        po::options_description description
            ("Project for demoing the Affdex SDK Detector class (grabbing and processing frames from the camera).");
        assembleProgramOptions(description, program_options);

        //verifying command line arguments
        po::variables_map args;
        try {
            po::store(po::command_line_parser(argsc, argsv).options(description).run(), args);
            if (args["help"].as<bool>()) {
                std::cout << description << std::endl;
                return 0;
            }
            po::notify(args);
        }
        catch (po::error& e) {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << "For help, use the -h option.\n\n";
            return 1;
        }

        //Check for object or occupant argument present or not. If nothing is present then enable face by default.

        if (args.count("object") && args.count("occupant")) {
            std::cerr << "Can't turn on Object and occupant detection\n";
            std::cerr << "ERROR: Can't use --occupant and --object at the same time\n";
            std::cerr << "For help, use the -h option.\n\n";
            return 1;
        }
        else if (args.count("object")) {
            std::cout << "Setting up object detection\n";
            program_options.detection_type = program_options.OBJECT;
        }
        else if (args.count("occupant")) {
            std::cout << "Setting up occupant detection\n";
            program_options.detection_type = program_options.OCCUPANT;
        }
        else {
            std::cout << "Setting up face detection\n";
        }

        program_options.write_video = args.count("output");
        if (program_options.write_video) {
            // must use .avi extension!
            string output_ext = boost::filesystem::extension(program_options.output_video_path);
            if (output_ext != ".avi") {
                std::cerr << "Invalid output file extension, must use .avi\n";
                return 1;
            }
        }

        // set data_dir to env_var if not set on cmd line
        program_options.data_dir = validatePath(program_options.data_dir, DATA_DIR_ENV_VAR);

        if (program_options.resolution.size() != 2) {
            std::cerr << "Only two numbers must be specified for resolution.\n";
            return 1;
        }

        if (program_options.resolution[0] <= 0 || program_options.resolution[1] <= 0) {
            std::cerr << "Resolutions must be positive numbers.\n";
            return 1;
        }

        if (program_options.draw_id && !program_options.draw_display) {
            std::cerr << "Can't draw face id while drawing to screen is disabled\n";
            std::cerr << description << std::endl;
            return 1;
        }

        //initialize the output file
        boost::filesystem::path csv_path(program_options.output_file_path);
        csv_path.replace_extension(".csv");
        std::ofstream csv_file_stream(csv_path.c_str());

        if (!csv_file_stream.is_open()) {
            std::cerr << "Unable to open csv file " << program_options.output_file_path << std::endl;
            return 1;
        }

        // create the Detector
        if (program_options.sync) {
            frame_detector = std::unique_ptr<vision::Detector>(new vision::SyncFrameDetector(program_options.data_dir,
                                                                                             program_options.num_faces));
        }
        else {
            frame_detector =
                std::unique_ptr<vision::Detector>(new vision::FrameDetector(program_options.data_dir,
                                                                            program_options.process_framerate,
                                                                            program_options.num_faces));
        }
        // prepare listeners
        StatusListener status_listener;

        // Connect to the webcam and configure it
        cv::VideoCapture webcam(program_options.camera_id);

        // Note: not all webcams support these configuration properties
        webcam.set(CV_CAP_PROP_FPS, program_options.camera_framerate);
        webcam.set(CV_CAP_PROP_FRAME_WIDTH, program_options.resolution[0]);
        webcam.set(CV_CAP_PROP_FRAME_HEIGHT, program_options.resolution[1]);

        const auto start_time = std::chrono::system_clock::now();
        if (!webcam.isOpened()) {
            std::cerr << "Error opening webcam\n";
            return 1;
        }
        //Setup video writer
        if (program_options.write_video) {
            cv::Mat mat;
            if (!webcam.read(mat)) {   //Capture an image from the camera
                std::cerr << "Failed to read frame from webcam while setting up video writer\n";
                return 1;
            }
            program_options.output_video.open(program_options.output_video_path,
                                              CV_FOURCC('D', 'X', '5', '0'),
                                              program_options.camera_framerate,
                                              cv::Size(mat.size().width, mat.size().height),
                                              true);
            if (!program_options.output_video.isOpened()) {
                std::cerr << "Error opening output video: " << program_options.output_video_path << std::endl;
                return 1;
            }
        }
        switch (program_options.detection_type) {
            case program_options.OBJECT:
                processObjectStream(frame_detector, csv_file_stream, program_options, status_listener, webcam);
                break;
            case program_options.OCCUPANT:
                processOccupantStream(frame_detector, csv_file_stream, program_options, status_listener, webcam);
                break;
            case program_options.FACE:
                processFaceStream(frame_detector, csv_file_stream, program_options, status_listener, webcam);
                break;
            default:
                std::cerr << "This should never happen " << program_options.detection_type << std::endl;
                return 1;
        }

        frame_detector->stop();
        csv_file_stream.close();

        if (boost::filesystem::exists(program_options.output_file_path)) {
            std::cout << "Output written to file" << program_options.output_file_path << std::endl;
        }
    }
    catch (std::exception& ex) {
        StatusListener::printException(ex);
        if (frame_detector){
            frame_detector->stop();
        }
        return 1;
    }

    return 0;
}
