'''
Created on 14 jul. 2016

@author: esufan
@summary: SoftSAR program for focusing SAR images. IMPORTANT: For more information please refer to README.txt.
@email: eduardosufan333@gmail.com

1) SoftSAR USAGE
   -------------
To execute SoftSAR processor execute the following python module, with the additional 
command-line argument to choose the aircraft trajectory to be simulated:

    >>>python SoftSAR.py "path to configuration file" "trajectory type"

  where: "path to configuration file" is the relative path of xml configuration file. DEFAULT: TDBP_parameters.xml"

  "trajectory type" can be one of the following options (string):
 
    a) "nominal trajectory"
    b) "no Nominal trajectory"
    c) "accelerated trajectory"


2) SoftSAR CONFIGURATION
   ---------------------
To configure execution of SoftSAR processor, and parameters of SAR and trajectory you must edit the xml file
configuration "TDBP_parameters.xml".



Module features:
-Attributes: -.
-Methods:
  test_nominal_trajectory
  test_no_nominal_trajectory
  test_accelerated_trajectory
-Classes: -.

'''
# -*- coding: utf-8 -*-

from ConfigurationManager import ConfigurationManager
from AirplaneTrajectory import AirplaneTrajectory
from SimulatedImage import SimulatedImage
from TDBP import TDBP
import Utils as Utils
import time
import sys
import numpy as np

def test_nominal_trajectory(param, n_cores):
    """
    Test nominal trajectory, simulating a range compressed image and focusing it with TDBP focus.

    Parameters
    ----------
    param: object (ConfigurationManager instance).
      ConfigurationManager instance to read parameters from file.
      
    Returns (if "yes" in parameter of configuration file: General/OutputData/measure_resolution_octave_module)
    -------
    range_resolution: float
      Range resolution (-3dB).
    azimuth_resolution: float
      Azimuth resolution (-3dB).
    K: float
      Absolute constant calibration.
    """

    range_resolution = None
    azimuth_resolution = None
    K = None

    print "Executing SoftSAR TDBP processor - nominal trajectory."

    # Generate nominal trajectory.
    trajectory_nom = AirplaneTrajectory(param)
    trajectory_nom.generate_MRU_nominal()
    Utils.plot_trajectory_2D(trajectory_nom)

    # Simulate SAR range compressed image with the trajectory previously generated.
    si_nom = SimulatedImage(param, trajectory_nom)
    si_nom.generate_img(param)
    Utils.plot_image(si_nom.image, "Simulated SAR Image [power]- NOMINAL TRAJECTORY", "rainbow")

    # Focus simulated image  previously generated with TDBP.
    start = time.time()
    tdbp_nom = TDBP(param, si_nom)

    tdbp_nom.multiprocessing_image(param, n_cores)

    end = time.time()
    print "Focusing Time: Nominal trajectory", (end - start)
    Utils.plot_image(tdbp_nom.focused_image, "Focused SAR Image [dB] - Nominal Trajectory", "rainbow", tdbp_nom.foc_y, tdbp_nom.foc_x)
    print "Azimuth pixel spacing: ", tdbp_nom.azimuth_pixel_spacing, "[m]"
    print "Range pixel spacing: ", tdbp_nom.range_pixel_spacing, "[m]"

    measure_resolution =  param.get_string_parameter("General/OutputData/measure_resolution_octave_module")
    if measure_resolution == "yes":
        
        range_resolution, azimuth_resolution, K = Utils.measure_image_resolution_octave(tdbp_nom.focused_image, 201.0, 201.0, 38.0, 64.0,  tdbp_nom.azimuth_pixel_spacing, tdbp_nom.range_pixel_spacing, 30.0)

        print "Range resolution (-3dB): ", range_resolution, "[m]"
        print "Azimuth resolution (-3dB): ", azimuth_resolution, "[m]"
        print "Calibration Constant: ", K

    # Get output directory.
    output_directory =  param.get_string_parameter("General/OutputData/output_directory")
    Utils.save_data_mat_file(tdbp_nom.focused_image, output_directory + "/nominal_image.mat")
    return range_resolution, azimuth_resolution, K

#***********************************************************************************************************************************************************
#***********************************************************************************************************************************************************

def test_no_nominal_trajectory(param, n_cores):
    """
    Test no nominal trajectory (setting attitude and flight parameters from xml), simulating a range compressed image and focusing it with TDBP focus.

    Parameters
    ----------
    param: object (ConfigurationManager instance).
      ConfigurationManager instance to read parameters from file.

    Returns (if "yes" in parameter of configuration file: General/OutputData/measure_resolution_octave_module)
    -------
    range_resolution: float
      Range resolution (-3dB).
    azimuth_resolution: float
      Azimuth resolution (-3dB).
    K: float
      Absolute constant calibration.
    """

    range_resolution = None
    azimuth_resolution = None
    K = None

    print "Executing SoftSAR TDBP processor - no nominal trajectory."

    # Generate no nominal trajectory.
    trajectory_no_nom = AirplaneTrajectory(param)
    trajectory_no_nom.generate_MRU_no_nominal()
    Utils.plot_trajectory_2D(trajectory_no_nom)
    Utils.plot_trajectory_3D(trajectory_no_nom)

    # Simulate SAR range compressed image with the trajectory previously generated.
    si_no_nom = SimulatedImage(param, trajectory_no_nom)
    si_no_nom.generate_img(param)
    Utils.plot_image(si_no_nom.image, "Simulated SAR Image [power]- NO NOMINAL TRAJECTORY", "rainbow")

    # Focus simulated image  previously generated with TDBP.
    start = time.time()
    tdbp_no_nom = TDBP(param, si_no_nom)
    tdbp_no_nom.multiprocessing_image(param, n_cores)
    end = time.time()
    print "Focusing Time: No nominal trajectory", (end - start)

    Utils.plot_image(tdbp_no_nom.focused_image, "Focused SAR Image [dB] - No nominal Trajectory", "rainbow", tdbp_no_nom.foc_y, tdbp_no_nom.foc_x)
    print "Azimuth pixel spacing: ", tdbp_no_nom.azimuth_pixel_spacing, "[m]"
    print "Range pixel spacing: ", tdbp_no_nom.range_pixel_spacing, "[m]"

    measure_resolution =  param.get_string_parameter("General/OutputData/measure_resolution_octave_module")
    if measure_resolution == "yes":
        range_resolution, azimuth_resolution, K = Utils.measure_image_resolution_octave(tdbp_no_nom.focused_image, 201.0, 201.0, 38.0, 64.0,  tdbp_no_nom.azimuth_pixel_spacing, tdbp_no_nom.range_pixel_spacing, 30.0)

        print "Range resolution (-3dB): ", range_resolution, "[m]"
        print "Azimuth resolution (-3dB): ", azimuth_resolution, "[m]"
        print "Calibration Constant: ", K

    # Get output directory.
    output_directory =  param.get_string_parameter("General/OutputData/output_directory")
    Utils.save_data_mat_file(tdbp_no_nom.focused_image, output_directory + "/no_nominal_image.mat")
    return range_resolution, azimuth_resolution, K

#***********************************************************************************************************************************************************
#***********************************************************************************************************************************************************

def test_accelerated_trajectory(param, n_cores):
    """
    Test accelerated trajectory, simulating a range compressed image and focusing it with TDBP focus.

    Parameters
    ----------
    param: object (ConfigurationManager instance).
      ConfigurationManager instance to read parameters from file.
      
    Returns (if "yes" in parameter of configuration file: General/OutputData/measure_resolution_octave_module)
    -------
    range_resolution: float
      Range resolution (-3dB).
    azimuth_resolution: float
      Azimuth resolution (-3dB).
    K: float
      Absolute constant calibration.
    """

    range_resolution = None
    azimuth_resolution = None
    K = None

    print "Executing SoftSAR TDBP processor - accelerated trajectory."

    # Generate accelerated trajectory.
    trajectory_acc = AirplaneTrajectory(param)
    trajectory_acc.generate_MRUV_x_axis()
    Utils.plot_trajectory_2D(trajectory_acc)

    # Simulate SAR range compressed image with the trajectory previously generated.
    si_acc = SimulatedImage(param, trajectory_acc)
    si_acc.generate_img(param)
    Utils.plot_image(si_acc.image, "Simulated SAR Image [power]- ACCELERATED TRAJECTORY", "rainbow")

    # Focus simulated image  previously generated with TDBP.
    start = time.time()
    tdbp_acc = TDBP(param, si_acc)
    tdbp_acc.multiprocessing_image(param, n_cores)
    end = time.time()
    print "Focusing Time: Accelerated trajectory", (end - start)

    Utils.plot_image(tdbp_acc.focused_image, "Focused SAR Image [dB] - Accelerated Trajectory", "rainbow", tdbp_acc.foc_y, tdbp_acc.foc_x)
    print "Azimuth pixel spacing: ", tdbp_acc.azimuth_pixel_spacing, "[m]"
    print "Range pixel spacing: ", tdbp_acc.range_pixel_spacing, "[m]"

    measure_resolution =  param.get_string_parameter("General/OutputData/measure_resolution_octave_module")
    if measure_resolution == "yes":
        range_resolution, azimuth_resolution, K = Utils.measure_image_resolution_octave(tdbp_acc.focused_image, 201.0, 201.0, 38.0, 64.0,  tdbp_acc.azimuth_pixel_spacing, tdbp_acc.range_pixel_spacing, 30.0)

        print "Range resolution (-3dB): ", range_resolution, "[m]"
        print "Azimuth resolution (-3dB): ", azimuth_resolution, "[m]"
        print "Calibration Constant: ", K

    # Get output directory.
    output_directory =  param.get_string_parameter("General/OutputData/output_directory")
    Utils.save_data_mat_file(tdbp_acc.focused_image, output_directory + "/accelerated_image.mat")
    return range_resolution, azimuth_resolution, K

if __name__ == '__main__':

    # Check for existence of input argument 1 and 2:
    if len(sys.argv)!=3:
        raise Exception("Incorrect input. You must add command-line arguments with xml configuration path (relative), a valid trajectory: nominal trajectory, no nominal trajectory or accelerated trajectory.\n")
        sys.exit()
    else:
        # Get execution argument to test a chosen trajectory, with a given file path of xml configuration:
        configuration_file_path = sys.argv[1]
        test_trajectory = sys.argv[2]


    # Object instance to read parameters from xml.
    param = ConfigurationManager(configuration_file_path)

    # Get number of cores to parallel processing.
    n_cores = param.get_int_parameter("General/OutputData/number_cores_used")
    print "Number of cores to be used during focusing: ", n_cores

    # Run the chosen trajectory.
    if test_trajectory=="nominal trajectory":
        test_nominal_trajectory(param, n_cores)
    elif test_trajectory=="no nominal trajectory":
        test_no_nominal_trajectory(param, n_cores)
    elif test_trajectory=="accelerated trajectory":
        test_accelerated_trajectory(param, n_cores)
    else: 
        raise Exception("Incorrect trajectory. Valid trajectories are: nominal trajectory, no nominal trajectory or accelerated trajectory.\n")

    print "END of SoftSAR execution."
    sys.exit()



