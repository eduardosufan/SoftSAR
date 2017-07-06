# **SoftSAR**: Software for SAR image focusing.
- Author: Eduardo Sufan 
 - eduardosufan333@gmail.com
- Colaborator: Matias Marenchino
 - mlmarenchino@gmail.com
- Colaborator: Marc Thibeault 
 - mthibeault@conae.gov.ar 
- Scientific advisor: Tomas Zajc
 - tzajc@conae.gov.ar
- Author of original TDBP codes: 
 - Mauro Mariotti mauro.mariotti.83@gmail.com
- Date: Year 2016-2017

BRIEF
--------------
SoftSAR is a software application destinated to SAR image focusing. Implemented in Python, is capable to focus with Time Domain Back projection algorithm range compressed SAR images. 
In summary, according to an xml configuration, an airplane trajectory is simulated and a SAR range compressed image. Then, TDBP algorithm is applied to focus the image. Finally, range and azimuth resolution is obtained to be compared with teotherical expected resolution.
You can add new classes and methods to test different types of focusing algorithms, or different kinds of movile platform trajectories.
Resultant SAR image focused actually has a maximum of 1.1% of error respective to teotherical resolution.

FILES OVERVIEW
--------------
The directories has the following folders:
- src: contains all codes of the software "SoftSAR" TDBP focus algorithm
- test: contains associated testing.

FEATURES
--------
- Programmed in Python 2.7.
- Platform independent.
- Unit tested.
- Aircraft trajectory configurable.
- Range compressed image simulation according to flight path of aircraft.
- SAR focus image with Time Domain Back Projection Algorithm.
- L1 SAR image generated with user friendly graphics of aircraft trajectory, simulated image and focused image.
- All modules, classes and functions are documented with docstrings. Doxygen compatible.

INSTALLATION
------------
- No installation for SoftSAR is needed, just download code, configure and execute SoftSAR.py.
- You may need some additional libraries:
 * sudo apt-get install octave python-tk
 * pip install pyparsing oct2py cycler numpy matplotlib scipy

SoftSAR USAGE
-------------
To execute SoftSAR processor, go to "/src" folder and execute the following python module, with the additional 
command-line arguments to provide xml configuration file path and to choose the aircraft trajectory to be simulated:

    >>>python SoftSAR.py "xml configuration path" "trajectory type"

  Where "xml_configuration_path" must be xml confiration file path (relative path) and "trajectory type" can be one of the following options (string):
 
    a) "nominal trajectory"
    b) "no nominal trajectory"
    c) "accelerated trajectory"

SoftSAR CONFIGURATION
---------------------
To configure execution of SoftSAR processor, and parameters of SAR and trajectory you must edit the xml file
configuration "TDBP_parameters.xml".

DOCUMENTATION
-------------
- You can use doxygen file executing: doxygen SoftSAR_doxygen. This generates code documentation in html 
  (please refer to index.html file), latex and rtf format located in /Codigos/doxygen. All docstrings of modules, classes 
  and functions are loaded into user friendly file.

DETAILED INFORMATION
--------------------


1) SoftSAR USAGE
----------------

To execute SoftSAR processor execute the following python module, with the additional 
command-line arguments to provide xml configuration file path and to choose the aircraft trajectory to be simulated:

    >>>python SoftSAR.py "xml configuration path" "trajectory type"

  Where "xml_configuration_path" must be xml confiration file path (relative path) and "trajectory type" can be one of the following options (string):
 
    a) "nominal trajectory"
    b) "no nominal trajectory"
    c) "accelerated trajectory"

    a) In nominal trajectory, the ideal trajectory of the aircraft is simulated by moving on 
       the x axis of the coordinate system. Velocity is constant, and acceleration is zero.
       NOTE: this trajectory is in straight line over the target, with constant velocity read 
       from xml configuration file.
       It forces attitude and flight angles to zero values, independent of values saved into
       xml configuration file.

    b) In no nominal trajectory, the trajectory of the aircraft is simulated by moving over 
       x,y and z axis of the coordinate system, by computing trajectory based on "FlightParam"
       parameters set in xml configuration file.
       Velocity is constant in every axis, and acceleration is zero.
       NOTE: this trajectory is simulated reading the following parameters in xml configuration file.
             you must fill the corresponding values to simulate the deviation of the nominal trajectory.
 
              -FlightParam are parameters of flight to move aircraft over x, y and z axis.
               For example, with yaw: 5, The aircraft is directed to the next every point of space according to 
               a 5 degree deviation on the nominal x-axis.

               <FlightParam>
                 <pitch>0</pitch>
                 <roll >0</roll>
                 <yaw  >5</yaw>
               </FlightParam>

              -FlightAttitude are parameters of aicraft attitude, to simulate the orientation of aircraft 
               over flight path over corresponding axis. For example, for yaw:3, the aircraft moves over x
	       axis looking 3 degrees to right.
           
               <FlightAttitude>
                 <pitch>0</pitch>
                 <roll >0</roll>
                 <yaw  >3</yaw>
               </FlightAttitude>

    c) In accelerated trajectory the acceleration is constant over x axis.
       Trajectory is linear over x axis, in straight line over the target with non-constant velocity.
       To use this function set v_init < v_nom in parameters file, to have an accelerated trajectory.
       For example:
         <v_nom       >103</v_nom>
         <v_init      >40</v_init>
	    

       It forces attitude and flight angles to zero values, independent of values saved into
       xml configuration file.

**************************************************************************************************************
  
2) SoftSAR CONFIGURATION
------------------------

To configure execution of SoftSAR processor, and parameters of SAR and trajectory you must edit the xml file
configuration "TDBP_parameters.xml". Parameters are:  

    GENERAL PARAMETERS
    output_directory: string - relative path to save outputs.  
    measure_resolution_octave_module: string - "yes" or "no" to measure image resolution with octave function. (DEFAULT: "no")  
    number_cores_used: int - number of cores to be used during image focussing. (DEFAULT: 2)  

    SAR PARAMETERS
    c: Light speed (DEFAULT: 300000000)  
    f0: SAR central frequency (DEFAULT: 1300000000)  
    PRF: SAR Pulse Repetition Frequency (DEFAULT: 250)  
    fs: sar Sample frequency (DEFAULT: 50000000)  
    beamwidth: SAR beamwidth (DEFAULT: 6.16)  
    beamwidth_threshold: SAR beamwidth threshold used for limit antenna ilumination. (DEFAULT: 2)  
    B: SAR bandwith (DEFAULT: 38000000)  

    AIRCRAFT PARAMETERS
    NOMINAL FLIGHT
    z: aircraft height (DEFAULT: 4900)  
    look_angle: aircraft look angle (DEFAULT: 30)  
    squint_angle: aircraft squint angle (DEFAULT: 0)  
    v_nom: nominal aircraft velocity over x axis (DEFAULT: 103)  
    v_init: initial aircraft velocity over x axis (DEFAULT: 40)  
  
    FLIGHT PARAMETERS  
    pitch: pitch angle to compute next position with an y axis rotation (DEFAULT: 0)  
    roll: roll angle to compute next position with an x axis rotation (DEFAULT: 0)  
    yaw: yaw angle to compute next position with an z axis rotation (DEFAULT: 5)  

    ATTITUDE PARAMETERS  
    pitch: pitch attitude angle (DEFAULT: 0)  
    roll: roll attitude angle (DEFAULT: 0)  
    yaw: yaw attitude angle (DEFAULT: 5)  

    az_bw_sim_enl: beamwidth SAR enlobe. If 1 it simulates the -3dB azimuth lobe. Based on the nominal trajectory (DEFAULT: 3)  

    IMAGE SIMULATION
    Fast_time_pixel_margin_mono: Pixel margin in the fast time (DEFAULT: 10)  

    IMAGE FOCUSING
    n_pixels_nominal_range_resolution: number of pixels representing nominal resolution in range. (DEFAULT: 5)  
    n_pixels_nominal_azimuth_resolution: number of pixels representing nominal resolution in azimuth. (DEFAULT: 3)  
    length_range_axis: length of range axis. Axis will be "length times" nominal range resolution. (DEFAULT: 3)  
    length_azimuth_axis: length of azimuth axis. Axis will be "length times" nominal azimuth resolution. (DEFAULT: 9)  

