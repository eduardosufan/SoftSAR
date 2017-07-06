'''
Created on 18 may. 2017

@author: esufan
@summary: unittest SoftSAR.py module.

-Attributes: -.
-Methods: -.
-Classes: Test_SoftSAR
'''
import unittest
import sys
import numpy as np
sys.path.insert(0, '../src/')
import SoftSAR as SoftSAR
from ConfigurationManager import ConfigurationManager


class Test_SoftSAR(unittest.TestCase):
    """
    Class for testing SoftSAR module.
    
    Methods
    -------
    get_theoretical_resolution
    test_nominal_trajectory
    test_no_nominal_trajectory
    test_accelerated_trajectory
    """

    def setUp(self):
        # Get execution argument to test a chosen trajectory, with a given file path of xml configuration:
        configuration_file_path = "../TDBP_parameters.xml"

        # Object instance to read parameters from xml.
        self.param = ConfigurationManager(configuration_file_path)

	measure_resolution = self.param.get_string_parameter("General/OutputData/measure_resolution_octave_module")
        if measure_resolution != "yes":
            print "To run this test you have to configure measurement of resolution with oct2py."
            self.skipTest(MyTestCase)
            sys.exit()
        
        # Get number of cores to parallel processing.
        self.n_cores = self.param.get_int_parameter("General/OutputData/number_cores_used")


        self.get_theoretical_resolution()
        # Using 10% of error.
        self.percentage = 10.0
        self.percentage_rotated = 20.0

    def get_theoretical_resolution(self):
        """
        Get Theoretical resolution -3dB in range and azimuth for unique target in SAR image.
        range resolution: (0.886*c)/(2*B)
        azimuth resolution: (0.886*lambda)/(4*tan(beamwidth/2))

        Parameters
        ----------
        None.

        Returns
        -------
        range_resolution: float
          Range resolution (-3dB).
        azimuthh_resolution: float
          azimuthh resolution (-3dB).
        K: float
          Absolute constant calibration.
        """

        # Get parameters to obtain resolution.
        c = self.param.get_float_parameter("Radar/c")
        sar_B = self.param.get_float_parameter("Radar/B")
        # Using look angle instead squint angle because flat ground.
        sar_look_angle = (self.param.get_float_parameter("AirplaneTrajectory/NomAircraft/look_angle")*np.pi)/180.

        sar_f0 = self.param.get_float_parameter("Radar/f0")
        sar_beamwidth = (self.param.get_float_parameter("Radar/beamwidth")*np.pi)/180.
        sar_lambda = c/sar_f0

        self.range_resolution_3dB = (0.886*c)/(2*np.sin(sar_look_angle)*sar_B)
        self.azimuth_resolution_3dB = (0.886*sar_lambda)/(4*np.tan(sar_beamwidth/2))

    def test_nominal_trajectory(self):
        """
        Verify if image resolution has a maximum error of 10% with respect to the expected theoretical resolution in nominal trajectory.
        """

        # Call SoftSAR processor to obtain image resolution.
        range_resolution, azimuth_resolution, K = SoftSAR.test_nominal_trajectory(self.param, self.n_cores)

        # Computing percentage error in both directions.
        percentage_range_error = (np.abs(self.range_resolution_3dB - range_resolution)*100)/range_resolution
        percentage_azimuth_error = (np.abs(self.azimuth_resolution_3dB - azimuth_resolution)*100)/azimuth_resolution

        # Verify if error is less than 10% in both directions.
        self.assertLessEqual(percentage_range_error, self.percentage)
        self.assertLessEqual(percentage_azimuth_error, self.percentage)
        
    def test_no_nominal_trajectory(self):
        """
        Verify if image resolution has a maximum error of 10% with respect to the expected theoretical resolution in no nominal trajectory.
        """

        # Call SoftSAR processor to obtain image resolution.
        range_resolution, azimuth_resolution, K = SoftSAR.test_no_nominal_trajectory(self.param, self.n_cores)

        # Computing percentage error in both directions.
        percentage_range_error = (np.abs(self.range_resolution_3dB - range_resolution)*100)/range_resolution
        percentage_azimuth_error = (np.abs(self.azimuth_resolution_3dB - azimuth_resolution)*100)/azimuth_resolution

        # Verify if error is less than 10% in both directions.
        self.assertLessEqual(percentage_range_error, self.percentage_rotated)
        self.assertLessEqual(percentage_azimuth_error, self.percentage_rotated)


    def test_measure_resolution_ok(self):
        """
        Verify if "General/OutputData/measure_resolution_octave_module" is set to yes in xml configuration file.
        """

        measure_resolution = self.param.get_string_parameter("General/OutputData/measure_resolution_octave_module")
        self.assertEqual(measure_resolution, "yes")

    def test_accelerated_trajectory(self):
        """
        Verify if image resolution has a maximum error of 10% with respect to the expected theoretical resolution in accelerated trajectory.
        """

        # Call SoftSAR processor to obtain image resolution.
        range_resolution, azimuth_resolution, K = SoftSAR.test_accelerated_trajectory(self.param, self.n_cores)

        # Computing percentage error in both directions.
        percentage_range_error = (np.abs(self.range_resolution_3dB - range_resolution)*100)/range_resolution
        percentage_azimuth_error = (np.abs(self.azimuth_resolution_3dB - azimuth_resolution)*100)/azimuth_resolution

        # Verify if error is less than 10% in both directions.
        self.assertLessEqual(percentage_range_error, self.percentage)
        self.assertLessEqual(percentage_azimuth_error, self.percentage)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
