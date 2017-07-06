'''
Created on 24 ene. 2017

@author: esufan
@summary: unittest ConfigurationManager.py module.

-Attributes: -.
-Methods: -.
-Classes: Test_Parameter

'''
import unittest
import sys
sys.path.insert(0, '../src/')
from ConfigurationManager import ConfigurationManager

class Test_Parameter(unittest.TestCase):
    """
    Unittest for class ConfigurationManager.

    Attributes
    ----------
    None-.
    
    Methods
    -------
    testGetElement
    testGetElementAsString
    testGetElementAsInt
    testGetElementAsFloat
    testElementExists
    """
    
    def setUp(self):
        self.p = ConfigurationManager("../TDBP_parameters.xml")

    def testGetElement(self):
        parameter = self.p.get_object_parameter("./AirplaneTrajectory/NomAircraft/look_angle")
        self.assertEqual(parameter.tag,"look_angle")
        self.assertRaises(Exception, self.p.get_object_parameter,"./Im_not_exist")
        
    def testGetElementAsString(self):    
        parameter = self.p.get_string_parameter("./AirplaneTrajectory/NomAircraft/look_angle")
        self.assertEqual(parameter, "30")
    
    def testGetElementAsInt(self):    
        parameter = self.p.get_int_parameter("./AirplaneTrajectory/NomAircraft/look_angle")
        self.assertEqual(parameter, 30)
        
    def testGetElementAsFloat(self):    
        parameter = self.p.get_float_parameter("./AirplaneTrajectory/NomAircraft/look_angle")
        self.assertEqual(parameter, 30.0)
        
    def testElementExists(self):    
        parameter = self.p.parameter_exists("./AirplaneTrajectory/NomAircraft/look_angle")
        self.assertTrue(parameter)
        
        parameter = self.p.parameter_exists("FALSE")
        self.assertFalse(parameter)
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
