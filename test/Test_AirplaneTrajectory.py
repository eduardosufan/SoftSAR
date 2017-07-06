'''
Created on 13 feb. 2017

@author: esufan
@summary: unittest Test_AirplaneTrajectory.py module.

-Attributes: -.
-Methods: -.
-Classes: Test_AirplaneTrajectory

'''
import unittest
import numpy as np
import sys
sys.path.insert(0, '../src/')
from ConfigurationManager import ConfigurationManager
from AirplaneTrajectory import AirplaneTrajectory

class Test_AirplaneTrajectory(unittest.TestCase):
    """
    Unittest class for AirplaneTrajectory.
    
    Attributes
    ----------
    param: object (ConfigurationManager instance).
      ConfigurationManager instance to read parameters from file.
    traj_nom: object (AirplaneTrajectory instance).
      AirplaneTrajectory instance for nominal trajectory.
    traj_no_nom: object (AirplaneTrajectory instance).
      AirplaneTrajectory instance for no nominal trajectory.
    traj_acc: object (AirplaneTrajectory instance).
      AirplaneTrajectory instance for no nominal accelerated trajectory.

    Methods
    -------
    test_generate_MRU_nom
    test_generate_MRU_no_nom
    test_generate_MRUV_x
    """

    def setUp(self):
        self.param = ConfigurationManager("../TDBP_parameters.xml")
        self.traj_nom = AirplaneTrajectory(self.param)
        self.traj_no_nom = AirplaneTrajectory(self.param)
        self.traj_acc = AirplaneTrajectory(self.param)

    def test_generate_MRU_nom(self):
        """
        Check if all items in x axis have the same spatial difference (for nominal uniform trajectory)
        flight_y and flight_z must be zero.
        
        flight_vx must be constant over all trajectory, and flight_vy and flight_vz must be zero.
        """
        l_spatial_diff = []
        vel = self.param.get_float_parameter("./AirplaneTrajectory/NomAircraft/v_nom")
        self.traj_nom.generate_MRU_nominal()
        
        # get spatial difference between two coordinates in x.

        for i in range(np.size(self.traj_nom.flight_x)):
            if i!=self.traj_nom.flight_x.size - 1: # Border condition
                spatial_diff = self.traj_nom.flight_x[i] - self.traj_nom.flight_x[i+1]
                l_spatial_diff.append(spatial_diff)
            
        # verify if every spatial difference is almost the same. So we have an equally spaced trajectory over x.
        for i in range(len(l_spatial_diff)):
            if i!=len(l_spatial_diff) - 1: # Border condition
                self.assertAlmostEqual(l_spatial_diff[i], l_spatial_diff[i+1])

        # Check for 0 deviation over y and z axis.
        self.assertTrue(all(i == 0.0 for i in self.traj_nom.flight_y))
        self.assertTrue(all(i == 0.0 for i in self.traj_nom.flight_z))

        # check uniform velocity over x axis.
        self.assertEqual(self.traj_nom.flight_vx[0], vel)
        self.assertTrue(all(i == self.traj_nom.flight_vx[0] for i in self.traj_nom.flight_vx))

        # All velocities in y and z axis must be zero.
        self.assertTrue(all(i == 0.0 for i in self.traj_nom.flight_vy))
        self.assertTrue(all(i == 0.0 for i in self.traj_nom.flight_vz))
        
        
    def test_generate_MRU_no_nom(self):
        """
        Check if all items in x, y and z axis have the same spatial difference (for non nominal uniform trajectory)
        
        flight_vx, flight_vy, flight_vz must be constant over all trajectoryflight_vx.
        """
        # Lists of spatial differences between 2 points in trajectory.
        l_spatial_diffx = []
        l_spatial_diffy = []
        l_spatial_diffz = []
        
        # Get velocity and create Non-uniform trajectory.
        self.traj_no_nom.generate_MRU_no_nominal()
        
        # get spatial difference between two coordinates in xy,z.
        for i in range(np.size(self.traj_no_nom.flight_x)):
            if i!=self.traj_no_nom.flight_x.size - 1: # Border condition
                spatial_diffx = self.traj_no_nom.flight_x[i] - self.traj_no_nom.flight_x[i+1]
                spatial_diffy = self.traj_no_nom.flight_y[i] - self.traj_no_nom.flight_y[i+1]
                spatial_diffz = self.traj_no_nom.flight_z[i] - self.traj_no_nom.flight_z[i+1]
                l_spatial_diffx.append(spatial_diffx)
                l_spatial_diffy.append(spatial_diffy)
                l_spatial_diffz.append(spatial_diffz)
            
        # verify if every spatial difference is almost the same. So we have an equally spaced trajectory (i.e linear) over x, y and z.
        for i in range(len(l_spatial_diffx)):
            if i!=len(l_spatial_diffx) - 1: # Border condition
                self.assertAlmostEqual(l_spatial_diffx[i], l_spatial_diffx[i+1])
                self.assertAlmostEqual(l_spatial_diffy[i], l_spatial_diffy[i+1])
                self.assertAlmostEqual(l_spatial_diffz[i], l_spatial_diffz[i+1])

        # Get round values with 7 decimals to compare and compare uniformity of velocities.
        vxc = np.around(self.traj_no_nom.flight_vx[0], decimals=7) 
        vyc = np.around(self.traj_no_nom.flight_vy[0], decimals=7) 
        vzc = np.around(self.traj_no_nom.flight_vz[0], decimals=7) 

        self.assertTrue(all(i ==vxc for i in np.around(self.traj_no_nom.flight_vx, decimals=7)))
        self.assertTrue(all(i ==vyc for i in np.around(self.traj_no_nom.flight_vy, decimals=7)))
        self.assertTrue(all(i ==vzc for i in np.around(self.traj_no_nom.flight_vz, decimals=7)))

    def test_generate_MRUV_x(self):
        """
        Check if all items in flight_x have an incremental spatial difference (for non nominal non uniform velocity). Items in flight_y and flight_z must be zero.
        
        Check if all items in flight_vx, flight_vy and flight_vz have the same spatial difference (for constant increment of velocity).
        
        flight_ax must be greater than zero, flight_ax, flight_ax must be zero (constant acceleration over x axis).
        """

        # Lists of spatial differences between 2 points in trajectory.
        l_spatial_diffx = []
        l_spatial_diffy = []
        l_spatial_diffz = []

        # Lists of spatial differences between 2 points in velocity.
        l_spatial_diffvx = []
        l_spatial_diffvy = []
        l_spatial_diffvz = []

        # Check for different initial and nominal velocities to have an accelerated trajectory.
        self.assertGreater(self.traj_acc.v_nom, self.traj_acc.v_init)

        # Create Non-uniform accelerated trajectory.
        self.traj_acc.generate_MRUV_x_axis()

        # Get spatial difference between two spatial coordinates in x,y,z.
        for i in range(np.size(self.traj_acc.flight_x)):
            if i!=self.traj_acc.flight_x.size - 1: # Border condition
                spatial_diffx = self.traj_acc.flight_x[i] - self.traj_acc.flight_x[i+1]
                spatial_diffy = self.traj_acc.flight_y[i] - self.traj_acc.flight_y[i+1]
                spatial_diffz = self.traj_acc.flight_z[i] - self.traj_acc.flight_z[i+1]
                l_spatial_diffx.append(spatial_diffx)
                l_spatial_diffy.append(spatial_diffy)
                l_spatial_diffz.append(spatial_diffz)

        # Verify if every coordinate difference in x axis is increased over trajectory. Verify if every coordinate over y and z axis are equal to zero.
        for i in range(len(l_spatial_diffx)):
            if i!=len(l_spatial_diffx) - 1: # Border condition
                self.assertGreater(l_spatial_diffx[i], l_spatial_diffx[i+1])
                self.assertEqual(l_spatial_diffy[i], 0.0)
                self.assertEqual(l_spatial_diffz[i], 0.0)

        # Get spatial difference between two velocities in x,y,z.
        for i in range(np.size(self.traj_acc.flight_vx)):
            if i!=self.traj_acc.flight_vx.size - 1: # Border condition
                spatial_diffvx = self.traj_acc.flight_vx[i] - self.traj_acc.flight_vx[i+1]
                spatial_diffvy = self.traj_acc.flight_vy[i] - self.traj_acc.flight_vy[i+1]
                spatial_diffvz = self.traj_acc.flight_vz[i] - self.traj_acc.flight_vz[i+1]
                l_spatial_diffvx.append(spatial_diffvx)
                l_spatial_diffvy.append(spatial_diffvy)
                l_spatial_diffvz.append(spatial_diffvz)

        # Verify if every velocity difference is almost the same. So we have an linear velocity over x, y and z.
        for i in range(len(l_spatial_diffvx)):
            if i!=len(l_spatial_diffvx) - 1: # Border condition
                self.assertAlmostEqual(l_spatial_diffvx[i], l_spatial_diffvx[i+1]) 
                self.assertAlmostEqual(l_spatial_diffvy[i], l_spatial_diffvy[i+1]) 
                self.assertAlmostEqual(l_spatial_diffvz[i], l_spatial_diffvz[i+1]) 

        # Check for flight_ax greater than zero, flight_ax, flight_ax equals to zero (constant acceleration over x axis).
        self.assertGreater(self.traj_acc.flight_ax, 0.0)
        self.assertEquals(self.traj_acc.flight_ay, 0.0)
        self.assertEquals(self.traj_acc.flight_az, 0.0)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
