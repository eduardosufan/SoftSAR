'''
Created on 10 mar. 2017

@author: esufan
@summary: unittest Utils.py module.

-Attributes: -.
-Methods: -.
-Classes: Test_Utils

'''
import unittest
import numpy as np
import sys
sys.path.insert(0, '../src/')
import Utils as Utils
import os


class Test_Utils(unittest.TestCase):
    """
    Class for testing Utils module.
    
    Methods
    -------
    test_compare_arrays
    test_compare_numpy_vs_matlab_data
    """

    def test_compare_arrays(self):
        """
        Create arrays with arbitrary errors to test function.
        """

        # List containing indexes of errors in arrays.
        error_index = [0, 5, 7]

        # Arrays with errors in "error_index" indexes.
        array1 =np.array([1.45123222234, 2.0, 5.455, 1 + 1j, -0.872322212 + 0.98877741338j, 12399123, -1.0123213021312 - 1.12312312312j, -1.0123213021311 - 1.12312312311j])
        array2 =np.array([1.45123244444, 2.0, 5.455, 1 + 1j, -0.872322212 + 0.98877741338j, 12359123, -1.0123213021312 - 1.12312312312j, -1.0123456710318 - 1.12345672311j])

        r_dict = Utils.compare_arrays(array1, array2, 7, False)

        # Check if indexes errors are the same in returned dictionary.
        for i,j in enumerate(r_dict.keys()):
            self.assertEqual(error_index[i], j)

    def test_compare_numpy_vs_matlab_data(self):
        """
        Create arrays with arbitrary errors to test function.
        """

        # Paths with arrays.
        path_matlab_array = "./matlab_data.txt"
        path_numpy_array  = "./python_data.txt"

        # List containing indexes of errors in arrays.
        error_index = [0, 5, 7]

        # Matlab array to save with errors.
        matlab_to_save = np.array([0.0001760855-0.0000927256j,
                                   -0.0002311308+0.0000122892j,
                                   0.0002447444+0.0000968709j,
                                   -0.0002038497-0.0002121365j,
                                   0.0001077393+0.0003059170j,
                                   0.0000306011-0.0003521998j,
                                   -0.0001864669+0.0003330580j,
                                   0.0003283114-0.0002435598j,
                                   -0.0004244564+0.0000938909j,
                                   0.0004501374+0.0000918577j])

        # Python array to save with errors.
        python_to_save = np.array([0.1111760855-0.2132927256j,
                                  -0.0002311308+0.0000122892j,
                                  0.0002447444+0.0000968709j,
                                  -0.0002038497-0.0002121365j,
                                  0.0001077393+0.0003059170j,
                                  0.0000306011-0.12323521998j,
                                  -0.0001864669+0.0003330580j,
                                  0.1233283114-0.0002435598j,
                                  -0.0004244564+0.0000938909j,
                                  0.0004501374+0.0000918577j])

        # Save arrays and call method.
        Utils.save_array(matlab_to_save, path_matlab_array)
        Utils.save_array(python_to_save, path_numpy_array)
        r_dict = Utils.compare_numpy_vs_matlab_data(path_numpy_array, path_matlab_array)

        # Check if indexes errors are the same in returned dictionary.
        for i,j in enumerate(r_dict.keys()):
            self.assertEqual(error_index[i], j)

        # Remove files.
        os.remove("./matlab_data.txt")
        os.remove("./python_data.txt")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
