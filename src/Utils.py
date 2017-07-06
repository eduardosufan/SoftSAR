"""
Created on 22 feb. 2017

@author: esufan
@summary: Utils module with common resources, like plotting and rotation of axis.

-Attributes: -.
-Methods: 
  generic_plot_1D
  generic_plot_2D
  rotate_3D
  rotate_3D_airplane
  get_angles_antenna
  get_angles_antenna_distributed_target
  freq_axis
  plot_trajectory_2D
  plot_trajectory_3D
  plot_image
  save_array
  load_array
  compare_arrays
  compare_numpy_vs_matlab_data
  save_image_bin
  load_image_bin
  save_data_mat_file
  load_data_mat_file
-Classes: -.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cPickle as pickle
import scipy.io

np.set_printoptions(threshold=np.nan) # Show all elements of array.
np.set_printoptions(suppress=True)    # Suppress scientific notation.

# Not used, uncomment of use ->
#def generic_plot_1D(data, title="", xlabel="", ylabel=""):
#    """
#    Generic function to plot 1D data
#    Parameters
#    ----------
#    data: np.array
#        array generic to plot.
#    title: string
#        Title of figure.
#    xlabel: string
#        label of x axis.
#    ylabel: string
#        label of y axis.
#
#    Returns
#    -------
#    -.
#    """
#
#    plt.figure()
#    plt.plot(data)
#    plt.xlabel(xlabel)
#    plt.ylabel(ylabel)
#    plt.title(title)
#    plt.show()

# Not used, uncomment of use ->
#def generic_plot_2D(data1, data2, title="", xlabel="", ylabel=""):
#    """
#    Generic function to plot 2D data
#    Parameters
#    ----------
#    data1: np.array
#        array generic to plot (over x axis).
#    data2: np.array
#        array generic to plot (over y axis).
#    title: string
#        Title of figure.
#    xlabel: string
#        label of x axis.
#    ylabel: string
#        label of y axis.
#
#    Returns
#    -------
#    -.
#    """
#
#    plt.figure()
#    plt.plot(data1, data2)
#    plt.xlabel(xlabel)
#    plt.ylabel(ylabel)
#    plt.title(title)
#    plt.show()

def rotate_3D(u, o):
    """
    Provides rotation matrix for a generic axis "u" and angle "o".
    u vector is normalized.
    see: "Rotation matrix from axis and angle" in https://en.wikipedia.org/wiki/Rotation_matrix
            
    Parameters
    ----------
    u: np.array
        3x1 array generic axis to rotate.
    o: float
        angle to rotate in degrees.

    Returns
    -------
    np.array
        3x3 array rotation matrix.
    """

    # Normalize u vector.
    ux = float(u[0])/np.sqrt(np.sum(np.absolute(u)**2))
    uy = float(u[1])/np.sqrt(np.sum(np.absolute(u)**2))
    uz = float(u[2])/np.sqrt(np.sum(np.absolute(u)**2))
    
    cos_o = np.cos(o)
    sin_o = np.sin(o)
    
    return np.array([[cos_o + (ux**2)*(1-cos_o) , ux*uy*(1-cos_o) - uz*sin_o, ux*uz*(1-cos_o) + uy*sin_o],
                     [uy*ux*(1-cos_o) + uz*sin_o, cos_o + (uy**2)*(1-cos_o) , uy*uz*(1-cos_o) - ux*sin_o],
                     [uz*ux*(1-cos_o) - uy*sin_o, uz*uy*(1-cos_o) + ux*sin_o, cos_o + (uz**2)*(1-cos_o)]])

def rotate_3D_airplane(vector_3D, pitch=0.0, roll=0.0, yaw=0.0):
    """
    Rotate 3D axis with pitch, roll and yaw angles (rotations done in this order).

    Parameters
    ----------
    pitch: float
      pitch angle in radians to rotate. Default: 0.
    roll: float
      roll angle in radians to rotate. Default: 0.
    yaw: float
      yaw angle in radians to rotate. Default: 0.
    vector_3D: numpy array  
      array of 3xN dimensions to rotate

    Returns
    -------
    x, y ,z : numpy arrays  
      arrays of 1xlen_array dimensions after rotation
    """

    # Rotation matrix.
    RM_roll  = np.array([[1.,           0.,            0.], 
                         [0., np.cos(roll), -np.sin(roll)],
                         [0., np.sin(roll),  np.cos(roll)]])

    RM_pitch = np.array([[ np.cos(pitch), 0., np.sin(pitch)], 
                         [            0., 1.,            0.],
                         [-np.sin(pitch), 0., np.cos(pitch)]])

    RM_yaw   = np.array([[np.cos(yaw), -np.sin(yaw), 0.], 
                         [np.sin(yaw),  np.cos(yaw), 0.],
                         [         0.,           0., 1.]])

    # Obtain 3D matrix rotation.
    RM_3D = np.dot(RM_yaw, np.dot(RM_pitch, RM_roll))

    # Rotate vector.
    v_rot = np.dot(RM_3D, vector_3D)

    # Obtain and return arrays of x, y and z coordinates.
    return v_rot[0], v_rot[1], v_rot[2]

def get_angles_antenna(traj, target):
    """
    Get angles squint and look of view with respect to the antenna coordinate system.
    It assumes a standard reference system where the aircraft is directed over x direction,
    z direction is pointing upward and the y axis is defined according to the right-hand rule.
    The coordinate system is rotated around z to bring the y-axis orthogonal to v and over elevation
    in order to make z-axis and v coincide.

    Steps in coordinate system rotation:

        1) Rotate around z axis to bring the y-axis orthogonal to v.
        2) Rotate z axis to make to coincide with v vector.

        After these steps the coordinate system is lined up with the aircraft flight direction;
        Zero roll at this point that is y-axis lined up with the ground. Then it is rotated in order to account 
        for additional yaw, pitch and roll:

        3) A rotation around the transformed x-axis (now pointing almost completely downward). Accounting for yaw.
        4) A rotation around the transformed y-axis (now pointing almost completely as ground range). Accounting for pitch.
        5) A rotation around to transformed z-axis (now pointing as the aircraft, not velocity).

        After these steps the coordinate system is lined up with the aircraft
        orientation. Then is rotated in order to match the antenna principal
        axes:

        6) Introducing nominal look angle.
        7) Introducing nominal squint angle.

    Note:
        The original reference system is right-handed and centered on the aircraft. The z-axis is the elevation; the x-axis is the azimuth;
        the y-axis follows consequently (a right-looking radar is thought).
        A positive yaw angle rotates the aircraft leftwards. It results in a negative local_squint angle.
        The convention for the squint angle follows the right-hand with respect to an upward axis. It is positive for a sensor approaching a
        target and negative flying away from it (right-looking radar is assumed).

    Parameters
    ----------
    traj: object (AirplaneTrajectory instance).
      AirplaneTrajectory instance, containing position, velocities vectors and attitude angles:
        rx, ry, rz = position vectors.
        vx, vy, vz = velocity vectors.
        attitude angles = pitch, roll, yaw attitude trajectory angles.
        look_angle: float
          nominal look angle read from xml file
        squint_angle: float
          nominal squint angle read from xml file
    target: numpy array
      Target position.

    Returns
    -------
    local_squint: numpy array
      Squint of the (r - target) vector in a reference system lined up with the antenna.
    local_look: numpy array
      Elevation angle of the (r - target) vector in a reference system lined up with the antenna.
    """

    # Number of elements to iterate.
    nt = len(traj.t_axis)

    # Initialize arrays.
    local_look = np.empty(nt)
    local_squint = np.empty(nt)

    # Number of elements of target (1 if its punctual target).
    Ntarget = 1


    # Starting axes
    axis0 = np.eye(3)

    for i in range(nt):

        # Instant velocities at "i" time.
        v = np.array([traj.flight_vx[i], traj.flight_vy[i], traj.flight_vz[i]])
        v_abs = np.sqrt(np.sum(np.absolute(v)**2))
        v_theta = np.arccos(float(v[2])/v_abs)
        v_phi = np.arctan2(v[1], v[0])

        # Instant position at "i" time
        r = np.array([traj.flight_x[i], traj.flight_y[i], traj.flight_z[i]])

        # Axis rotations ->
        # y axis orthogonal to velocity vector.
        R1 = rotate_3D(axis0[:, 2], v_phi)
        axis1 = np.dot(R1, axis0)

        # z axis lined up with velocity vector. x axis is pointing to nadir.
        R2 = rotate_3D(axis1[:, 1], v_theta)
        axis2 = np.dot(R2, axis1)

        # Adjust yaw.
        yaw_axis = -axis2[:, 0]
        R3 = rotate_3D(yaw_axis, traj.flight_att_yaw)
        axis3 = np.dot(R3, axis2)

        # Adjust pitch.
        pitch_axis = -axis3[:, 1]
        R4 = rotate_3D(pitch_axis, traj.flight_att_pitch)
        axis4 = np.dot(R4, axis3)

        # Adjust roll.
        roll_axis = axis4[:, 2]
        R5 = rotate_3D(roll_axis, traj.flight_att_roll)
        axis5 = np.dot(R5, axis4)

        # Adjust with nominal look angle.
        R6 = rotate_3D(roll_axis, -traj.nom_look)
        axis6 = np.dot(R6, axis5)

        # Adjust with nominal squint angle.
        squint_axis = axis6[:, 1]
        R7 = rotate_3D(squint_axis, traj.nom_squint)
        axis7 = np.dot(R7, axis6)

        # distance between antenna an target.
        distance = np.sqrt(np.sum( np.absolute(target - r*np.ones((1, Ntarget)))**2 ))

        local_coo = np.dot(np.transpose(axis7),np.transpose(target - r*np.ones((1, Ntarget))))

        # Compute local look and squint angle. Take [0] element due is not a distributed target.
        local_look[i] = -np.arctan2(local_coo[1, :], local_coo[0, :])[0]
        local_squint[i] = np.pi/2 - np.arccos(local_coo[2, :]/distance)[0]

    return local_look, local_squint

def get_angles_antenna_distributed_target(traj, target):
    """
    Get angles squint and look of view with respect to the antenna coordinate system.
    It assumes a standard reference system where the aircraft is directed over x direction,
    z direction is pointing upward and the y axis is defined according to the right-hand rule.
    The coordinate system is rotated around z to bring the y-axis orthogonal to v and over elevation
    in order to make z-axis and v coincide.
    target must be a 2D numpy matrix, i.e. a distributed target.

    Steps in coordinate system rotation:

        1) Rotate around z axis to bring the y-axis orthogonal to v.
        2) Rotate z axis to make to coincide with v vector.

        After these steps the coordinate system is lined up with the aircraft flight direction;
        Zero roll at this point that is y-axis lined up with the ground. Then it is rotated in order to account 
        for additional yaw, pitch and roll:

        3) A rotation around the transformed x-axis (now pointing almost completely downward). Accounting for yaw.
        4) A rotation around the transformed y-axis (now pointing almost completely as ground range). Accounting for pitch.
        5) A rotation around to transformed z-axis (now pointing as the aircraft, not velocity).

        After these steps the coordinate system is lined up with the aircraft
        orientation. Then is rotated in order to match the antenna principal
        axes:

        6) Introducing nominal look angle.
        7) Introducing nominal squint angle.
        
    Note:
        The original reference system is right-handed and centered on the aircraft. The z-axis is the elevation; the x-axis is the azimuth;
        the y-axis follows consequently (a right-looking radar is thought).
        A positive yaw angle rotates the aircraft leftwards. It results in a negative local_squint angle.
        The convention for the squint angle follows the right-hand with respect to an upward axis. It is positive for a sensor approaching a
        target and negative flying away from it (right-looking radar is assumed).

    Parameters
    ----------
    traj: object (AirplaneTrajectory instance).
      AirplaneTrajectory instance, containing position, velocities vectors and attitude angles:
        rx, ry, rz = position vectors.
        vx, vy, vz = velocity vectors.
        attitude angles = pitch, roll, yaw attitude trajectory angles.
        look_angle: float
          nominal look angle read from xml file
        squint_angle: float
          nominal squint angle read from xml file
    target: numpy array 2D
      Distributed target.

    Returns
    -------
    local_squint: numpy array
      Squint of the (r - target) vector in a reference system lined up with the antenna.
    local_look: numpy array
      Elevation angle of the (r - target) vector in a reference system lined up with the antenna.
    """

    # Number of elements of target (number of columns in matrix for distributed target).
    Ntarget = target.shape[1]

    # Number of elements to iterate.
    nt = len(traj.t_axis)
    n = target.shape[1]

    # Initialize arrays.
    local_look = np.empty([n,nt])
    local_squint = np.empty([n,nt])

    # Starting axes
    axis0 = np.eye(3)

    for i in range(nt):

        # Instant velocities at "i" time.
        v = np.array([traj.flight_vx[i], traj.flight_vy[i], traj.flight_vz[i]])
        v_abs = np.sqrt(np.sum(np.absolute(v)**2))
        v_theta = np.arccos(float(v[2])/v_abs)
        v_phi = np.arctan2(v[1], v[0])

        # Instant position at "i" time
        r = np.array([traj.flight_x[i], traj.flight_y[i], traj.flight_z[i]])

        # Axis rotations ->
        # y axis orthogonal to velocity vector.
        R1 = rotate_3D(axis0[:, 2], v_phi)
        axis1 = np.dot(R1, axis0)

        # z axis lined up with velocity vector. x axis is pointing to nadir.
        R2 = rotate_3D(axis1[:, 1], v_theta)
        axis2 = np.dot(R2, axis1)

        # Adjust yaw.
        yaw_axis = -axis2[:, 0]
        R3 = rotate_3D(yaw_axis, traj.flight_att_yaw)
        axis3 = np.dot(R3, axis2)

        # Adjust pitch.
        pitch_axis = -axis3[:, 1]
        R4 = rotate_3D(pitch_axis, traj.flight_att_pitch)
        axis4 = np.dot(R4, axis3)

        # Adjust roll.
        roll_axis = axis4[:, 2]
        R5 = rotate_3D(roll_axis, traj.flight_att_roll)
        axis5 = np.dot(R5, axis4)

        # Adjust with nominal look angle.
        R6 = rotate_3D(roll_axis, -traj.nom_look)
        axis6 = np.dot(R6, axis5)

        # Adjust with nominal squint angle.
        squint_axis = axis6[:, 1]
        R7 = rotate_3D(squint_axis, traj.nom_squint)
        axis7 = np.dot(R7, axis6)

        # distance between antenna an target.
        distance = np.sqrt(np.sum( np.absolute( target - np.dot(np.expand_dims(r, axis=1), np.ones((1, Ntarget))) )**2 , axis = 0))

        #local_coo = np.dot(   np.transpose(axis7), np.transpose(  target - np.dot(np.expand_dims(r, axis=1), np.ones((1, Ntarget)) )  )   )
        local_coo = np.dot(   np.transpose(axis7),  target - np.dot(np.expand_dims(r, axis=1), np.ones((1, Ntarget)) )     )

        # Compute local look and squint angle.
        ll = -np.arctan2(local_coo[1, :], local_coo[0, :])
        ls = np.pi/2 - np.arccos(local_coo[2, :]/distance)
        local_look[:,i] = ll
        local_squint[:,i] = ls

    return local_look, local_squint

def freq_axis(dt, N, omega_flag=False, fftshift_flag=True):
    """
    Generates a frequency axis for a signal, given a sampling interval time and number of samples.

    Parameters
    ----------
    dt: float
      Time sampling interval of signal.
    N: float
      Number of samples.
    omega_flag: Bool
      If it's true frequency axis is returned as pulsation axis, if it's false frequency axis is returned in frequency unit. Default: False.
    fftshift_flag: Bool
      If it's true frequency axis is centered at zero, if it's false frequency axis is not centered at zero. Default: True.

    Returns
    -------
    f_axis : numpy arrays  
      Frequency axis.
    """

    f_axis = np.arange(0,N, 1)/float((N*dt))
    if omega_flag:
        f_axis = 2*np.pi*f_axis

    if fftshift_flag:
        f_axis = f_axis - f_axis[int(np.ceil((N - 1)/2.))]

    return f_axis

def plot_trajectory_2D(traj):
    """
    Plot airplane trajectory in 2D.

    Parameters
    ----------
    traj: object (AirplaneTrajectory instance).
      AirplaneTrajectory instance.

    Returns
    -------
    -.
    """

    font = {'weight' : 'bold',
            'size'   : 18}

    plt.rc('font', **font)
    
    plt.figure()
    plt.subplot(211)
    plt.plot(traj.flight_x, 'r--', label='x')
    plt.plot(traj.flight_y, 'g--', label='y')
    plt.plot(traj.flight_z, 'b--', label='z')
    plt.legend()
    #plt.xlabel('Samples in space')
    plt.ylabel('Position')
    plt.title('Position of airplane')

    plt.subplot(212)
    plt.plot(traj.flight_vx, 'r--', label='vx')
    plt.plot(traj.flight_vy, 'g--', label='vy')
    plt.plot(traj.flight_vz, 'b--', label='vz')
    plt.legend()
    plt.xlabel('Samples in space')
    plt.ylabel('Velocity')
    plt.title('Velocity of airplane')

    plt.show()

def plot_trajectory_3D(traj):
    """
    Plot airplane trajectory in 3D.

    Parameters
    ----------
    traj: object (AirplaneTrajectory instance).
      AirplaneTrajectory instance.

    Returns
    -------
    -.
    """

    fig = plt.figure()
    fp = Axes3D(fig)

    fp.plot(traj.flight_x, traj.flight_y, traj.flight_z, label='Airplane trajectory')
    fp.legend()
    fp.set_xlabel('x position')
    fp.set_ylabel('y position')
    fp.set_zlabel('z position')

    figv = plt.figure()
    fv = figv.gca(projection='3d')
    fv.plot(traj.flight_vx, traj.flight_vy, traj.flight_vz, label='Airplane velocity')
    fv.set_xscale('linear')
    fv.set_yscale('linear')
    fv.set_zscale('linear')

    fv.legend()
    fv.set_xlabel('x velocity')
    fv.set_ylabel('y velocity')
    fv.set_zlabel('z velocity')
    plt.show()

def plot_image(sar_img, title = "SAR Image", color_map = "seismic", y_axis = None, x_axis = None):
    """
    Plot intensity SAR image (similar to Matlab imagesc).

    Parameters
    ----------
    sar_img: numpy array with real data.
      SAR image to plot, real or imaginary parts of information.
      DEFAULT: "SAR Image"
    title: string
      Title of figure.
    color_map: string
      String containing the desired colormap. for example: "seismic", RdBu, "grey", "Spectral", "rainbow", etc.
      see colormaps examples in: http://matplotlib.org/examples/color/colormaps_reference.html
      DEFAULT: "seismic"

    Returns
    -------
    -.
    """

    # Create a new figure.
    plt.figure()

    # Set corners (axis) with extents: (Left, Right, Bottom, Top)
    if x_axis is None and y_axis is None:
        # N: rows, M: columns.
        (N, M) = sar_img.shape
        exts = (0, M, N, 0)
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        plt.imshow(np.transpose(np.abs(sar_img)**2), interpolation = 'nearest', aspect = 'auto', cmap = color_map, extent = exts)
    else:
        # Get power of SAR image.
        sar_img_pow = np.abs(sar_img)**2
        # Get SAR image in dB.
        sar_img_dB = 10*np.log10(sar_img_pow)
        
        # Normalize image at 0dB obtaining maximum value of image.
        row_max, column_max = np.where(sar_img_dB == sar_img_dB.max())
        K_dB = sar_img_dB[row_max[0], column_max[0]]
        sar_img_dB = sar_img_dB - K_dB

        left = min(y_axis)
        right = max(y_axis)
        bottom = min(x_axis)
        top = max(x_axis)
        exts = (left, right, bottom, top)
        plt.xlabel('Range (y)')
        plt.ylabel('Azimuth (x)')
        # Plot figure.
        plt.imshow(np.transpose(sar_img_dB), interpolation = 'nearest', aspect = 'auto', cmap = color_map, extent = exts)

    # Set title, labels, colorbar and show image.
    plt.title(title)
    plt.colorbar()
    plt.show()

def measure_image_resolution_octave(sar_img, range_pos, azimut_pos, RCS, OSF, paz, prg, incang):
    """
    Measure image range and azimuth resolution calling octave function get_resolution_octave.m 
    coded by Tomas Zajc (tzajc@conae.gov.ar), from CONAE Argentina.
    
    IMPORTANT: Parameters and returning values of get_resolution_octave.m are: 

    get_resolution_octave.m parameters:
    slc_sar: complex - SAR image of NxM, where N is the number of samples in azimuth and M is the number of samples in range.
    sample: float - pixel position of target in range.
    line: float - pixel position of target in azimuth.
    RCS: float - Radar Cross Section of target (in dB, typical value: 38 for CONAE corner reflectors). See http://www.rfcafe.com/references/electrical/ew-radar-handbook/images/imgp76.gif
    OSF: float - OverSamplig Factor. (typical value: 16) 
    paz: float - pixel size in azimuth. Theoretically is v/PRF, where v is the platform velocity and PRF is the Pulse Repetition Frequency.
    prg: float - pixel size in range. Theoretically is c/(2*rsf), where c is speed of light and rsf is the range resampling frequency.
    incang: float - incidence angle of target.

    get_resolution_octave.m returns:
    resolution: array containing:
                rgr: float - range resolution (-3dB).
                azr: float - azimuth resolution (-3dB).
                K: float - absolute calibration constant. Calibrate image with 10*log10(abs(slc).^2) - K.

    Parameters
    ----------
    sar_img: numpy array.
      SAR image to get resolution of MxN, where M is the number of samples in range and N is the number of samples in azimuth.
    range_pos: float.
      pixel position of target in range.
    azimut_pos: float.
      pixel position of target in azimuth.
    RCS: float.
      Radar Cross Section of target (in dB, typical value: 38 for CONAE corner reflectors). See http://www.rfcafe.com/references/electrical/ew-radar-handbook/images/imgp76.gif
    OSF: float.
      OverSamplig Factor. (typical value: 16) 
    paz: float.
      pixel size in azimuth. Theoretically is v/PRF, where v is the platform velocity and PRF is the Pulse Repetition Frequency.
    prg: float.
      pixel size in range. Theoretically is c/(2*rsf), where c is speed of light and rsf is the range resampling frequency.
    incang: float.
      incidence angle of target.

    Returns
    -------
    range_resolution: float
      Range resolution (-3dB).
    azimuth_resolution: float
      Azimuth resolution (-3dB).
    K: float
      Absolute constant calibration.
    """

    try:
        from oct2py import octave
    except:
        print "Please install oct2py library to use this function (https://pypi.python.org/pypi/oct2py)"
        raise

    # Get resolution from octave code. Using oct2py wrapper.
    resolution = octave.get_resolution_octave(np.transpose(sar_img), range_pos, azimut_pos, RCS, OSF, paz, prg, incang)

    range_resolution = resolution[0][0]
    azimuth_resolution = resolution[0][1]
    K = resolution[0][2]

    return range_resolution, azimuth_resolution, K

def save_array(data, filename = "numpy_array"):
    """
    Generic function to save numpy array (text mode with complex data) in file.

    Parameters
    ----------
    data: numpy array.
      Data to be saved into file.
    filename: string
      File name.

    Returns
    -------
    -.
    """

    np.savetxt(filename, data, fmt='%.10f%+.10fj')

def load_array(filename = "numpy_array.txt"):
    """
    Generic function to load numpy array from file as complex data. Data has to be saved over different rows in file.

    Parameters
    ----------
    filename: string
      File name with array in txt format.

    Returns
    data: numpy array.
      Data loaded from file.
    -------
    -.
    """

    return np.loadtxt(filename, delimiter="\n", dtype=np.complex128)

def compare_arrays(array1, array2, decimal_tolerance = 7, plot_key_errors = False):
    """
    Compare two arrays and return indexes and values with differences.
    Arrays to compare must have the same length.

    Parameters
    ----------
    array1: numpy array
      Numpy array to compare (real and complex data).
    array2: numpy array
      Numpy array to compare (real and complex data).
    decimal_tolerance: int
      Number of digits of decimal part to be compared.
      DEFAULT: 7.
    plot_results: boolean
      If true: plot results over generic 1D plot. If false, plot is not done.
      DEFAULT: False.

    Returns
    -------
    diff_dict: dict.
      dictionary containing as keys the indexes with errors, and as a values the values with errors.
    """

    # Dictionary with encountered differences.
    diff_dict = {}

    # Iterate over arrays and find differences.
    for i in range(np.size(array1)):
        # If there is an error, save the values.
        if np.around(array1[i]-array2[i], decimal_tolerance) != 0+0j:
            diff_dict[i] = [array1[i], array2[i]]

    # Not used, uncomment of use ->
    #if plot_key_errors:
    #    plt.stem(np.arange(len(diff_dict.keys())), diff_dict.keys())
    #    plt.show()

    return diff_dict

def compare_numpy_vs_matlab_data(path_python_data, path_matlab_data, decimal_tolerance = 7, plot_key_errors = False):
    """
    Compare numpy and matlab array stored ond disk.

    Parameters
    ----------
    path_python_data: string
      Path to array stored in disk to compare.
    path_matlab_data: string
      Path to array stored in disk to compare.
    decimal_tolerance: int
      Number of digits of decimal part to be compared.
      DEFAULT: 7.
    plot_results: boolean
      If true: plot results over generic 1D plot. If false, plot is not done.
      DEFAULT: False.

    Returns
    -------
    diff_dict: dict.
      dictionary containing as keys the indexes with errors, and as a values the values with errors.
    """

    # Load arrays from disk.
    array1 = load_array(filename=path_python_data)
    array2 = load_array(filename=path_matlab_data)

    # Use function to compare.
    return compare_arrays(array1, array2, decimal_tolerance, plot_key_errors)

def save_image_bin(image, pathfile):
    """
    Save image file in disk as pickle format.

    Parameters
    ----------
    image: numpy array
        Matrix numpy array with complex data to be saved.
    pathfile: string
      File path to be saved.

    Returns
    -------
    -.
    """

    with open(pathfile, "w") as fd:
        pickle.dump(image, fd, protocol=1)

def load_image_bin(pathfile):
    """
    Load image file in pickle format from disk.

    Parameters
    ----------
    pathfile: string
      File path to load.

    Returns
    -------
    image: numpy array
        Matrix numpy array with complex data loaded.
    """

    with open(pathfile, "r") as fd:
        image = pickle.load(fd)

    return image

def save_data_mat_file(data, pathfile):
    """
    Save data as .mat file file in disk.

    Parameters
    ----------
    image: numpy array
        Matrix numpy array data to be saved.
    pathfile: string
      File path to be saved.

    Returns
    -------
    -.
    """

    scipy.io.savemat(pathfile, mdict={'data': data})

# Not used, uncomment of use ->
#def load_data_mat_file(pathfile):
#    """
#    Load data from file .mat format from disk. Be careful with structrure format loaded in Matlab.
#
#    Parameters
#    ----------
#    pathfile: string
#      File path to load.
#
#    Returns
#    -------
#    data: numpy array
#      Data loaded from file.
#    """
#
#    data = scipy.io.loadmat(pathfile)
#
#    return data

