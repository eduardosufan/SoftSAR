'''
Created on 13 mar. 2017

@author: esufan
@summary: Module containing TDBP algorithm to focus simulated SAR image.

-Attributes: -.
-Methods: focus_wrapper.
-Classes: TDBP

'''

import numpy as np
from AirplaneTrajectory import AirplaneTrajectory
import Utils
import multiprocessing
from functools import partial
from numpy import dtype, complex128

def focus_wrapper(tdbp, list_rows):
    """
    Wrapper of TDBP.focus() method to be used in parallel multiprocessing TDBP focus.
    Wrapper needs to be at top level of module and use an instance of TDBP class to call instance method.

    Parameters
    ----------
    tdbp: object (TDBP instance).
      TDBP instance to focus image.
    list_rows: list of ints.
      List containing rows of simulated image to execute in parallel mode.

    Returns
    -------
    list of numpy complex.
      List containing entire focused row (numpy complex data) calculated in parallel mode.
    """

    return tdbp.focus_multiprocessing(list_rows)

class TDBP(object):
    """
    Focus SAR image with Time Domain Back Projection Algorithm.
    
    Attributes
    ----------

    foc_x: numpy array
      x axis (azimuth) acquisition path (in meters).
    foc_y: numpy array
      y axis (range) acquisition path (in meters).

    # Image to be focused.
    foc_X: numpy array
      x axis meshgrid of acquisition path (in meters).
    foc_Y: numpy array
      y axis meshgrid of acquisition path (in meters).
    foc_Z: numpy array
      z axis meshgrid of acquisition path (in meters).

    squint_threshold = float
      Beamwidth to be used for the focusing.

    nx: int
      Number of x axis elements of 2D image.
    ny: int
      Number of x axis elements of 2D image.
    N: int
      Number of x axis and y axis elements of 2D image (N: nx*ny).

    nominal_squint: float
      Nominal squint angle.
    nominal_look: float
      Nominal look angle.

    radar_beamwidth: float
      Radar beamwidth.
    beamwidth_threshold: float
      Radar beamwidth threshold used for limit the beamwidth used in focus.

    simulated_image: object (SimulatedImage instance).
      SimulatedImage instance with related data of range compressed image to be focused.
    
    trajectory_nom: object (AirplaneTrajectory instance).
      AirplaneTrajectory instance with nominal trajectory, with data to be used in focusing.

    local_look_ref_traj: numpy array
      Local look of reference trajectory to be used in focusing.
    local_squint_ref_traj: numpy array
      Local squint of reference trajectory to be used in focusing.
    distances_ref_traj: numpy array
      Distances between airplane and target of reference trajectory to be used in focusing.

    focused_image: numpy array
      2D array with focused SAR image.

    range_pixel_spacing: float
      Range pixel spacing (size of each pixel).

    azimuth_pixel_spacing: float
      Azimuth pixel spacing (size of each pixel).

    Methods
    -------
    set_parameters
    get_nominal_resolution
    get_reference_focus_data
    focus_multiprocessing
    """

    def __init__(self, param, simulated_image):
        '''
        Constructor

        Parameters
        ----------
        param: object (ConfigurationManager instance).
          ConfigurationManager instance to read parameters from file.
        simulated_image: object (SimulatedImage instance).
          SimulatedImage instance of simulated SAR image to be focus.
        '''

        self.foc_x = np.array([])
        self.foc_y = np.array([])

        # Image to be focused.
        self.foc_X = np.array([])
        self.foc_Y = np.array([])
        self.foc_Z = np.array([])

        # Beamwidth to be used for the focusing
        self.squint_threshold = 0.0

        # Number of elements of 2D image. N is nx*xy.
        self.nx = 0
        self.ny = 0
        self.N = 0

        # Nominal pointing.
        self.nominal_squint = 0.0
        self.nominal_look = 0.0

        # Radar beamwidth.
        self.radar_beamwidth = 0.0
        # Beamwidth threshold used for limit beamwidth of SAR antenna. DEFAULT VALUE: 2.0
        self.beamwidth_threshold = 2.0

        # Create nominal trajectory instance to be used as reference for focusing.
        self.trajectory_nom = AirplaneTrajectory(param)
        self.trajectory_nom.generate_MRU_nominal()

        # Data to be used in focusing derived from trajectory of airplane.
        self.local_look_ref_traj = np.array([])
        self.local_squint_ref_traj = np.array([])
        self.distances_ref_traj = np.array([])

        # Focused image.
        self.focused_image = np.array([])

        # Range and azimuth pixel spacing.
        self.range_pixel_spacing = 0.0
        self.azimuth_pixel_spacing = 0.0

        # Object SimulatedImage to be focused.
        self.simulated_image = simulated_image

        # Load parameters to focus.
        self.set_parameters(param)
        self.spectrum_whitening()
        
        self.param = param

    def set_parameters(self, param):
        """
        Read parameters from xml file and set the needed attributes of class.

        Parameters
        ----------
        param: object (ConfigurationManager instance).
          ConfigurationManager instance to read parameters from file.

        Returns
        -------
        -.
        """

        # Load parameters from xml.
        self.radar_beamwidth = (param.get_float_parameter("Radar/beamwidth")*np.pi)/180.
        self.beamwidth_threshold = param.get_float_parameter("Radar/beamwidth_threshold") # Warning: Use this parameter to limit beamwidth!
        # Squint threshold limited by antenna beamwidth.
        self.squint_threshold = (self.radar_beamwidth)/self.beamwidth_threshold

        self.get_nominal_resolution(param)

        # Load reference data to focus, whit the trajectory used in the image simulation as reference.
        self.get_reference_focus_data(self.simulated_image.traj)
        
    def spectrum_whitening(self):
        """
        Azimut spectrum whitening of simulated image before focusing.

        Parameters
        ----------
        -.

        Returns
        -------
        -.
        """

        # Get number of rows and columns of original image.
        [raw_rows, raw_columns] = np.shape(self.simulated_image.image)

        self.new_raw = np.empty([raw_rows, raw_columns], dtype=np.complex128)

        # Normalize spectrum of original image with mean power of every azimut.
        for i in range(raw_columns):
            range_mean_power = np.mean(abs(self.simulated_image.image[:,i])**2)
            self.new_raw[:,i]=self.simulated_image.image[:,i]/range_mean_power

    def get_nominal_resolution(self, param):
        """
        Get nominal squint and look angle from nominal trajectory as reference.

        Parameters
        ----------
        param: object (ConfigurationManager instance).
          ConfigurationManager instance to read parameters from file.

        Returns
        -------
        """

        # Load necessary parameters from xml.
        c = param.get_float_parameter("Radar/c")
        sar_f0 = param.get_float_parameter("Radar/f0")
        sar_lambda = c/sar_f0
        # SAR bandwidth.
        B = param.get_float_parameter("Radar/B")
        # Load look angle.
        look_angle = (param.get_float_parameter("AirplaneTrajectory/NomAircraft/look_angle")*np.pi)/180.

        # Get number of pixels representing nominal resolution in range and azimuth.
        n_pixels_nominal_range_resolution = param.get_float_parameter("ImageFocusing/n_pixels_nominal_range_resolution")
        n_pixels_nominal_azimuth_resolution = param.get_float_parameter("ImageFocusing/n_pixels_nominal_azimuth_resolution")

        # Get length of range and azimuth axis. Axis will be "length times" nominal resolution in each case.
        length_range_axis = param.get_float_parameter("ImageFocusing/length_range_axis")
        length_azimuth_axis = param.get_float_parameter("ImageFocusing/length_azimuth_axis")

        # Horizontal spatial wavenumbers.
        kx = np.dot(4*np.pi/sar_lambda, np.sin(self.simulated_image.abs_squint + np.pi/2))*np.cos(self.simulated_image.abs_look - np.pi/2)

        # Get nominal_squint to compute nominal azimuth resolution.
        nominal_target_look, nominal_target_squint = Utils.get_angles_antenna(self.trajectory_nom, self.simulated_image.nom_target)

        kx_3dB_ind = np.argmin(abs(nominal_target_squint - self.radar_beamwidth/2))
        kx_3dB = kx[kx_3dB_ind]

        # Nominal azimuth resolution (first zero of the sinc in space for a constant illumination of spatial frequencies from -kx to kx)
        daz_nom = np.pi/kx_3dB
        # dx is the pixel spacing in azimuth. The nominal resolution in azimuth will fit in "n_pixels_nominal_azimuth_resolution" pixels.
        dx = daz_nom/n_pixels_nominal_azimuth_resolution
        self.azimuth_pixel_spacing = dx

        # Azimuth axis.
        Dx_mono = daz_nom*length_azimuth_axis
        self.foc_x = np.arange(-Dx_mono, Dx_mono + dx, dx)
        self.nx = len(self.foc_x)
        self.foc_x = self.foc_x - self.foc_x[int(round(self.nx/2))]
        self.foc_x = self.foc_x + self.simulated_image.nom_target[0]

        #********************************************************************************************************************************
        # Nominal range resolution.
        dr_nom = c/(2*B)
        dgr_nom = dr_nom/np.sin(look_angle)
        # dy is the pixel spacing in range. The nominal resolution in range will fit in "n_pixels_nominal_range_resolution" pixels.
        dy = dgr_nom/n_pixels_nominal_range_resolution
        self.range_pixel_spacing = dy

        # Range axis.
        Dy_mono = dgr_nom*length_range_axis
        self.foc_y = np.arange(-Dy_mono, Dy_mono + dy, dy)
        self.ny = len(self.foc_y)
        self.foc_y = self.foc_y - self.foc_y[int(round(self.ny/2))]
        self.foc_y = self.foc_y + self.simulated_image.nom_target[1]

        # Number of elements of resultant image.
        self.N = self.nx*self.ny
        [self.foc_X, self.foc_Y] = np.meshgrid(self.foc_x, self.foc_y)
        self.foc_Z = self.simulated_image.nom_target[2]*np.ones((self.ny, self.nx))

        # Initialize focused image.
        self.focused_image = np.empty([self.ny, self.nx], dtype=np.complex128)

    def get_reference_focus_data(self, reference_trajectory):
        """
        Get reference data from trajectory data of corresponding simulated image to focus with.
        It's necessary in TDPB algorithm to know data from acquisition path to compensate non-linearities in trajectory.
        
        Parameters
        ----------
        -.
                
        Returns
        -------
        -.
        """

        # Reorder image.
        foc_X = self.foc_X.reshape((self.N), order="F")
        foc_Y = self.foc_Y.reshape((self.N), order="F")
        foc_Z = self.foc_Z.reshape((self.N), order="F")
        image = np.asanyarray([foc_X, foc_Y, foc_Z])

        # Get local_look and local_squint for the distributed target.
        self.local_look_ref_traj, self.local_squint_ref_traj = Utils.get_angles_antenna_distributed_target(reference_trajectory, image)

        N = image.shape[1]
        NT = len(reference_trajectory.flight_x)

        # Compute distances between sensor and target.
        self.distances_ref_traj = np.sqrt( (np.full((N, NT), reference_trajectory.flight_x[:]) - np.full((N, NT), foc_X[:, None]) )**2 + 
                                           (np.full((N, NT), reference_trajectory.flight_y[:]) - np.full((N, NT), foc_Y[:, None]) )**2 + 
                                           (np.full((N, NT), reference_trajectory.flight_z[:]) - np.full((N, NT), foc_Z[:, None]) )**2)

    # @deprecated: now using multiprocessing focus in parallel execution.
    #def focus(self):
    #    """
    #    Focus SAR image with TDBP algorithm. NOTE: Image must be range compressed.
    #    It uses local squint angle (antenna coordinate system) and distances to target to focus.
    #    
    #    Parameters
    #    ----------
    #    -.
    #            
    #    Returns
    #    -------
    #    -.
    #    """
    #
    #    # Light speed.
    #    c = 300000000.0
    #    # SAR bandwidth, central frequency and lambda.
    #    sar_B = self.param.get_float_parameter("Radar/B")
    #    sar_f0 = self.param.get_float_parameter("Radar/f0")
    #    sar_lambda = c/sar_f0
    #
    #    # Initialize focused image.
    #    self.focused_image = np.empty([self.ny, self.nx], dtype=np.complex128)
    #
    #    nt_fast_time = self.simulated_image.traj.nt
    #    nt_slow_time = self.simulated_image.Nt
    #
    #    # Azimut spectrum whitening before focus.
    #    [raw_rows, raw_columns]=np.shape(self.simulated_image.image)
    #
    #    new_raw = self.simulated_image.image
    #
    #    for i in range(raw_columns):
    #        range_mean_power = np.mean(abs(self.simulated_image.image[:,i])**2)
    #        new_raw[:,i]=self.simulated_image.image[:,i]/range_mean_power
    #
    #    #for x_foc_ind in range(self.nx):
    #        for y_foc_ind in range(self.ny):
    #            foc_lin_ind = x_foc_ind*self.ny + y_foc_ind
    #
    #            # Synthetic range compressed data (matched 2D filter).
    #            # Antenna Enclosure (lobe).
    #            #doppler_amplitude_lin = (np.sinc(self.local_squint_ref_traj[foc_lin_ind, :]/self.radar_beamwidth*(2*0.443) ))**2
    #            doppler_amplitude_lin = (np.sinc(self.local_squint_ref_traj[foc_lin_ind, :]/self.radar_beamwidth*0.886 ))**2
    #            doppler_amplitude = np.tile(doppler_amplitude_lin, [self.simulated_image.Nt, 1])
    #
    #            # Range amplitude: range positions in raw data of backscattered signal. These are the sincs with range 
    #            # migration (range compressed image).
    #            range_amplitude = np.sinc( sar_B*( (np.tile(self.simulated_image.t_axis_fast_time, [nt_fast_time, 1])).transpose()
    #                                               - np.tile(2*self.distances_ref_traj[foc_lin_ind, :]/c, [nt_slow_time, 1]) ) )
    #
    #            # Limit bandwidth to threshold given by a window. Use only 3dB of antenna lobe for azimuth, limited by squint threshold.
    #            doppler_threshold_win = np.absolute( np.tile(self.local_squint_ref_traj[foc_lin_ind, :], [nt_slow_time, 1]) ) < self.squint_threshold
    #            raw_amplitude = doppler_amplitude*range_amplitude*doppler_threshold_win
    #
    #            # Phase of backscattered signal (2*pi*2*r/lambda).
    #            raw_phase = np.exp(-1j*4*np.pi/sar_lambda*np.tile(self.distances_ref_traj[foc_lin_ind, :], [nt_slow_time, 1]))
    #
    #            # Get module of raw_amplitude (for every xn, yn).
    #            mod_raw_amplitude = np.sum(abs(raw_amplitude)**2)
    #            # Repeat over x,y (slow time and fast time) to normalize.
    #            mod_raw_amplitude = np.tile(mod_raw_amplitude, [nt_slow_time, nt_fast_time])
    #
    #            # Get raw odographer with raw_amplitude and raw_phase, i.e. with amplitude and phase information, and normalize.
    #            raw_to_foc = (np.conjugate(raw_phase))*raw_amplitude/mod_raw_amplitude
    #
    #            # Focused image: get projection on the matched filter. It convolves the raw image with the calculated odographer.
    #            #self.focused_image[y_foc_ind, x_foc_ind] = np.sum(self.simulated_image.image*raw_to_foc)
    #            self.focused_image[y_foc_ind, x_foc_ind] = np.sum(new_raw*raw_to_foc)

#************************************************************************************************************************************

    def focus_multiprocessing(self, row):
        """
        Focus SAR image with TDBP algorithm. NOTE: Image must be range compressed.
        It uses local squint angle (antenna coordinate system) and distances to target to focus.

        Parameters
        ----------
        row: int.
          image row to be focus.
        
        Returns
        -------
        list of numpy complex.
          List containing entire focused row (numpy complex data) calculated in parallel mode.
        """

        # Light speed.
        c = 300000000.0
        # SAR bandwidth, central frequency and lambda.
        sar_B = self.param.get_float_parameter("Radar/B")
        sar_f0 = self.param.get_float_parameter("Radar/f0")
        sar_lambda = c/sar_f0

        nt_fast_time = self.simulated_image.traj.nt
        nt_slow_time = self.simulated_image.Nt

        # Partial row calculated in parallel mode focusing.
        partial_row = np.empty(self.ny, dtype=np.complex128)

        x_foc_ind = row
        for y_foc_ind in range(self.ny):
            foc_lin_ind = x_foc_ind*self.ny + y_foc_ind

            # Synthetic range compressed data (matched 2D filter).
            # Antenna Enclosure (lobe).
            doppler_amplitude_lin = (np.sinc(self.local_squint_ref_traj[foc_lin_ind, :]/self.radar_beamwidth*0.886 ))**2
            doppler_amplitude = np.tile(doppler_amplitude_lin, [self.simulated_image.Nt, 1])

            # Range amplitude: range positions in raw data of backscattered signal. These are the sincs with range 
            # migration (range compressed image).
            range_amplitude = np.sinc( sar_B*( (np.tile(self.simulated_image.t_axis_fast_time, [nt_fast_time, 1])).transpose()
                                               - np.tile(2*self.distances_ref_traj[foc_lin_ind, :]/c, [nt_slow_time, 1]) ) )

            # Limit bandwidth to threshold given by a window. Use only 3dB of antenna lobe for azimuth, limited by squint threshold.
            doppler_threshold_win = np.absolute( np.tile(self.local_squint_ref_traj[foc_lin_ind, :], [nt_slow_time, 1]) ) < self.squint_threshold
            raw_amplitude = doppler_amplitude*range_amplitude*doppler_threshold_win

            # Phase of backscattered signal (2*pi*2*r/lambda).
            raw_phase = np.exp(-1j*4*np.pi/sar_lambda*np.tile(self.distances_ref_traj[foc_lin_ind, :], [nt_slow_time, 1]))

            # Get module of raw_amplitude (for every xn, yn).
            mod_raw_amplitude = np.sum(abs(raw_amplitude)**2)
            # Repeat over x,y (slow time and fast time) to normalize.
            mod_raw_amplitude = np.tile(mod_raw_amplitude, [nt_slow_time, nt_fast_time])

            # Get raw odographer with raw_amplitude and raw_phase, i.e. with amplitude and phase information, and normalize.
            raw_to_foc = (np.conjugate(raw_phase))*raw_amplitude/mod_raw_amplitude

            partial_row[y_foc_ind] = np.sum(self.new_raw*raw_to_foc)

        return list(partial_row)

    def multiprocessing_image(self, param, n_cores):
        """
        Launch parallel execution of focusing in all available cores of SoftSAR running on CPU.
        Every row of simulated image is focused independently in a Python process.
        Resultant focused row is then gathered into complete SAR fosued image.

        Parameters
        ----------
        param: object (ConfigurationManager instance).
          ConfigurationManager instance to read parameters from file.

        Returns
        -------
        sar_image: numpy array
          Complex SAR focused image.
        """

        # Start [get_cpu()] number of worker processes.
        #pool = multiprocessing.Pool()
        pool = multiprocessing.Pool(processes=n_cores)

        # Get number of rows.
        n_rows = self.nx
        list_n_rows = range(n_rows)

        # Initialize focused image.
        sar_image = np.empty([self.ny, self.nx], dtype=np.complex128) 

        # Generate partial function as a wrapper of instance method (instance method are not pickable by Pool.map).
        partial_func = partial(focus_wrapper, self)

        # Launch parallel execution.
        pool_result = pool.map(partial_func, list_n_rows)

        pool.close()
        pool.join()

        # Assign the pool_result to sar image.
        for line,result in enumerate(pool_result):
            sar_image[:,line] = np.asarray(result)

        #Assign to class attribute.
        self.focused_image = sar_image

