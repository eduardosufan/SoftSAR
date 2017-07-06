'''
Created on 14 feb. 2017

@author: esufan
@summary: Module containing SimulatedImage for SAR image simulation.

-Attributes: -.
-Methods: -.
-Classes: SimulatedImage

'''
import numpy as np
import Utils
from Image import Image

class SimulatedImage(Image):
    """
    Simulate SAR image to focus.
    
    Attributes
    ----------
    nom_target: numpy array.
      Target nominal position of airplane.

    distances: numpy array.
      Distances between sensor and target over trajectory.
    abs_look: numpy array.
      Look angle between sensor and target over trajectory.
    abs_squint: numpy array.
      Squint angle between sensor and target over trajectory.

    local_look: numpy array.
      Angles of r - r_target in the antenna coordinate system.
    local_squint: numpy array.
      Angles of r - r_target in the antenna coordinate system
    
    t_axis_fast_time: numpy array.
      Fast time axis.
    self.Nt: int
      Number of elements in fast time axis.
    freq_axis_fftshift: numpy array.
      Frequency axis.
      
    fast_time_pixel_margin_mono: float.
      Pixel margin in the fast time.
    
    radar_dt: float.
      Fast time sampling interval.
    
    image: numpy array.
      Range compressed generated image.

    traj: object (AirplaneTrajectory instance).
      AirplaneTrajectory instance.

    Methods
    -------
    set_parameters
    generate_img
    """

    def __init__(self, param, traj):
        """
        Constructor

        Parameters
        ----------
        param: object (ConfigurationManager instance).
          ConfigurationManager instance to read parameters from file.
        traj: object (AirplaneTrajectory instance).
          AirplaneTrajectory instance of simulated trajectory to simulate a SAR acquisition.
        """

        # Target nominal position and angles. Coordinate system centered on the airplane.
        self.nom_target = np.array([])

        # Distances and angles between sensor and target over trajectory.
        self.distances = np.array([])
        self.abs_look = np.array([])
        self.abs_squint = np.array([])

        # Angles of r - r_target in the antenna coordinate system.
        self.local_look = np.array([])
        self.local_squint = np.array([])

        # Fast time axis.
        self.t_axis_fast_time = np.array([])
        # Number of elements in fast time axis.
        self.Nt = 0
        # Frequency axis.
        self.freq_axis_fftshift = np.array([])

        # Pixel margin in the fast time.
        self.fast_time_pixel_margin_mono = 0.0
        
        # Fast time sampling interval.
        self.radar_dt = 0.0

        # Generated image.
        self.image = np.array([])

        # Object traj with generated airplane trajectory.
        self.traj = traj

        # Load parameters to create trajectories.
        self.set_parameters(param)

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

        # Load pixel margin in the fast time.
        self.fast_time_pixel_margin_mono = param.get_float_parameter("ImageSimulation/Fast_time_pixel_margin_mono")

        # Load fast time sampling interval.
        radar_fs = param.get_float_parameter("Radar/fs")
        self.radar_dt = 1/radar_fs
        # Load ground range.
        gr = self.traj.z*np.tan(self.traj.nom_look)

        # Get target nominal position.
        self.nom_target = np.array([0, -gr, -self.traj.z])

        # Compute distances between sensor and target.
        self.distances = np.sqrt( (self.traj.flight_x - self.nom_target[0])**2 + (self.traj.flight_y - self.nom_target[1])**2 + (self.traj.flight_z - self.nom_target[2])**2)

        # Compute absolute look and squint.
        self.abs_look = np.arccos((self.nom_target[2] - self.traj.flight_z)/self.distances)
        self.abs_squint = np.arctan2(self.nom_target[1] - self.traj.flight_y, self.nom_target[0] - self.traj.flight_x)

    def generate_img(self, param):
        """
        Generate range compressed image.

        Parameters
        ----------
        param: object (ConfigurationManager instance).
          ConfigurationManager instance to read parameters from file.

        Returns
        -------
        -.
        """

        # Light speed.
        c = 300000000.0
        
        # Load beamwidth, bandwidth and central frequency to use locally.
        sar_bmw = (param.get_float_parameter("Radar/beamwidth")*np.pi)/180.
        sar_B = param.get_float_parameter("Radar/B")
        sar_f0 = param.get_float_parameter("Radar/f0")

        # Get angles squint and look of view with respect to the antenna coordinate system.
        #self.get_angles_antenna()
        self.local_look, self.local_squint = Utils.get_angles_antenna(self.traj, self.nom_target)

        # Set fast time axis.
        start = 2*(min(self.distances))/c - self.fast_time_pixel_margin_mono*self.radar_dt
        end = 2*(max(self.distances))/c + self.fast_time_pixel_margin_mono*self.radar_dt
        step = self.radar_dt
        self.t_axis_fast_time = np.arange(start, end, step)

        # Number of elements in fast time axis.
        self.Nt = np.size(self.t_axis_fast_time)

        self.freq_axis_fftshift = Utils.freq_axis(self.radar_dt, self.Nt, False, True)

        sar_lambda = c/sar_f0

        # Doppler amplitude (envolvente de la antena).
        doppler_amplitude = (np.sinc( (np.tile(self.local_squint, [self.Nt, 1]))/sar_bmw*(2*0.443) ))**2

        # Range amplitude: range positions in raw data of backscattered signal.
        Nd = np.size(self.distances)
        range_amplitude = np.sinc( sar_B*( (np.tile(self.t_axis_fast_time, [Nd, 1])).transpose() - np.tile(2*self.distances/c, [self.Nt, 1]) ) )

        # Signal phase received: 2*pi*2*r/lambda.
        signal_phase = np.exp(-1j*4*np.pi/sar_lambda*np.tile(self.distances, [self.Nt, 1]))

        # Generate range compressed simulated image.
        self.image = doppler_amplitude*range_amplitude*signal_phase

