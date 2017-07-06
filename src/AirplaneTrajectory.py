'''
Created on 26 ene. 2017

@author: esufan
@summary: Module containing AirplaneTrajectory for generation of uniform and non-uniform airplane trajectories in SAR acquisition.

-Attributes: -.
-Methods: -.
-Classes: AirplaneTrajectory

'''

import numpy as np
import Utils as Utils

class AirplaneTrajectory(object):
    """
    Generate uniform and non-uniform airplane trajectory.

    Attributes
    ----------
    flight_x: numpy array
      Positions over 3D system coordinate (x axis).
    flight_y: numpy array
      Positions over 3D system coordinate (y axis).
    flight_z: numpy array
      Positions over 3D system coordinate (z axis).
      
    flight_vx: numpy array
      Velocities over 3D system coordinate (x axis).
    flight_vy: numpy array
      Velocities over 3D system coordinate (y axis).
    flight_vz: numpy array
      Velocities over 3D system coordinate (z axis).
    
    flight_ax: float
      Constant acceleration over 3D system coordinate (x axis).
    flight_ay: float
      Constant acceleration over 3D system coordinate (y axis).
    flight_az: float
      Constant acceleration over 3D system coordinate (z axis).

    flight_param_pitch: float
      Flight parameter pitch angle, used to compute next point for no nominal trajectory.
    flight_param_roll: float
      Flight parameter roll angle, used to compute next point for no nominal trajectory.
    flight_param_yaw: float
      Flight parameter yaw angle, used to compute next point for no nominal trajectory.
    
    flight_att_pitch: float
      Flight attitude pitch angle of airplane.
    flight_att_roll: float
      Flight attitude roll angle of airplane.
    flight_att_yaw: float
      Flight attitude yaw angle of airplane.
      
    nom_look: float.
      Nominal look angle. Coordinate system centered on the airplane.
    nom_squint: float.
      Nominal squint angle. Coordinate system centered on the airplane.
    z: float
      Height of airplane.

    v_init: float
      Initial velocity of airplane.
    v_nom: float
      Nominal velocity of airplane.
      
    t_axis: numpy array
      Time axis centered at 0.
    t_axis_pos: numpy array
      Positive time axis. 
    self.nt = int
      Number of elements of time axis.

    Methods
    -------
    set_parameters
    generate_MRU_nominal
    generate_MRU_no_nominal
    generate_velocity_MRU_no_nominal
    generate_MRUV_x_axis
    """

    def __init__(self, param):
        """
        Constructor

        Parameters
        ----------
        param: object (ConfigurationManager instance).
          ConfigurationManager instance to read parameters from file.
        """

        # Positions.
        self.flight_x = np.array([])
        self.flight_y = np.array([])
        self.flight_z = np.array([])

        # Velocities.
        self.flight_vx = np.array([])
        self.flight_vy = np.array([])
        self.flight_vz = np.array([])
        
        # Accelerations.
        self.flight_ax = 0.0
        self.flight_ay = 0.0
        self.flight_az = 0.0

        # Flight parameters attributes 
        self.flight_param_pitch = 0.0
        self.flight_param_roll = 0.0
        self.flight_param_yaw = 0.0

        # Flight attitude airplane attributes
        self.flight_att_pitch = 0.0
        self.flight_att_roll = 0.0
        self.flight_att_yaw = 0.0

        self.nom_look = 0.0
        self.nom_squint = 0.0
        self.z = 0.0

        # Initial and nominal velocity of airplane.
        self.v_init = 0.0
        self.v_nom = 0.0

        # Time axis centered at zero and positive time axis.
        self.t_axis = np.array([])
        self.t_axis_pos = np.array([])
        # Number of elements of time axis.
        self.nt = 0

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

        # Obtain flight parameters to compute trajectory.
        self.flight_param_pitch = (param.get_float_parameter("AirplaneTrajectory/FlightParam/pitch")*np.pi)/180.
        self.flight_param_roll = (param.get_float_parameter("AirplaneTrajectory/FlightParam/roll")*np.pi)/180.
        self.flight_param_yaw = (param.get_float_parameter("AirplaneTrajectory/FlightParam/yaw")*np.pi)/180.

        # Obtain flight attitude in trajectory.
        self.flight_att_pitch = (param.get_float_parameter("AirplaneTrajectory/FlightAttitude/pitch")*np.pi)/180.
        self.flight_att_roll = (param.get_float_parameter("AirplaneTrajectory/FlightAttitude/roll")*np.pi)/180.
        self.flight_att_yaw = (param.get_float_parameter("AirplaneTrajectory/FlightAttitude/yaw")*np.pi)/180.

        # Load look angle, squint angle and height.
        self.nom_look = (param.get_float_parameter("AirplaneTrajectory/NomAircraft/look_angle")*np.pi)/180.
        self.nom_squint = (param.get_float_parameter("AirplaneTrajectory/NomAircraft/squint_angle")*np.pi)/180.
        self.z = param.get_float_parameter("AirplaneTrajectory/NomAircraft/z")

        # Load Nominal and initial velocity.
        self.v_nom = param.get_float_parameter("AirplaneTrajectory/NomAircraft/v_nom")
        self.v_init = param.get_float_parameter("AirplaneTrajectory/NomAircraft/v_init")

        # Load look angle.
        la = (param.get_float_parameter("AirplaneTrajectory/NomAircraft/look_angle")*np.pi)/180.
        # Load r. 
        r = param.get_float_parameter("AirplaneTrajectory/NomAircraft/z")/np.cos(la)
        # Load beamwidth.
        bmw = (param.get_float_parameter("Radar/beamwidth")*np.pi)/180.
        Dx_mono =  r*(np.tan(bmw/2.))*param.get_float_parameter("AirplaneTrajectory/az_bw_sim_enl")

        # Generate time axis.
        T = (2*Dx_mono)/self.v_nom                                                   # Time the SAR to see the target.
        dt = 1./param.get_float_parameter("Radar/PRF") 
        self.t_axis_pos = np.arange(0, T, dt)                                             # Positive time axis with dt increment.
        self.nt =len(self.t_axis_pos)                                                          # Number of elements of t_axis_pos.
        self.t_axis = self.t_axis_pos - self.t_axis_pos[int(round(self.nt/2))-1]                    # time axis centered at 0.

    def generate_MRU_nominal(self):
        """
        Forces to generate Nominal linear trajectory for airplane.
        This trajectory is in straight line over the target, with constant velocity read from xml file.
        It forces attitude and flight angles to zero values, independent of values saved into xml file.
        
        Parameters
        ----------
        -.
                
        Returns
        -------
        -.
        """

        # Set linear trajectory and constant velocity "v_nom".
        self.flight_x = self.t_axis*self.v_nom
        self.flight_y = np.zeros(len(self.t_axis))
        self.flight_z = np.zeros(len(self.t_axis))

        self.flight_vx = np.full((len(self.t_axis)), self.v_nom)
        self.flight_vy = np.zeros(len(self.t_axis))
        self.flight_vz = np.zeros(len(self.t_axis))

        # Flight parameters attributes 
        self.flight_param_pitch = 0.0
        self.flight_param_roll = 0.0
        self.flight_param_yaw = 0.0

        # Flight attitude airplane attributes
        self.flight_att_pitch = 0.0
        self.flight_att_roll = 0.0
        self.flight_att_yaw = 0.0

    def generate_MRU_no_nominal(self):
        """
        Generate no nominal linear trajectory for airplane, with constant velocity (over x,y and z axis).
        
        Parameters
        ----------
        -.
        Returns
        -------
        -.
        """

        # Vector to rotate.
        self.flight_x = (self.t_axis*self.v_nom)
        self.flight_y = np.zeros(len(self.t_axis))
        self.flight_z = np.zeros(len(self.t_axis))

        # Stack vectors to rotate.
        v_3D = np.array([self.flight_x, self.flight_y, self.flight_z])

        # Rotate vector.        
        self.flight_x, self.flight_y, self.flight_z = Utils.rotate_3D_airplane(v_3D, self.flight_param_pitch, self.flight_param_roll, self.flight_param_yaw)

        # Get velocities.
        self.generate_velocity_MRU_no_nominal()

    def generate_velocity_MRU_no_nominal(self):
        """
        Generate linear velocity for airplane: the direction of trajectory vector is equal to direction of velocity vector.
        Parameters
        ----------
        -.

        Returns
        -------
        -.
        """

        # Stack position arrays and rotate to get velocities over real positions in "time" axis y.
        # vm=(xf-xi)/(tf-ti) gives mean velocity in the middle of xf and xi positions.
        # With the rotation to the left and right the velocity is calculate over an existing position xn,yn,zn from trajectory.
        flight_traj = np.array([self.flight_x, self.flight_y, self.flight_z])
        flight_traj = np.roll(flight_traj, -1, axis=1) - np.roll(flight_traj, 1, axis=1)
        
        # Get module of flight trajectories (for every xn,yn,zn).
        mod_flight_traj = np.sqrt(np.sum(abs(flight_traj)**2, axis=0))
        # Repeat over x,y,z to normalize.
        mod_flight_traj = np.tile(mod_flight_traj, [3,1])
        
        # Compute velocity vector.
        flight_vel = (flight_traj/mod_flight_traj)*self.v_nom
        
        # Reassign values in column border conditions.
        flight_vel[:,0] = flight_vel[:,1]
        flight_vel[:,-1] = flight_vel[:,-2]
        
        # Set velocities over 3D axis.
        self.flight_vx = flight_vel[0, :]
        self.flight_vy = flight_vel[1, :]
        self.flight_vz = flight_vel[2, :]

    def generate_MRUV_x_axis(self):
        """
        Generate accelerated linear trajectory for airplane over x axis.
        Linear trajectory is in straight line over the target, with non-constant velocity and constant acceleration over x axis.
        To use this function set v_init < v_nom in parameters file, to have an accelerated trajectory.
        
        Parameters
        ----------
        -.
                
        Returns
        -------
        -.
        """

        # Set linear trajectory to compute constant acceleration over x axis.
        self.flight_x = self.t_axis_pos*self.v_nom
        self.flight_y = np.zeros(len(self.t_axis_pos))
        self.flight_z = np.zeros(len(self.t_axis_pos))
        
        # Set linear velocity over x axis.
        self.flight_vx = np.full((len(self.t_axis_pos)), self.v_nom)
        self.flight_vy = np.zeros(len(self.t_axis_pos))
        self.flight_vz = np.zeros(len(self.t_axis_pos))

        # Get initial and final times, initial and final positions for airplane flight over x axis.
        ti = self.t_axis_pos[0]
        tf = self.t_axis_pos[-1]
        xi = self.flight_x[0]
        xf = self.flight_x[-1]

        # Time and velocity of flight over x axis.
        Dt = tf - ti
        vf_x = float(xf - xi)/(Dt)

        # Compute acceleration.
        self.flight_ax = 2*(vf_x - self.v_init)/Dt

        # Compute position over instant times "t_axis_pos[i]".
        for i in range(np.size(self.t_axis_pos)):
            self.flight_vx[i] = self.flight_ax*self.t_axis_pos[i] + self.v_init
            self.flight_x[i] = 0.5*self.flight_ax*((self.t_axis_pos[i])**2) + self.v_init*self.t_axis_pos[i] + xi
        
        # Set axis centered at zero.
        nt =len(self.flight_x)                                                          # Number of elements of t_axis_pos.
        self.flight_x = self.flight_x - self.flight_x[int(round(nt/2))-1]                    # time axis centered at 0.

