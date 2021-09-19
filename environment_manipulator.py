"""
This script provides the environment for a free flying spacecraft with a
three-link manipulator.

The spacecraft & manipulator are tasked with simultaneously docking to and detumbling a piece of debris
Originally this was going to be completed in two tasks, but after reading

Virgili-Llop and Romano 2019, Simultaneous Capture and Detumble of a Resident Space Object by a Free-flying Spacecraft-manipulator System

a simultaneous approach was chosen.

The policy is trained in a DYNAMICS environment (in contrast to my previous work) for a number of reasons:
    1) Training in kinematics assumes that each state has no influence on any other state. The perfect controller
       is assumed to handle everything. This is fine for planar motion where there is no coupling between
       the states, but it does not apply to complex scenarios where there is coupling between the states.
       For example, moving the shoulder joint will affect the motion of the spacecraft--a kinematics 
       environment would not capture that.
    2) By training in a dynamics environment, the policy will overfit the simulated dynamics.
       For that reason, I'll try to make them as accurate as possible. However, I will still be commanding 
       acceleration signals which an on-board controller is responsible for tracking. So, small
       changes should be tolerated so long as they do not spoil the learned logic.
    3) I'll have to use a controller in simulation which will also become overfit. However,
       overfitting to a real controller is probably better than overfitting to an ideal controller
       like we were doing before. Plus, we know what sort of controller we actually have in the lab.
    4) These changes are needed to solve the dynamic coupling problem present in most complex scenarios.

All dynamic environments I create will have a standardized architecture. The
reason for this is I have one learning algorithm and many environments. All
environments are responsible for:
    - dynamics propagation (via the step method)
    - initial conditions   (via the reset method)
    - reporting environment properties (defined in __init__)
    - animating the motion (via the render method):
        - Rendering is done all in one shot by passing the completed TOTAL_STATEs
          from an episode to the render() method.

Outputs:
    Reward must be of shape ()
    State must be of shape (OBSERVATION_SIZE,)
    Done must be a bool

Inputs:
    Action input is of shape (ACTION_SIZE,)

Communication with agent:
    The agent communicates to the environment through two queues:
        agent_to_env: the agent passes actions or reset signals to the environment (this script)
        env_to_agent: the environment (this script) returns information to the agent

Reward system:
        - Zero reward at nearly all timesteps except when docking is achieved
        - A mid-way reward for when the end-effector comes within a set distance from the docking port, to help guide the learning slightly.
        - A large reward when docking occurs. The episode also terminates when docking occurs
        - A variety of penalties to help with docking, such as:
            - penalty for end-effector angle (so it goes into the docking cone properly)
            - penalty for relative velocity during the docking (so the end-effector doesn't jab the docking cone)
            - penalty for relatlive angular velocity of the end-effector during docking
            - penalty for residual angular momentum of the combined system after docking to pursuade simultaneous debumbling.
        - A penalty for colliding with the target. Instead of assigning a penalty, the episode simply ends. A penalty alone caused
          the chaser to accept penalties in pursuit of docking, and a penalty + ending made the chaser avoid the target all together.
        
        Once learned, some optional rewards can be applied to see how it affects the motion:
            - In the future, a penalty for attitude disturbance on the chaser base attitude??? 
            - A penalty to all accelerations??
            - Extend the forbidden area into a larger cone to force the approach to be more cone-shaped??

State clarity:
    - Note: TOTAL_STATE contains all relevant information describing the problem, and all the information needed to animate the motion
        = TOTAL_STATE is returned from the environment to the agent.
        = A subset of the TOTAL_STATE, called the 'observation', is passed to the policy network to calculate acitons. This takes place in the agent
        = The TOTAL_STATE is passed to the animator below to animate the motion.
        = The chaser and target state are contained in the environment. They are packaged up via self.make_total_state() before being returned to the agent.
        = The total state information returned must be as commented beside self.TOTAL_STATE_SIZE.
        
        
Started December 2, 2020
@author: Kirk Hovell (khovell@gmail.com)
"""
import numpy as np
import os
import signal
import multiprocessing
import queue
from scipy.integrate import odeint # Numerical integrator

#import code # for debugging
#code.interact(local=dict(globals(), **locals())) # Ctrl+D or Ctrl+Z to continue execution

import shutil

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

from shapely.geometry import Point, Polygon # for collision detection

class Environment:

    def __init__(self):
        ##################################
        ##### Environment Properties #####
        ##################################
        """
        ==The State==
        Some absolute states and some relative states. Some manipulator states in the inertial frame and some in the body frame. In the end,
        a subset of these are used in the actual observation. Many redundant elements were used for debugging.
        The positions are in inertial frame, unless noted otherwise, but the manipulator angles are in the joint frame.
        
        """
        self.ON_CEDAR                 = False # False for Graham, Béluga, Niagara, and RCDC
        self.ACTIONS_IN_INERTIAL      = True # Are actions being calculated in the inertial frame or body frame?
        self.TOTAL_STATE_SIZE         = 29 # [chaser_x, chaser_y, chaser_theta, chaser_x_dot, chaser_y_dot, chaser_theta_dot, shoulder_theta, elbow_theta, wrist_theta, shoulder_theta_dot, elbow_theta_dot, wrist_theta_dot, target_x, target_y, target_theta, target_x_dot, target_y_dot, target_theta_dot, ee_x, ee_y, ee_x_dot, ee_y_dot, relative_x_b, relative_y_b, relative_theta, ee_x_b, ee_y_b, ee_x_dot_b, ee_y_dot_b]
        ### Note: TOTAL_STATE contains all relevant information describing the problem, and all the information needed to animate the motion
        #         TOTAL_STATE is returned from the environment to the agent.
        #         A subset of the TOTAL_STATE, called the 'observation', is passed to the policy network to calculate acitons. This takes place in the agent
        #         The TOTAL_STATE is passed to the animator below to animate the motion.
        #         The chaser and target state are contained in the environment. They are packaged up before being returned to the agent.
        #         The total state information returned must be as commented beside self.TOTAL_STATE_SIZE.
        #self.IRRELEVANT_STATES                = [15,16,18,19,20,21] # [target_velocity & end-effector states] indices of states who are irrelevant to the policy network
        #self.IRRELEVANT_STATES                = [0,1,12,13,14,15,16,18,19,20,21,25,26,27,28] # [relative position and chaser info] indices of states who are irrelevant to the policy network
        self.IRRELEVANT_STATES                = [12,13,14,15,16,18,19,20,21,25,26,27,28] # [relative position and chaser info + chaser x&y for table falling] indices of states who are irrelevant to the policy network
        #self.IRRELEVANT_STATES                = [0,1, 6, 7, 9,10,12,13,14,15,16,18,19,20,21] # [ee_b_wristangle_relative_pos_body_accels] indices of states who are irrelevant to the policy network
        self.OBSERVATION_SIZE                 = self.TOTAL_STATE_SIZE - len(self.IRRELEVANT_STATES) # the size of the observation input to the policy
        self.ACTION_SIZE                      = 6 # [x_dot_dot, y_dot_dot, theta_dot_dot, shoulder_theta_dot_dot, elbow_theta_dot_dot, wrist_theta_dot_dot] in the inertial frame for x, y, theta; in the joint frame for the others.
        self.MAX_X_POSITION                   = 3.5 # [m]
        self.MAX_Y_POSITION                   = 2.4 # [m]
        self.MAX_VELOCITY                     = 0.1 # [m/s]
        self.MAX_BODY_ANGULAR_VELOCITY        = 15*np.pi/180 # [rad/s] for body
        self.MAX_ARM_ANGULAR_VELOCITY         = 30*np.pi/180 # [rad/s] for joints
        self.MAX_LINEAR_ACCELERATION          = 0.02#0.015 # [m/s^2]
        self.MAX_ANGULAR_ACCELERATION         = 0.05#0.04 # [rad/s^2]
        self.MAX_ARM_ANGULAR_ACCELERATION     = 0.1 # [rad/s^2]
        self.MAX_THRUST                       = 0.5 # [N] Experimental limitation
        self.MAX_BODY_TORQUE                  = 0.064 # [Nm] # Experimental limitation
        self.MAX_JOINT1n2_TORQUE              = 0.02 # [Nm] # Limited by the simulator NOT EXPERIMENT
        self.MAX_JOINT3_TORQUE                = 0.0002 # [Nm] Limited by the simulator NOT EXPERIMENT
        self.LOWER_ACTION_BOUND               = np.array([-self.MAX_LINEAR_ACCELERATION, -self.MAX_LINEAR_ACCELERATION, -self.MAX_ANGULAR_ACCELERATION, -self.MAX_ARM_ANGULAR_ACCELERATION, -self.MAX_ARM_ANGULAR_ACCELERATION, -self.MAX_ARM_ANGULAR_ACCELERATION]) # [m/s^2, m/s^2, rad/s^2, rad/s^2, rad/s^2, rad/s^2]
        self.UPPER_ACTION_BOUND               = np.array([ self.MAX_LINEAR_ACCELERATION,  self.MAX_LINEAR_ACCELERATION,  self.MAX_ANGULAR_ACCELERATION,  self.MAX_ARM_ANGULAR_ACCELERATION,  self.MAX_ARM_ANGULAR_ACCELERATION,  self.MAX_ARM_ANGULAR_ACCELERATION]) # [m/s^2, m/s^2, rad/s^2, rad/s^2, rad/s^2, rad/s^2]
                
        self.LOWER_STATE_BOUND                = np.array([ 0.0, 0.0, 0.0, -self.MAX_VELOCITY, -self.MAX_VELOCITY, -self.MAX_BODY_ANGULAR_VELOCITY,  # Chaser 
                                                          -np.pi/2, -np.pi/2, -np.pi/2, # Shoulder_theta, Elbow_theta, Wrist_theta
                                                          -self.MAX_ARM_ANGULAR_VELOCITY, -self.MAX_ARM_ANGULAR_VELOCITY, -self.MAX_ARM_ANGULAR_VELOCITY, # Shoulder_theta_dot, Elbow_theta_dot, Wrist_theta_dot
                                                          0.0, 0.0, 0.0, -self.MAX_VELOCITY, -self.MAX_VELOCITY, -self.MAX_BODY_ANGULAR_VELOCITY, # Target
                                                          0.0, 0.0, -3*self.MAX_VELOCITY, -3*self.MAX_VELOCITY, # End-effector
                                                          -self.MAX_X_POSITION, -self.MAX_Y_POSITION, 0, #relative_x_i, relative_y_i, relative_theta,
                                                          -0.688, -0.8, -0.2, -0.2]) #ee_x_b, ee_y_b, ee_x_dot_b, ee_y_dot_b
                                                          # [m, m, rad, m/s, m/s, rad/s, rad, rad, rad, rad/s, rad/s, rad/s, m, m, rad, m/s, m/s, rad/s, m, m, m/s, m/s, m, m, rad, m, m, m/s, m/s] // lower bound for each element of TOTAL_STATE
        self.UPPER_STATE_BOUND                = np.array([ self.MAX_X_POSITION, self.MAX_Y_POSITION, 2*np.pi, self.MAX_VELOCITY, self.MAX_VELOCITY, self.MAX_BODY_ANGULAR_VELOCITY,  # Chaser 
                                                          np.pi/2, np.pi/2, np.pi/2, # Shoulder_theta, Elbow_theta, Wrist_theta
                                                          self.MAX_ARM_ANGULAR_VELOCITY, self.MAX_ARM_ANGULAR_VELOCITY, self.MAX_ARM_ANGULAR_VELOCITY, # Shoulder_theta_dot, Elbow_theta_dot, Wrist_theta_dot
                                                          self.MAX_X_POSITION, self.MAX_Y_POSITION, 2*np.pi, self.MAX_VELOCITY, self.MAX_VELOCITY, self.MAX_BODY_ANGULAR_VELOCITY, # Target
                                                          self.MAX_X_POSITION, self.MAX_Y_POSITION, 3*self.MAX_VELOCITY, 3*self.MAX_VELOCITY, # End-effector
                                                          self.MAX_X_POSITION, self.MAX_Y_POSITION, 2*np.pi, #relative_x_i, relative_y_i, relative_theta,
                                                          0.688, 0.8, 0.2, 0.2]) #ee_x_b, ee_y_b, ee_x_dot_b, ee_y_dot_b
                                                          # [m, m, rad, m/s, m/s, rad/s, rad, rad, rad, rad/s, rad/s, rad/s, m, m, rad, m/s, m/s, rad/s, m, m, m/s, m/s, m, m, rad, m, m, m/s, m/s] // Upper bound for each element of TOTAL_STATE
        #self.INITIAL_CHASER_POSITION          = np.array([self.MAX_X_POSITION/2, self.MAX_Y_POSITION/2, 0.0]) # [m, m, rad]
        self.INITIAL_CHASER_POSITION          = np.array([self.MAX_X_POSITION/3, self.MAX_Y_POSITION/2, 0.0]) # [m, m, rad]
        self.INITIAL_CHASER_VELOCITY          = np.array([0.0,  0.0, 0.0]) # [m/s, m/s, rad/s]
        self.INITIAL_ARM_ANGLES               = np.array([0.0,  0.0, 0.0]) # [rad, rad, rad]
        self.INITIAL_ARM_RATES                = np.array([0.0,  0.0, 0.0]) # [rad/s, rad/s, rad/s]
        #self.INITIAL_TARGET_POSITION          = np.array([self.MAX_X_POSITION/2, self.MAX_Y_POSITION/2, 0.0]) # [m, m, rad]
        self.INITIAL_TARGET_POSITION          = np.array([self.MAX_X_POSITION*2/3, self.MAX_Y_POSITION/2, 0.0]) # [m, m, rad]
        self.INITIAL_TARGET_VELOCITY          = np.array([0.0,  0.0, 0.0]) # [m/s, m/s, rad/s]
        self.NORMALIZE_STATE                  = True # Normalize state on each timestep to avoid vanishing gradients
        self.RANDOMIZE_INITIAL_CONDITIONS     = True # whether or not to randomize the initial conditions
        self.RANDOMIZE_DOMAIN                 = False # whether or not to randomize the physical parameters (length, mass, size)
        self.RANDOMIZATION_LENGTH_X           = 0.05#3.5/2-0.3 # [m] half-range uniform randomization X position
        self.RANDOMIZATION_LENGTH_Y           = 0.05#2.4/2-0.3 # [m] half-range uniform randomization Y position
        self.RANDOMIZATION_CHASER_VELOCITY    = 0.0 # [m/s] half-range uniform randomization chaser velocity
        self.RANDOMIZATION_CHASER_OMEGA       = 0.0 # [rad/s] half-range uniform randomization chaser omega
        self.RANDOMIZATION_ANGLE              = np.pi # [rad] half-range uniform randomization chaser and target base angle
        self.RANDOMIZATION_ARM_ANGLE          = np.pi/2 # [rad] half-range uniform randomization arm angle
        self.RANDOMIZATION_ARM_RATES          = 0.0 # [rad/s] half-range uniform randomization arm rates
        self.RANDOMIZATION_TARGET_VELOCITY    = 0.0 # [m/s] half-range uniform randomization target velocity
        self.RANDOMIZATION_TARGET_OMEGA       = 10*np.pi/180 # [rad/s] half-range uniform randomization target omega
        self.MIN_V                            = -100.
        self.MAX_V                            =  125.
        self.N_STEP_RETURN                    =   5
        self.DISCOUNT_FACTOR                  = 0.95**(1/self.N_STEP_RETURN)
        self.TIMESTEP                         = 0.2 # [s]
        self.CALIBRATE_TIMESTEP               = False # Forces a predetermined action and prints more information to the screen. Useful in calculating gains and torque limits
        self.CLIP_DURING_CALIBRATION          = True # Whether or not to clip the control forces during calibration
        self.PREDETERMINED_ACTION             = np.array([0.01,-0.015,0.03,-0.07,0.01,0.1])
        self.DYNAMICS_DELAY                   = 0 # [timesteps of delay] how many timesteps between when an action is commanded and when it is realized
        self.AUGMENT_STATE_WITH_ACTION_LENGTH = 0 # [timesteps] how many timesteps of previous actions should be included in the state. This helps with making good decisions among delayed dynamics.
        self.MAX_NUMBER_OF_TIMESTEPS          = 300# per episode
        self.ADDITIONAL_VALUE_INFO            = False # whether or not to include additional reward and value distribution information on the animations
        self.SKIP_FAILED_ANIMATIONS           = True # Error the program or skip when animations fail?        
        #self.KI                               = [17.0,17.0,0.295,0.02,0.0036,0.00008] # Integral gains for the integral-acceleration controller of the body and arm (x, y, theta, theta1, theta2, theta3)
        self.KI                               = [0,0,0,0,0,0] # [Integral controller is turned off because the feedforward controller is working perfectly] Integral gains for the integral-acceleration controller of the body and arm (x, y, theta, theta1, theta2, theta3)
                                
        # Physical properties (See Fig. 3.1 in Alex Cran's MASc Thesis for definitions)
        self.LENGTH   = 0.3 # [m] side length
        self.PHI      = 74.3106*np.pi/180#np.pi/2 # [rad] angle of anchor point of arm with respect to spacecraft body frame
        self.B0       = 0.2515#(self.LENGTH/2)/np.cos(np.pi/2-self.PHI) # scalar distance from centre of mass to arm attachment point
        self.MASS     = 11.211# [kg] for chaser
        self.M1       = 0.3450 # [kg] link mass
        self.M2       = 0.3350 # [kg] link mass
        self.M3       = 0.1110 # [kg] link mass        
        self.A1       = 0.196822 # [m] base of link to centre of mass
        self.B1       = 0.107678 # [m] centre of mass to end of link
        self.A2       = 0.198152 # [m] base of link to centre of mass
        self.B2       = 0.106348 # [m] centre of mass to end of link
        self.A3       = 0.062097 # [m] base of link to centre of mass
        self.B3       = 0.025153 # [m] centre of mass to end of link
        self.INERTIA  = 0.202150 # [kg m^2] from Crain and Ulrich
        self.INERTIA1 = 0.003704 # [kg m^2] from Crain and Ulrich
        self.INERTIA2 = 0.003506 # [kg m^2] from Crain and Ulrich       
        self.INERTIA3 = 0.000106 # [kg m^2] from Crain and Ulrich
        
        # Target Physical Properties
        self.TARGET_MASS = 12.0390 # [kg]
        self.TARGET_INERTIA = 0.225692 # [kg m^2 theoretical]
        
        # Platform physical properties        
        self.LENGTH_RANDOMIZATION          = 0.1 # [m] standard deviation of the LENGTH randomization when domain randomization is performed.        
        self.MASS_RANDOMIZATION            = 1.0 # [kg] standard deviation of the MASS randomization when domain randomization is performed.

        # Real docking port
        self.DOCKING_PORT_MOUNT_POSITION = np.array([0.0743, 0.2288]) # [m] with respect to the centre of mass
        self.DOCKING_PORT_CORNER1_POSITION = self.DOCKING_PORT_MOUNT_POSITION + [ 0.0508, 0.0432562] # position of the docking cone on the target in its body frame
        self.DOCKING_PORT_CORNER2_POSITION = self.DOCKING_PORT_MOUNT_POSITION + [-0.0508, 0.0432562] # position of the docking cone on the target in its body frame
                
        # Reward function properties
        self.DOCKING_REWARD                        = 100 # A lump-sum given to the chaser when it docks
        self.SUCCESSFUL_DOCKING_RADIUS             = 0.04 # [m] distance at which the magnetic docking can occur
        self.MAX_DOCKING_ANGLE_PENALTY             = 50 # A penalty given to the chaser, upon docking, for having an angle when docking. The penalty is 0 upon perfect docking and MAX_DOCKING_ANGLE_PENALTY upon perfectly bad docking
        self.DOCKING_EE_VELOCITY_PENALTY           = 50 # A penalty given to the chaser, upon docking, for every 1 m/s end-effector collision velocity upon docking
        self.ALLOWED_EE_COLLISION_VELOCITY         = 0 # [m/s] the end-effector is not penalized if it collides with the docking port at up to this speed.
        self.DOCKING_ANGULAR_VELOCITY_PENALTY      = 25 # A penalty given to the chaser, upon docking, for every 1 rad/s angular body velocity upon docking
        self.ALLOWED_EE_COLLISION_ANGULAR_VELOCITY = 0 # [rad/s] the end-effector is not penalized if it collides with the docking port at up to this angular velocity.
        self.END_ON_FALL                           = True # end episode on a fall off the table        
        self.FALL_OFF_TABLE_PENALTY                = 100.
        self.CHECK_CHASER_TARGET_COLLISION         = True
        self.TARGET_COLLISION_PENALTY              = 0 # [rewards/timestep] penalty given for colliding with target  
        self.CHECK_END_EFFECTOR_COLLISION          = True # Whether to do collision detection on the end-effector
        self.CHECK_END_EFFECTOR_FORBIDDEN          = True # Whether to expand the collision area to include the forbidden zone
        self.END_EFFECTOR_COLLISION_PENALTY        = 0 # [rewards/timestep] Penalty for end-effector collisions (with target or optionally with the forbidden zone)
        self.END_ON_COLLISION                      = True # Whether to end the episode upon a collision.
        self.GIVE_MID_WAY_REWARD                   = True # Whether or not to give a reward mid-way towards the docking port to encourage the learning to move in the proper direction
        self.MID_WAY_REWARD_RADIUS                 = 0.1 # [ms] the radius from the DOCKING_PORT_MOUNT_POSITION that the mid-way reward is given
        self.MID_WAY_REWARD                        = 25 # The value of the mid-way reward
        self.ANGULAR_MOMENTUM_PENALTY              = 50 # Max angular momentum penalty to give...
        self.AT_MAX_ANGULAR_MOMENTUM               = 2 # [kg m^2/s] which is given at this angular momentum
        self.END_ON_ARM_LIMITS                     = False # Whether or not to end the episode when an arm link reaches its limit
        self.ARM_LIMIT_PENALTY                     = 5 #[rewards/timestep/link] Penalty for manipulator joints reaching their limits
        
        
        # Some calculations that don't need to be changed
        self.TABLE_BOUNDARY    = Polygon(np.array([[0,0], [self.MAX_X_POSITION, 0], [self.MAX_X_POSITION, self.MAX_Y_POSITION], [0, self.MAX_Y_POSITION], [0,0]]))
        self.VELOCITY_LIMIT    = np.array([self.MAX_VELOCITY, self.MAX_VELOCITY, self.MAX_BODY_ANGULAR_VELOCITY, self.MAX_ARM_ANGULAR_VELOCITY, self.MAX_ARM_ANGULAR_VELOCITY, self.MAX_ARM_ANGULAR_VELOCITY]) # [m/s, m/s, rad/s] maximum allowable velocity/angular velocity; enforced by the controller
        self.ANGLE_LIMIT       = np.pi/2 # Used as a hard limit in the dynamics in order to protect the arm from hitting the chaser
        self.LOWER_STATE_BOUND = np.concatenate([self.LOWER_STATE_BOUND, np.tile(self.LOWER_ACTION_BOUND, self.AUGMENT_STATE_WITH_ACTION_LENGTH)]) # lower bound for each element of TOTAL_STATE
        self.UPPER_STATE_BOUND = np.concatenate([self.UPPER_STATE_BOUND, np.tile(self.UPPER_ACTION_BOUND, self.AUGMENT_STATE_WITH_ACTION_LENGTH)]) # upper bound for each element of TOTAL_STATE        
        self.OBSERVATION_SIZE  = self.TOTAL_STATE_SIZE - len(self.IRRELEVANT_STATES) # the size of the observation input to the policy
        
        # Enabling the extra printing
        self.extra_printing = True


    ######################################
    ##### Resettings the Environment #####
    ######################################
    def reset(self, test_time):
        # This method resets the state
        """ NOTES:
               - if test_time = True -> do not add "controller noise" to the kinematics
        """
        # Reset the seed for max randomness
        np.random.seed()
                
        # Resetting the time
        self.time = 0.        

        # Logging whether it is test time for this episode
        self.test_time = test_time
        
        # Resetting the mid-way flag
        self.not_yet_mid_way = True

        # If we are randomizing the initial conditions and state
        if self.RANDOMIZE_INITIAL_CONDITIONS:
            # Randomizing initial state in Inertial frame
            self.chaser_position = self.INITIAL_CHASER_POSITION + np.random.uniform(low = -1, high = 1, size = 3)*[self.RANDOMIZATION_LENGTH_X, self.RANDOMIZATION_LENGTH_Y, self.RANDOMIZATION_ANGLE]
            # Randomizing initial claser velocity in Inertial Frame
            self.chaser_velocity = self.INITIAL_CHASER_VELOCITY + np.random.uniform(low = -1, high = 1, size = 3)*[self.RANDOMIZATION_CHASER_VELOCITY, self.RANDOMIZATION_CHASER_VELOCITY, self.RANDOMIZATION_CHASER_OMEGA]
            # Randomizing target state in Inertial frame
            self.target_position = self.INITIAL_TARGET_POSITION + np.random.uniform(low = -1, high = 1, size = 3)*[self.RANDOMIZATION_LENGTH_X, self.RANDOMIZATION_LENGTH_Y, self.RANDOMIZATION_ANGLE]
            # Randomizing target velocity in Inertial frame
            self.target_velocity = self.INITIAL_TARGET_VELOCITY + np.random.uniform(low = -1, high = 1, size = 3)*[self.RANDOMIZATION_TARGET_VELOCITY, self.RANDOMIZATION_TARGET_VELOCITY, self.RANDOMIZATION_TARGET_OMEGA]
            # Randomizing arm angles in Body frame
            self.arm_angles = self.INITIAL_ARM_ANGLES + np.random.uniform(low = -1, high = 1, size = 3)*[self.RANDOMIZATION_ARM_ANGLE, self.RANDOMIZATION_ARM_ANGLE, self.RANDOMIZATION_ARM_ANGLE]
            # Randomizing arm angular rates in body frame
            self.arm_angular_rates = self.INITIAL_ARM_RATES + np.random.uniform(low = -1, high = 1, size = 3)*[self.RANDOMIZATION_ARM_RATES, self.RANDOMIZATION_ARM_RATES, self.RANDOMIZATION_ARM_RATES]
            

        else:
            # Constant initial state in Inertial frame
            self.chaser_position = self.INITIAL_CHASER_POSITION
            # Constant chaser velocity in Inertial frame
            self.chaser_velocity = self.INITIAL_CHASER_VELOCITY
            # Constant target location in Inertial frame
            self.target_position = self.INITIAL_TARGET_POSITION
            # Constant target velocity in Inertial frame
            self.target_velocity = self.INITIAL_TARGET_VELOCITY
            # Constant initial arm position in Body frame
            self.arm_angles = self.INITIAL_ARM_ANGLES
            # Constand arm angular velocity in Body frame
            self.arm_angular_rates = self.INITIAL_ARM_RATES
        
        # TODO: Build domain randomization
        
        # Update docking component locations
        self.update_end_effector_and_docking_locations()
        
        # Also update the end-effector position & velocity in the body frame
        self.update_end_effector_location_body_frame()
        
        # Update relative pose
        self.update_relative_pose_body_frame()
        
        # Check for collisions
        self.check_collisions()
        # If we are colliding (unfairly) upon a reset, reset the environment again!
        if self.end_effector_collision or self.forbidden_area_collision or self.chaser_target_collision or self.elbow_target_collision or not(self.chaser_on_table):
            # Reset the environment again!
            self.reset(test_time)        
 
        # Initializing the previous velocity and control effort for the integral-acceleration controller
        self.previous_velocity       = np.zeros(self.ACTION_SIZE)
        self.previous_control_effort = np.zeros(self.ACTION_SIZE)
        
        # Initializing integral anti-wind-up that checks if the joints angles have been reached
        self.joints_past_limits = [False, False, False]

        # Resetting the action delay queue
        if self.DYNAMICS_DELAY > 0:
            self.action_delay_queue = queue.Queue(maxsize = self.DYNAMICS_DELAY + 1)
            for i in range(self.DYNAMICS_DELAY):
                self.action_delay_queue.put(np.zeros(self.ACTION_SIZE), False)
                

    def update_end_effector_and_docking_locations(self):
        """
        This method returns the location of the end-effector of the manipulator
        based off the current state in the Inertial frame
        
        It also updates the docking port position on the target
        """
        ##########################
        ## End-effector Section ##
        ##########################
        # Unpacking the state
        x, y, theta                           = self.chaser_position
        x_dot, y_dot, theta_dot               = self.chaser_velocity
        theta_1, theta_2, theta_3             = self.arm_angles
        theta_1_dot, theta_2_dot, theta_3_dot = self.arm_angular_rates

        x_ee = x + self.B0*np.cos(self.PHI + theta) + (self.A1 + self.B1)*np.cos(np.pi/2 + theta + theta_1) + \
               (self.A2 + self.B2)*np.cos(np.pi/2 + theta + theta_1 + theta_2) + \
               (self.A3 + self.B3)*np.cos(np.pi/2 + theta + theta_1 + theta_2 + theta_3)

        x_ee_dot = x_dot - self.B0*np.sin(self.PHI + theta)*(theta_dot) - (self.A1 + self.B1)*np.sin(np.pi/2 + theta + theta_1)*(theta_dot + theta_1_dot) - \
                           (self.A2 + self.B2)*np.sin(np.pi/2 + theta + theta_1 + theta_2)*(theta_dot + theta_1_dot + theta_2_dot) - \
                           (self.A3 + self.B3)*np.sin(np.pi/2 + theta + theta_1 + theta_2 + theta_3)*(theta_dot + theta_1_dot + theta_2_dot + theta_3_dot)
                           
        y_ee = y + self.B0*np.sin(self.PHI + theta) + (self.A1 + self.B1)*np.sin(np.pi/2 + theta + theta_1) + \
               (self.A2 + self.B2)*np.sin(np.pi/2 + theta + theta_1 + theta_2) + \
               (self.A3 + self.B3)*np.sin(np.pi/2 + theta + theta_1 + theta_2 + theta_3)
        
        y_ee_dot = y_dot + self.B0*np.cos(self.PHI + theta)*(theta_dot) + (self.A1 + self.B1)*np.cos(np.pi/2 + theta + theta_1)*(theta_dot + theta_1_dot) + \
                           (self.A2 + self.B2)*np.cos(np.pi/2 + theta + theta_1 + theta_2)*(theta_dot + theta_1_dot + theta_2_dot) + \
                           (self.A3 + self.B3)*np.cos(np.pi/2 + theta + theta_1 + theta_2 + theta_3)*(theta_dot + theta_1_dot + theta_2_dot + theta_3_dot)

        # Updates the position of the end-effector in the Inertial frame
        self.end_effector_position = np.array([x_ee, y_ee])
        
        # End effector velocity
        self.end_effector_velocity = np.array([x_ee_dot, y_ee_dot]) 
        
        ###################
        ## Elbow Section ##
        ###################
        x_elbow = x + self.B0*np.cos(self.PHI + theta) + (self.A1 + self.B1)*np.cos(np.pi/2 + theta + theta_1)                  
        y_elbow = y + self.B0*np.sin(self.PHI + theta) + (self.A1 + self.B1)*np.sin(np.pi/2 + theta + theta_1)
                  
        self.elbow_position = np.array([x_elbow, y_elbow])        
        
        ##########################
        ## Docking port Section ##
        ##########################
        # Make rotation matrix
        C_Ib_target = self.make_C_bI(self.target_position[-1]).T
        
        # Position in Inertial = Body position (inertial) + C_Ib * EE position in body
        self.docking_port_position = self.target_position[:-1] + np.matmul(C_Ib_target, self.DOCKING_PORT_MOUNT_POSITION)
        
        # Velocity in Inertial = target_velocity + omega_target [cross] r_{port/G}
        self.docking_port_velocity = self.target_velocity[:-1] + self.target_velocity[-1] * np.matmul(self.make_C_bI(self.target_position[-1]).T,[-self.DOCKING_PORT_MOUNT_POSITION[1], self.DOCKING_PORT_MOUNT_POSITION[0]])

    def update_end_effector_location_body_frame(self):
        """
        This method returns the location of the end-effector of the manipulator
        based off the current state in the chaser's body frame
        """
        ##########################
        ## End-effector Section ##
        ##########################
        # Unpacking the state
        theta_1, theta_2, theta_3             = self.arm_angles
        theta_1_dot, theta_2_dot, theta_3_dot = self.arm_angular_rates

        x_ee = self.B0*np.cos(self.PHI) + (self.A1 + self.B1)*np.cos(np.pi/2 + theta_1) + \
               (self.A2 + self.B2)*np.cos(np.pi/2 + theta_1 + theta_2) + \
               (self.A3 + self.B3)*np.cos(np.pi/2 + theta_1 + theta_2 + theta_3)

        x_ee_dot = (self.A1 + self.B1)*np.sin(np.pi/2 + theta_1)*(theta_1_dot) - \
                           (self.A2 + self.B2)*np.sin(np.pi/2 + theta_1 + theta_2)*(theta_1_dot + theta_2_dot) - \
                           (self.A3 + self.B3)*np.sin(np.pi/2 + theta_1 + theta_2 + theta_3)*(theta_1_dot + theta_2_dot + theta_3_dot)
                           
        y_ee = self.B0*np.sin(self.PHI) + (self.A1 + self.B1)*np.sin(np.pi/2 + theta_1) + \
               (self.A2 + self.B2)*np.sin(np.pi/2 + theta_1 + theta_2) + \
               (self.A3 + self.B3)*np.sin(np.pi/2 + theta_1 + theta_2 + theta_3)
        
        y_ee_dot = (self.A1 + self.B1)*np.cos(np.pi/2 + theta_1)*(theta_1_dot) + \
                           (self.A2 + self.B2)*np.cos(np.pi/2 + theta_1 + theta_2)*(theta_1_dot + theta_2_dot) + \
                           (self.A3 + self.B3)*np.cos(np.pi/2 + theta_1 + theta_2 + theta_3)*(theta_1_dot + theta_2_dot + theta_3_dot)

        # Updates the position of the end-effector in the chaser's body frame
        self.end_effector_position_body = np.array([x_ee, y_ee])
        
        # End effector velocity in the chaser's body frame
        self.end_effector_velocity_body = np.array([x_ee_dot, y_ee_dot]) 


    def make_total_state(self):
        
        # Assembles all the data into the shape of TOTAL STATE so that it is consistent
        # chaser_x, chaser_y, chaser_theta, chaser_x_dot, chaser_y_dot, chaser_theta_dot, 
        # shoulder_theta, elbow_theta, wrist_theta, shoulder_theta_dot, elbow_theta_dot, wrist_theta_dot, 
        # target_x, target_y, target_theta, target_x_dot, target_y_dot, target_theta_dot, 
        # ee_x_I, ee_y_I, ee_x_dot_I, ee_y_dot_I,
        # relative_x_b, relative_y_b, relative_theta,
        # ee_x_b, ee_y_b, ee_x_dot_b, ee_y_dot_b]
        
        total_state = np.concatenate([self.chaser_position[:2], np.array([self.chaser_position[2] % (2*np.pi)]), self.chaser_velocity, self.arm_angles, self.arm_angular_rates, self.target_position[:2], np.array([self.target_position[2] % (2*np.pi)]), self.target_velocity, self.end_effector_position, self.end_effector_velocity, self.relative_position_inertial, self.relative_angle, self.end_effector_position_body, self.end_effector_velocity_body])
        
        return total_state
    
    def update_relative_pose_body_frame(self):
        # Calculate the relative_x, relative_y, relative_angle
        # All in the chaser's body frame
                
        chaser_angle = self.chaser_position[-1]        
        # Rotation matrix (inertial -> body)
        C_bI = self.make_C_bI(chaser_angle)
                
        # [X,Y] relative position in inertial frame
        self.relative_position_inertial = self.target_position[:-1] - self.chaser_position[:-1]    
        
        # Rotate it to the body frame and save it
        self.relative_position_body = np.matmul(C_bI, self.relative_position_inertial)
        
        # Relative angle and wrap it to [0, 2*np.pi]
        self.relative_angle = np.array([(self.target_position[-1] - self.chaser_position[-1])%(2*np.pi)])

    
    def make_chaser_state(self):
        
        # Assembles all chaser-relevant data into a state to be fed to the equations of motion
        
        total_chaser_state = np.concatenate([self.chaser_position, self.arm_angles, self.chaser_velocity, self.arm_angular_rates])
        
        return total_chaser_state
    #####################################
    ##### Step the Dynamics forward #####
    #####################################
    def step(self, action):

        # Integrating forward one time step using the calculated action.
        # Oeint returns initial condition on first row then next TIMESTEP on the next row

        ############################
        #### PROPAGATE DYNAMICS ####
        ############################

        # First, calculate the control effort
        control_effort = self.controller(action)

        # Anything that needs to be sent to the dynamics integrator
        dynamics_parameters = [control_effort, self.LENGTH, self.PHI, self.B0, self.MASS, self.M1, self.M2, self.M3, self.A1, self.B1, self.A2, self.B2, self.A3, self.B3, self.INERTIA, self.INERTIA1, self.INERTIA2, self.INERTIA3]
        
        # Building the state
        current_chaser_state = self.make_chaser_state()
        
        # Propagate the dynamics forward one timestep
        next_states = odeint(dynamics_equations_of_motion, current_chaser_state, [self.time, self.time + self.TIMESTEP], args = (dynamics_parameters,), full_output = 0)

        # Saving the new state
        new_chaser_state = next_states[1,:]
        
        # The inverse of make_chaser_state()
        self.chaser_position = new_chaser_state[0:3]
        self.arm_angles = new_chaser_state[3:6]
        self.chaser_velocity = new_chaser_state[6:9]
        self.arm_angular_rates = new_chaser_state[9:12]
        
        # Setting a hard limit on the manipulator angles
        #TODO: Investigate momentum transfer when limits are hit. It seems like I have to do 
        #      this either through conservation of momentum or a collision force?
        self.joints_past_limits = np.abs(self.arm_angles) > self.ANGLE_LIMIT
        if np.any(self.joints_past_limits):
            # Hold the angle at the limit
            self.arm_angles[self.joints_past_limits] = np.sign(self.arm_angles[self.joints_past_limits]) * self.ANGLE_LIMIT
            # Set the angular rate to zero
            self.arm_angular_rates[self.joints_past_limits] = 0
            # Set the past control effort to 0 to prevent further wind-up
            # Removed because wind-up was totally removed. Instead, stop it from increasing in the controller 
            #self.previous_control_effort[3:][self.joints_past_limits] = 0            

        # Step target's state ahead one timestep
        self.target_position += self.target_velocity * self.TIMESTEP
        
        # Update docking locations
        self.update_end_effector_and_docking_locations()
        
        # Also update the end-effector position & velocity in the body frame
        self.update_end_effector_location_body_frame()
        
        # Update relative pose
        self.update_relative_pose_body_frame()
        
        # Check for collisions
        self.check_collisions()
        
        # Increment the timestep
        self.time += self.TIMESTEP

        # Calculating the reward for this state-action pair
        reward = self.reward_function(action)

        # Check if this episode is done
        done = self.is_done()

        # Return the (reward, done)
        return reward, done


    def controller(self, action):
        # This function calculates the control effort based on the state and the
        # desired acceleration (action)
        
        ########################################
        ### Integral-acceleration controller ###
        ########################################
        desired_accelerations = action
        if self.CALIBRATE_TIMESTEP:
            desired_accelerations = self.PREDETERMINED_ACTION
        
        # Stopping the command of additional velocity when we are already at our maximum
        current_velocity = np.concatenate([self.chaser_velocity, self.arm_angular_rates])        
        if not self.CALIBRATE_TIMESTEP:
            desired_accelerations[(np.abs(current_velocity) > self.VELOCITY_LIMIT) & (np.sign(desired_accelerations) == np.sign(current_velocity))] = 0
        
        # Approximating the current accelerations
        current_accelerations = (current_velocity - self.previous_velocity)/self.TIMESTEP
        self.previous_velocity = current_velocity
        
        """ This integral-acceleration transpose jacobian controller is implemented but not used """
#            #######################################################################
#            ### Integral-acceleration transpose Jacobian controller for the arm ###
#            #######################################################################
#            """
#            Using the jacobian, J, for joint 3 as described in Eq 3.31 of Alex Crain's MASc Thesis
#            v = velocity of end-effector in inertial frame
#            w = angular velocity of end-effector about its centre of mass
#            q = [chaser_x, chaser_y, theta, theta_1, theta_2, theta_3]
#            [v,w] = J*qdot
#            Therefore, using the transpose Jacobian trick
#            forces_torques = J.T * F_ee
#            where F_ee is [ee_f_x, ee_f_y, ee_tau_z]
#            and forces_torques is [body_Fx, body_Fy, body_tau, tau1, tau2, tau3]
#            J = self.make_jacobian()
#            """
#    
#            current_ee_velocity = np.array([self.end_effector_velocity[0], self.end_effector_velocity[1], np.sum((self.arm_angular_rates)) + self.chaser_velocity[-1]])
#            current_ee_pose_acceleration = (current_ee_velocity - self.previous_ee_pose_velocity)/self.TIMESTEP
#            
#            # Calculating the end-effector acceleration error
#            ee_acceleration_error = desired_ee_accelerations - current_ee_pose_acceleration
#            
#            # End-effector control effort
#            ee_control_effort = self.previous_ee_control_effort + self.KI[3:] * ee_acceleration_error
#            
#            # Saving the current velocity and control effort for the next timetsep
#            self.previous_ee_pose_velocity = current_ee_pose_velocity
#            self.previous_ee_control_effort = ee_control_effort
#            
#            # Using the Transpose Jacobian
#            joint_space_torque = np.matmul(self.make_jacobian().T, ee_control_effort)
#    
#            # Assuming the body control effort was calculated previously, concatenate/add it here
#            control_effort = np.concatenate([body_control_effort, joint_space_torque[3:]]).reshape([6,1])
#            
  
        ##########################################################
        ### Integral-acceleration controller on the all states ###
        ########################################################## 
        """
        # Calculate the acceleration error
        acceleration_error = desired_accelerations - current_accelerations
        
        # If the joint is currently at its limit and the desired acceleration is worsening the problem, set the acceleration error to 0. This will prevent further integral wind-up but not release the current wind-up.
        acceleration_errors_to_zero = (self.joints_past_limits) & (np.sign(desired_accelerations[3:]) == np.sign(self.arm_angles))
        acceleration_error[3:][acceleration_errors_to_zero] = 0
                
        # Apply the integral controller 
        control_effort = self.previous_control_effort + self.KI * acceleration_error

        # Clip commands to ensure they respect the hardware limits
        limits = np.concatenate([np.tile(self.MAX_THRUST,2), [self.MAX_BODY_TORQUE], np.tile(self.MAX_JOINT1n2_TORQUE,2), [self.MAX_JOINT3_TORQUE]])        
        
        # If we are trying to calibrate gains and torque bounds...
        if self.CALIBRATE_TIMESTEP:
            print("Accelerations: ", current_accelerations, " Unclipped Control Effort: ", control_effort, end = "")
            if self.CLIP_DURING_CALIBRATION:
                control_effort = np.clip(control_effort, -limits, limits)
                print(" Clipped Control Effort: ", control_effort)
            else:
                print(" ")
                #pass
        else:
            control_effort = np.clip(control_effort, -limits, limits)
            #pass

        # Logging current control effort for next time step
        self.previous_control_effort = control_effort
        # [F_x, F_y, torque, torque1, torque2, torque3]
        return control_effort.reshape([self.ACTION_SIZE,1])
        """
        
        ########################################################
        ### Integral Controller with Feedfoward Compensation ###
        ########################################################
        # Added May 31, 2021, replacing the above integral-only controller
        
        # Calculate the acceleration error
        acceleration_error = desired_accelerations - current_accelerations
        
        # If the joint is currently at its limit and the desired acceleration is worsening the problem, set the acceleration error to 0. This will prevent further integral wind-up but not release the current wind-up.
        acceleration_errors_to_zero = (self.joints_past_limits) & (np.sign(desired_accelerations[3:]) == np.sign(self.arm_angles))
        acceleration_error[3:][acceleration_errors_to_zero] = 0
                
        # Apply the integral controller 
        control_effort = self.previous_control_effort + self.KI * acceleration_error
        self.previous_control_effort = np.copy(control_effort)
        
        # Apply the feedforward compensation
        current_chaser_state = self.make_chaser_state()
        dynamics_parameters = [control_effort, self.LENGTH, self.PHI, self.B0, self.MASS, self.M1, self.M2, self.M3, self.A1, self.B1, self.A2, self.B2, self.A3, self.B3, self.INERTIA, self.INERTIA1, self.INERTIA2, self.INERTIA3]
        desired_velocities = current_velocity + desired_accelerations*self.TIMESTEP
        control_effort += np.matmul(calculate_mass_matrix(current_chaser_state, 0, dynamics_parameters), desired_accelerations) + np.matmul(calculate_coriolis_matrix(current_chaser_state, 0, dynamics_parameters), desired_velocities)
        
        # Clip commands to ensure they respect the hardware limits
        limits = np.concatenate([np.tile(self.MAX_THRUST,2), [self.MAX_BODY_TORQUE], np.tile(self.MAX_JOINT1n2_TORQUE,2), [self.MAX_JOINT3_TORQUE]])        
        
        # If we are trying to calibrate gains and torque bounds...
        if self.CALIBRATE_TIMESTEP:
            #print("Current Accelerations: ", current_accelerations, "Desired Accelerations: ", desired_accelerations, " Unclipped Control Effort: ", control_effort, end = "")
            if self.CLIP_DURING_CALIBRATION:
                unclipped = control_effort
                control_effort = np.clip(control_effort, -limits, limits)
                #print(" Clipped Control Effort: ", control_effort)
            else:
                #print(" ")
                pass
            print("Current, Desired, Unclipped, Clipped\n", np.concatenate([current_accelerations.reshape([1,-1]), desired_accelerations.reshape([1,-1]), unclipped.reshape([1, -1]), control_effort.reshape([1,-1])], axis=0))
        else:
            control_effort = np.clip(control_effort, -limits, limits)
            #pass
            
        # [F_x, F_y, torque, torque1, torque2, torque3]
        return control_effort.reshape([self.ACTION_SIZE,1])
        
    
    def make_jacobian_Jc1(self):
        # This method calculates the jacobian Jc1 for the arm

        PHI = self.PHI
        q0 = self.chaser_position[-1]
        q1 = self.arm_angles[0]
        
        b0 = self.B0
        a1 = self.A1
        
        S0 = np.sin(PHI + q0)
        S1 = np.sin(np.pi/2 + q0 + q1) 
        C0 = np.cos(PHI + q0)
        C1 = np.cos(np.pi/2 + q0 + q1) 
        
        Jc1_13 = -b0*S0 - a1*S1
        Jc1_14 = -a1*S1
        Jc1_23 = b0*C0 + a1*C1
        Jc1_24 = a1*C1
        
        jacobian = np.array([[1,0,Jc1_13,Jc1_14,0,0],
                             [0,1,Jc1_23,Jc1_24,0,0],
                             [0,0,1,1,0,0]])
        
        return jacobian
    
    def make_jacobian_Jc2(self):
        # This method calculates the jacobian Jc2 for the arm

        PHI = self.PHI
        q0 = self.chaser_position[-1]
        q1 = self.arm_angles[0]
        q2 = self.arm_angles[1]
        
        b0 = self.B0
        a1 = self.A1
        b1 = self.B1
        a2 = self.A2
        
        L1 = a1 + b1
        
        S0 = np.sin(PHI + q0)
        S1 = np.sin(np.pi/2 + q0 + q1) 
        S2 = np.sin(np.pi/2 + q0 + q1 + q2) 
        C0 = np.cos(PHI + q0)
        C1 = np.cos(np.pi/2 + q0 + q1) 
        C2 = np.cos(np.pi/2 + q0 + q1 + q2)
        
        Jc2_13 = -b0*S0 - L1*S1 -a2*S2
        Jc2_14 = -L1*S1 - a2*S2
        Jc2_15 = -a2*S2
        Jc2_23 = b0*C0 + L1*C1 + a2*C2
        Jc2_24 = L1*C1 + a2*C2
        Jc2_25 = a2*C2
        
        jacobian = np.array([[1,0,Jc2_13,Jc2_14,Jc2_15,0],
                             [0,1,Jc2_23,Jc2_24,Jc2_25,0],
                             [0,0,1,1,1,0]])
        
        return jacobian
    
    def make_jacobian_Jc3(self):
        # This method calculates the jacobian Jc3 for the arm

        PHI = self.PHI
        q0 = self.chaser_position[-1]
        q1 = self.arm_angles[0]
        q2 = self.arm_angles[1]
        q3 = self.arm_angles[2]
        
        b0 = self.B0
        a1 = self.A1
        b1 = self.B1
        a2 = self.A2
        b2 = self.B2
        a3 = self.A3
        
        L1 = a1 + b1
        L2 = a2 + b2
        
        S0 = np.sin(PHI + q0)
        S1 = np.sin(PHI + q0 + q1) 
        S2 = np.sin(PHI + q0 + q1 + q2) 
        S3 = np.sin(PHI + q0 + q1 + q2 + q3) 
        C0 = np.cos(PHI + q0)
        C1 = np.cos(PHI + q0 + q1) 
        C2 = np.cos(PHI + q0 + q1 + q2)
        C3 = np.cos(PHI + q0 + q1 + q2 + q3) 
        
        Jc3_13 = -b0*S0 - L1*S1 - L2*S2 - a3*S3
        Jc3_14 = -L1*S1 -L2*S2 - a3*S3
        Jc3_15 = -L2*S2 -a3*S3
        Jc3_16 = -a3*S3
        Jc3_23 = b0*C0 + L1*C1 + L2*C2 +a3*C3
        Jc3_24 = L1*C1 + L2*C2 + a3*C3
        Jc3_25 = L2*C2 + a3*C3
        Jc3_26 = a3*C3
        
        jacobian = np.array([[1,0,Jc3_13,Jc3_14,Jc3_15,Jc3_16],
                             [0,1,Jc3_23,Jc3_24,Jc3_25,Jc3_26],
                             [0,0,1,1,1,1]])
        
        return jacobian
    
    
    def combined_angular_momentum(self):
        # This method returns the angular momentum of the combined chaser-manipulator-target system. It assumes that docking has occurred.
        
        #########################################
        ### Calculate chaser's centre of mass ###
        #########################################
        x, y, theta               = self.chaser_position
        theta_1, theta_2, theta_3 = self.arm_angles
        chaser_body_com = np.array([x,y])
        link1_com = np.array([x + self.B0*np.cos(self.PHI + theta) + self.A1*np.cos(np.pi/2 + theta + theta_1),
                              y + self.B0*np.sin(self.PHI + theta) + self.A1*np.sin(np.pi/2 + theta + theta_1)])
        link2_com = link1_com + np.array([self.B1*np.cos(np.pi/2 + theta + theta_1) + self.A2*np.cos(np.pi/2 + theta + theta_1 + theta_2),
                                          self.B1*np.sin(np.pi/2 + theta + theta_1) + self.A2*np.sin(np.pi/2 + theta + theta_1 + theta_2)])
        link3_com = link2_com + np.array([self.B2*np.cos(np.pi/2 + theta + theta_1 + theta_2) + self.A3*np.cos(np.pi/2 + theta + theta_1 + theta_2 + theta_3),
                                          self.B2*np.sin(np.pi/2 + theta + theta_1 + theta_2) + self.A3*np.sin(np.pi/2 + theta + theta_1 + theta_2 + theta_3)])            
        chaser_com = (self.MASS*chaser_body_com + self.M1*link1_com + self.M2*link2_com + self.M3*link3_com)/(self.MASS + self.M1 + self.M2 + self.M3)

        ##############################################################
        ### Calculate link inertial velocities using the Jacobians ###
        ##############################################################
        Jc1 = self.make_jacobian_Jc1()
        Jc2 = self.make_jacobian_Jc2()
        Jc3 = self.make_jacobian_Jc3()
        
        arm1_rates = np.matmul(Jc1, np.concatenate([self.chaser_velocity, self.arm_angular_rates]))
        arm2_rates = np.matmul(Jc2, np.concatenate([self.chaser_velocity, self.arm_angular_rates]))
        arm3_rates = np.matmul(Jc3, np.concatenate([self.chaser_velocity, self.arm_angular_rates]))
        
        # Extract arm centre-of-mass velocity and arm inertial angular rate
        v1 = arm1_rates[:-1]
        omega1 = arm1_rates[-1]
        v2 = arm2_rates[:-1]
        omega2 = arm2_rates[-1]
        v3 = arm3_rates[:-1]
        omega3 = arm3_rates[-1]
        
        ########################################################################################
        ### Calculate the linear and angular momentum of the chaser about its centre of mass ###
        ########################################################################################
        # Calculate angular momentum of each chaser object about the chaser's centre of mass
        h_com_chaser = self.INERTIA*self.chaser_velocity[-1] + self.MASS*np.cross(self.chaser_position[:-1] - chaser_com, self.chaser_velocity[:-1])
        h_com_1 = self.INERTIA1*omega1 + self.M1*np.cross(link1_com - chaser_com, v1)
        h_com_2 = self.INERTIA1*omega2 + self.M2*np.cross(link2_com - chaser_com, v2)
        h_com_3 = self.INERTIA1*omega3 + self.M3*np.cross(link3_com - chaser_com, v3)
        # And the combined angular momentum
        total_angular_momentum_chaser_com = h_com_chaser + h_com_1 + h_com_2 + h_com_3
        
        # Calculate the linear momentum of the chaser
        p_chaser = self.MASS*self.chaser_velocity[:-1]
        p_1 = self.M1*v1
        p_2 = self.M1*v2
        p_3 = self.M1*v3            
        total_linear_momentum_chaser = p_chaser + p_1 + p_2 + p_3    
        
        #print("Total linear momentum of the chaser: ", total_linear_momentum_chaser)
        
        ####################################################################################
        ### Calculate linear and angular momentum of the target about its centre of mass ###
        ####################################################################################
        linear_momentum_target = self.TARGET_MASS*self.target_velocity[:-1]
        angular_momentum_target_com = self.TARGET_INERTIA*self.target_velocity[-1]        
        
        ############################################################################
        ### Calculate combined chaser-manipulator-target centre of mass location ###
        ############################################################################
        combined_com = ((self.MASS + self.M1 + self.M2 + self.M3)*chaser_com + self.TARGET_MASS*self.target_position[:-1])/(self.MASS + self.M1 + self.M2 + self.M3 + self.TARGET_MASS)
        
        #####################################################################
        ### Calculate combined chaser-manipulator-target angular momentum ###
        #####################################################################
        h_total_combined_com = total_angular_momentum_chaser_com + np.cross(chaser_com - combined_com, total_linear_momentum_chaser) + angular_momentum_target_com + np.cross(self.target_position[:-1] - combined_com, linear_momentum_target)
        
        # Calculate total combined linear momentum, for fun
        #p_total_post_capture = total_linear_momentum_chaser + linear_momentum_target
        
        ##########################################################################################################################
        ### Calculate total inertial of the combined system about the chaser-manipulator-target centre of mass (for curiosity) ###
        ##########################################################################################################################
        # A bunch of parallel axis theorems
        total_inertia = self.INERTIA + self.MASS * np.linalg.norm(self.chaser_position[:-1] - combined_com)**2 + \
                        self.INERTIA1 + self.M1  * np.linalg.norm(link1_com - combined_com)**2 + \
                        self.INERTIA2 + self.M2  * np.linalg.norm(link2_com - combined_com)**2 + \
                        self.INERTIA3 + self.M3  * np.linalg.norm(link3_com - combined_com)**2 + \
                        self.TARGET_INERTIA + self.TARGET_MASS * np.linalg.norm(self.target_position[:-1] - combined_com)**2
        # And the effective postcapture angular velocity of the system (assuming all joints become rigid in the capture position)
        # If the moment of inertia changes, this will change.
        combined_angular_velocity = h_total_combined_com/total_inertia*180/np.pi # [deg/s]
        
        return h_total_combined_com, combined_angular_velocity
    

    def reward_function(self, action):
        # Returns the reward for this TIMESTEP as a function of the state and action
        
        """
        Reward system:
                - Zero reward at all timesteps except when docking is achieved
                - A large reward when docking occurs. The episode also terminates when docking occurs
                - A variety of penalties to help with docking, such as:
                    - penalty for end-effector angle (so it goes into the docking cone properly)
                    - penalty for relative velocity during the docking (so the end-effector doesn't jab the docking cone)
	- penalty for angular velocity of the end-effector upon docking
                - A penalty for colliding with the target
                - 
         """ 
                
        # Initializing the reward
        reward = 0
        
        # Give a large reward for docking
        if self.docked:
            
            reward += self.DOCKING_REWARD
            
            # Penalize for end-effector angle
            end_effector_angle_inertial = self.chaser_position[-1] + np.sum(self.arm_angles) + np.pi/2
            
            # Docking cone angle in the target body frame
            docking_cone_angle_body = np.arctan2(self.DOCKING_PORT_CORNER1_POSITION[1] - self.DOCKING_PORT_CORNER2_POSITION[1], self.DOCKING_PORT_CORNER1_POSITION[0] - self.DOCKING_PORT_CORNER2_POSITION[0])
            docking_cone_angle_inertial = docking_cone_angle_body + self.target_position[-1] - np.pi/2 # additional -pi/2 since we must dock perpendicular into the cone
            
            # Calculate the docking angle error
            docking_angle_error = (docking_cone_angle_inertial - end_effector_angle_inertial + np.pi) % (2*np.pi) - np.pi # wrapping to [-pi, pi] 
            
            # Penalize for any non-zero angle
            reward -= np.abs(np.sin(docking_angle_error/2)) * self.MAX_DOCKING_ANGLE_PENALTY

            # Calculating the docking velocity error
            docking_relative_velocity = self.end_effector_velocity - self.docking_port_velocity
            
            # Applying the penalty
            reward -= np.maximum(0, np.linalg.norm(docking_relative_velocity) - self.ALLOWED_EE_COLLISION_VELOCITY) * self.DOCKING_EE_VELOCITY_PENALTY # 
            
            # Penalize for relative end-effector angular velocity upon docking
            end_effector_angular_velocity = self.chaser_velocity[-1] + np.sum(self.arm_angular_rates)
            reward -= np.maximum(0, np.abs(end_effector_angular_velocity - self.target_velocity[-1]) - self.ALLOWED_EE_COLLISION_ANGULAR_VELOCITY) * self.DOCKING_ANGULAR_VELOCITY_PENALTY
                        
            # Calculate combined angular momentum of docked system
            h_total_combined_com, combined_angular_velocity = self.combined_angular_momentum()
            
            # Add the penalty
            reward -= self.ANGULAR_MOMENTUM_PENALTY*np.abs(h_total_combined_com)/self.AT_MAX_ANGULAR_MOMENTUM
                        
            if self.test_time:
                print("Docking successful! Reward given: %.1f; distance: %.3f m -> Relative ee velocity: %.3f m/s; penalty: %.1f -> Docking angle error: %.2f deg; penalty: %.1f -> EE angular rate error: %.3f; penalty %.1f -> Combined angular momentum: %.3f Nms; penalty: %.1f, Postcapture angular rate %.2f deg/s; Precapture target angular rate: %.2f deg/s" %(reward, np.linalg.norm(self.end_effector_position - self.docking_port_position), np.linalg.norm(docking_relative_velocity), np.maximum(0, np.linalg.norm(docking_relative_velocity) - self.ALLOWED_EE_COLLISION_VELOCITY) * self.DOCKING_EE_VELOCITY_PENALTY, docking_angle_error*180/np.pi, np.abs(np.sin(docking_angle_error/2)) * self.MAX_DOCKING_ANGLE_PENALTY,np.abs(self.chaser_velocity[-1] - self.target_velocity[-1]),np.maximum(0, np.abs(end_effector_angular_velocity - self.target_velocity[-1]) - self.ALLOWED_EE_COLLISION_ANGULAR_VELOCITY) * self.DOCKING_ANGULAR_VELOCITY_PENALTY, h_total_combined_com, self.ANGULAR_MOMENTUM_PENALTY*np.abs(h_total_combined_com)/self.AT_MAX_ANGULAR_MOMENTUM, combined_angular_velocity, self.target_velocity[-1]*180/np.pi))
        
        
        # Give a reward for passing a "mid-way" mark
        if self.GIVE_MID_WAY_REWARD and self.not_yet_mid_way and self.mid_way:
            if self.test_time:
                print("Just passed the mid-way mark. Distance: %.3f at time %.1f" %(np.linalg.norm(self.end_effector_position - self.docking_port_position), self.time))
            self.not_yet_mid_way = False
            reward += self.MID_WAY_REWARD
        
        # Giving a penalty for colliding with the target. These booleans are updated in self.check_collisions()
        if self.chaser_target_collision:
            reward -= self.TARGET_COLLISION_PENALTY
        
        if self.end_effector_collision:
            reward -= self.END_EFFECTOR_COLLISION_PENALTY
        
        if self.forbidden_area_collision:
            reward -= self.END_EFFECTOR_COLLISION_PENALTY
            
        if self.elbow_target_collision:
            reward -= self.END_EFFECTOR_COLLISION_PENALTY
        
        # Give a penalty when an arm segment reaches its limit
        if np.any(self.joints_past_limits):
            reward -= self.ARM_LIMIT_PENALTY*np.sum(self.joints_past_limits)            
        
        # If we've fallen off the table or rotated too much, penalize this behaviour
        if (not(self.chaser_on_table) or np.abs(self.chaser_position[-1]) > 6*np.pi) and self.END_ON_FALL:
            reward -= self.FALL_OFF_TABLE_PENALTY
        
        return reward
    
    def check_collisions(self):
        """ Calculate whether the different objects are colliding with the target.
            It also checks if the chaser has fallen off the table, if the end-effector has docked,
            and if it has reached the mid-way mark
        
            Returns 7 booleans: end_effector_collision, forbidden_area_collision, chaser_target_collision, chaser_on_table, mid_way, docked, and elbow_target_collision
        """
        
        ##################################################
        ### Calculating Polygons in the inertial frame ###
        ##################################################
        
        # Target    
        target_points_body = np.array([[ self.LENGTH/2,-self.LENGTH/2],
                                       [-self.LENGTH/2,-self.LENGTH/2],
                                       [-self.LENGTH/2, self.LENGTH/2],
                                       [ self.LENGTH/2, self.LENGTH/2]]).T    
        # Rotation matrix (body -> inertial)
        C_Ib_target = self.make_C_bI(self.target_position[-1]).T        
        # Rotating body frame coordinates to inertial frame
        target_body_inertial = np.matmul(C_Ib_target, target_points_body) + np.array([self.target_position[0], self.target_position[1]]).reshape([2,-1])
        target_polygon = Polygon(target_body_inertial.T)
        
        # Forbidden Area
        forbidden_area_body = np.array([[self.LENGTH/2, self.LENGTH/2],   
                                        [self.DOCKING_PORT_CORNER1_POSITION[0],self.DOCKING_PORT_CORNER1_POSITION[1]],
                                        [self.DOCKING_PORT_MOUNT_POSITION[0],self.DOCKING_PORT_MOUNT_POSITION[1]],
                                        [self.DOCKING_PORT_CORNER2_POSITION[0],self.DOCKING_PORT_CORNER2_POSITION[1]],
                                        [-self.LENGTH/2,self.LENGTH/2],
                                        [self.LENGTH/2, self.LENGTH/2]]).T        
        # Rotating body frame coordinates to inertial frame
        forbidden_area_inertial = np.matmul(C_Ib_target, forbidden_area_body) + np.array([self.target_position[0], self.target_position[1]]).reshape([2,-1])         
        forbidden_polygon = Polygon(forbidden_area_inertial.T)
        
        # End-effector
        end_effector_point = Point(self.end_effector_position)
        
        # Chaser
        chaser_points_body = np.array([[ self.LENGTH/2,-self.LENGTH/2],
                                       [-self.LENGTH/2,-self.LENGTH/2],
                                       [-self.LENGTH/2, self.LENGTH/2],
                                       [ self.LENGTH/2, self.LENGTH/2]]).T    
        # Rotation matrix (body -> inertial)
        C_Ib_chaser = self.make_C_bI(self.chaser_position[-1]).T        
        # Rotating body frame coordinates to inertial frame
        chaser_body_inertial = np.matmul(C_Ib_chaser, chaser_points_body) + np.array([self.chaser_position[0], self.chaser_position[1]]).reshape([2,-1])
        chaser_polygon = Polygon(chaser_body_inertial.T)
        
        # Elbow position in the inertial frame
        elbow_point = Point(self.elbow_position)
        
        ###########################
        ### Checking collisions ###
        ###########################
        self.end_effector_collision = False
        self.forbidden_area_collision = False
        self.chaser_target_collision = False
        self.mid_way = False        
        self.docked = False        
        self.elbow_target_collision = False
        
        if self.CHECK_END_EFFECTOR_COLLISION and end_effector_point.within(target_polygon):
            if self.test_time and self.extra_printing:
                print("End-effector colliding with the target!")
            self.end_effector_collision = True
        
        if self.CHECK_END_EFFECTOR_FORBIDDEN and end_effector_point.within(forbidden_polygon):
            if self.test_time and self.extra_printing:
                print("End-effector within the forbidden area!")
            self.forbidden_area_collision = True
        
        if self.CHECK_CHASER_TARGET_COLLISION and chaser_polygon.intersects(target_polygon):
            if self.test_time and self.extra_printing:
                print("Chaser/target collision")
            self.chaser_target_collision = True
        
        # Elbow can be within the forbidden area
        if self.CHECK_END_EFFECTOR_COLLISION and elbow_point.within(target_polygon):
            if self.test_time and self.extra_printing:
                print("Elbow/target collision!")
            self.elbow_target_collision = True
        
        ##########################
        ### Mid-way or docked? ###
        ##########################
        # Docking Polygon (circle)
        docking_circle = Point(self.target_position[:-1] + np.matmul(C_Ib_target, self.DOCKING_PORT_MOUNT_POSITION)).buffer(self.SUCCESSFUL_DOCKING_RADIUS)
        
        # Mid-way Polygon (circle)
        mid_way_circle = Point(self.target_position[:-1] + np.matmul(C_Ib_target, self.DOCKING_PORT_MOUNT_POSITION)).buffer(self.MID_WAY_REWARD_RADIUS)
        
        if self.GIVE_MID_WAY_REWARD and self.not_yet_mid_way and end_effector_point.within(mid_way_circle):
            if self.test_time and self.extra_printing:
                print("Mid Way!")
            self.mid_way = True
        
        if end_effector_point.within(docking_circle):
            if self.test_time and self.extra_printing:
                print("Docked!")
            self.docked = True
            
        ######################################
        ### Checking if chaser in on table ###
        ######################################
        self.chaser_on_table = chaser_polygon.within(self.TABLE_BOUNDARY)                                                            


    def is_done(self):
        # Checks if this episode is done or not
        """
            NOTE: THE ENVIRONMENT MUST RETURN done = True IF THE EPISODE HAS
                  REACHED ITS LAST TIMESTEP
        """

        # If we've docked with the target
        if self.docked:
            return True

        # If we've fallen off the table or spun too many times, end the episode
        if (not(self.chaser_on_table) or np.abs(self.chaser_position[-1]) > 6*np.pi) and self.END_ON_FALL:
            if self.test_time and self.extra_printing:
                print("Fell off table!")
            return True

        # If we want to end the episode during a collision
        if self.END_ON_COLLISION and np.any([self.end_effector_collision, self.forbidden_area_collision, self.chaser_target_collision, self.elbow_target_collision]):
            if self.test_time and self.extra_printing:
                print("Ending episode due to a collision")
            return True
        
        # If we want to end when an arm segment reaches its limit
        if self.END_ON_ARM_LIMITS and np.any(self.joints_past_limits):
            if self.test_time and self.extra_printing:
                print("Ending episode due to arm limits being reached")
            return True
        
        # If we've run out of timesteps
        if round(self.time/self.TIMESTEP) == self.MAX_NUMBER_OF_TIMESTEPS:
            return True
        
        # The episode must not be done!
        return False


    def generate_queue(self):
        # Generate the queues responsible for communicating with the agent
        self.agent_to_env = multiprocessing.Queue(maxsize = 1)
        self.env_to_agent = multiprocessing.Queue(maxsize = 1)

        return self.agent_to_env, self.env_to_agent

    
    def make_C_bI(self, angle):
        
        C_bI = np.array([[ np.cos(angle), np.sin(angle)],
                         [-np.sin(angle), np.cos(angle)]]) # [2, 2]        
        return C_bI


    def run(self):
        ###################################
        ##### Running the environment #####
        ###################################
        """
        This method is called when the environment process is launched by main.py.
        It is responsible for continually listening for an input action from the
        agent through a Queue. If an action is received, it is to step the environment
        and return the results.
        
        TOTAL_STATE_SIZE = 29 # [chaser_x, chaser_y, chaser_theta, chaser_x_dot, chaser_y_dot, chaser_theta_dot, 
        shoulder_theta, elbow_theta, wrist_theta, shoulder_theta_dot, elbow_theta_dot, wrist_theta_dot, 
        target_x, target_y, target_theta, target_x_dot, target_y_dot, target_theta_dot, 
        ee_x_I, ee_y_I, ee_x_dot_I, ee_y_dot_I, relative_x_b, relative_y_b, relative_theta, 
        ee_x_b, ee_y_b, ee_x_dot_b, ee_y_dot_b]
            
        The positions are in the inertial frame but the manipulator angles are in the joint frame.
            
        """
        # Instructing this process to treat Ctrl+C events (called SIGINT) by going SIG_IGN (ignore).
        # This permits the process to continue upon a Ctrl+C event to allow for graceful quitting.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        # Loop until the process is terminated
        while True:
            # Blocks until the agent passes us an action
            action, *test_time = self.agent_to_env.get()


            if type(action) == bool and action == True:
                # The signal to reset the environment was received
                self.reset(test_time[0])
                
                # Return the TOTAL_STATE
                self.env_to_agent.put(self.make_total_state())
                
            elif type(action) == bool and action == False:
                # A signal to return if we docked, the target angular rate, and the combined angular momentum was received
                self.env_to_agent.put((self.docked, self.target_velocity[-1]*180/np.pi, self.combined_angular_momentum()))

            else:
                
                # Delay the action by DYNAMICS_DELAY timesteps. The environment accumulates the action delay--the agent still thinks the sent action was used.
                if self.DYNAMICS_DELAY > 0:
                    self.action_delay_queue.put(action,False) # puts the current action to the bottom of the stack
                    action = self.action_delay_queue.get(False) # grabs the delayed action and treats it as truth.               
                    
                # Rotating the [linear acceleration] action from the body frame into the inertial frame only if it is appropriate to do so
                if not self.ACTIONS_IN_INERTIAL:
                    action[0:2] = np.matmul(self.make_C_bI(self.chaser_position[-1]).T, action[0:2])

                ################################
                ##### Step the environment #####
                ################################ 
                reward, done = self.step(action)

                # Return (TOTAL_STATE, reward, done)
                self.env_to_agent.put((self.make_total_state(), reward, done))


#####################################################################
##### Generating the dynamics equations representing the motion #####
#####################################################################
def dynamics_equations_of_motion(chaser_state, t, parameters):
    # chaser_state = [self.chaser_position, self.arm_angles, self.chaser_velocity, self.arm_angular_rates]

    # Unpacking the chaser properties from the chaser_state
    x, y, theta, theta_1, theta_2, theta_3, x_dot, y_dot, theta_dot, theta_1_dot, theta_2_dot, theta_3_dot = chaser_state
    
    # state = x, y, theta, theta_1, theta_2, theta_3
    state_dot = np.array([x_dot, y_dot, theta_dot, theta_1_dot, theta_2_dot, theta_3_dot]).reshape([6,1])

    control_effort, LENGTH, PHI, B0, \
    MASS, M1, M2, M3, \
    A1, B1, A2, B2, A3, B3, \
    INERTIA, INERTIA1, INERTIA2, INERTIA3 = parameters # Unpacking parameters

    # Generate the mass matrix for this state
    MassMatrix = calculate_mass_matrix(chaser_state, t, parameters)
    
    # Generate the coriolis matrix for this state
    CoriolisMatrix = calculate_coriolis_matrix(chaser_state, t, parameters)
    
    second_derivatives = np.matmul(np.linalg.inv(MassMatrix),(control_effort - np.matmul(CoriolisMatrix, state_dot)))

    first_derivatives = np.array([x_dot, y_dot, theta_dot, theta_1_dot, theta_2_dot, theta_3_dot]).reshape([6,1])
    
    full_derivative = np.concatenate([first_derivatives, second_derivatives]).squeeze()
    
    return full_derivative


def calculate_mass_matrix(chaser_state, t, parameters):
    # Unpacking the chaser properties from the chaser_state
    x, y, theta, theta_1, theta_2, theta_3, x_dot, y_dot, theta_dot, theta_1_dot, theta_2_dot, theta_3_dot = chaser_state

    control_effort, LENGTH, PHI, B0, \
    MASS, M1, M2, M3, \
    A1, B1, A2, B2, A3, B3, \
    INERTIA, INERTIA1, INERTIA2, INERTIA3 = parameters # Unpacking parameters

    # Generating mass matrix using Alex's equations from InertiaFinc3LINK
    t2 = A1+B1
    t3 = A1*M1
    t4 = M2*t2
    t5 = M3*t2
    t6 = theta+theta_1
    t7 = np.cos(t6)
    t8 = t3+t4+t5
    t9 = A2*M2
    t10 = A2+B2
    t11 = M3*t10
    t12 = theta+theta_1+theta_2
    t13 = np.cos(t12)
    t14 = t9+t11
    t15 = theta+theta_1+theta_2+theta_3
    t16 = np.cos(t15)
    t17 = MASS+M1+M2+M3
    t18 = B0*M1
    t19 = B0*M2
    t20 = B0*M3
    t21 = t18+t19+t20
    t22 = PHI+theta
    t23 = np.sin(t6)
    t24 = np.sin(t12)
    t25 = np.sin(t15)
    t26 = A2*M3
    t27 = B2*M3
    t28 = t9+t26+t27
    t29 = np.sin(t22)
    t86 = t7*t8
    t87 = t13*t14
    t88 = A3*M3*t16
    t30 = -t86-t87-t88-t21*t29
    t31 = np.cos(t22)
    t32 = t21*t31
    t89 = t8*t23
    t90 = t14*t24
    t91 = A3*M3*t25
    t33 = t32-t89-t90-t91
    t34 = A1**2
    t35 = A2**2
    t36 = B0**2
    t37 = B1**2
    t38 = A1*A2*M2*2.0
    t39 = A1*A2*M3*2.0
    t40 = A2*B1*M2*2.0
    t41 = A1*B2*M3*2.0
    t42 = A2*B1*M3*2.0
    t43 = B1*B2*M3*2.0
    t44 = t38+t39+t40+t41+t42+t43
    t45 = np.cos(theta_2)
    t46 = t44*t45
    t47 = A2*A3*M3*2.0
    t48 = A3*B2*M3*2.0
    t49 = t47+t48
    t50 = np.cos(theta_3)
    t51 = t49*t50
    t52 = A1*A3*M3*2.0
    t53 = A3*B1*M3*2.0
    t54 = t52+t53
    t55 = theta_2+theta_3
    t56 = np.cos(t55)
    t57 = t54*t56
    t58 = PHI-theta_1
    t59 = np.sin(t58)
    t60 = -PHI+theta_1+theta_2
    t61 = np.sin(t60)
    t62 = -PHI+theta_1+theta_2+theta_3
    t63 = np.sin(t62)
    t64 = M1*t34
    t65 = M2*t34
    t66 = M3*t34
    t67 = M2*t35
    t68 = M3*t35
    t69 = A3**2
    t70 = M3*t69
    t71 = M2*t37
    t72 = M3*t37
    t73 = B2**2
    t74 = M3*t73
    t75 = A1*B1*M2*2.0
    t76 = A1*B1*M3*2.0
    t77 = A2*B2*M3*2.0
    t78 = A2*B0*M2
    t79 = A2*B0*M3
    t80 = B0*B2*M3
    t81 = t78+t79+t80
    t82 = A1*A3*M3
    t83 = A3*B1*M3
    t84 = t82+t83
    t85 = t56*t84
    t92 = A1*B0*M1
    t93 = A1*B0*M2
    t94 = A1*B0*M3
    t95 = B0*B1*M2
    t96 = B0*B1*M3
    t97 = t92+t93+t94+t95+t96
    t98 = t59*t97
    t112 = t61*t81
    t113 = A3*B0*M3*t63
    t99 = INERTIA1+INERTIA2+INERTIA3+t46+t51+t57+t64+t65+t66+t67+t68+t70+t71+t72+t74+t75+t76+t77+t98-t112-t113
    t100 = A1*A2*M2
    t101 = A1*A2*M3
    t102 = A2*B1*M2
    t103 = A1*B2*M3
    t104 = A2*B1*M3
    t105 = B1*B2*M3
    t106 = t100+t101+t102+t103+t104+t105
    t107 = t45*t106
    t108 = A2*A3*M3
    t109 = A3*B2*M3
    t110 = t108+t109
    t111 = t50*t110
    t114 = INERTIA2+INERTIA3+t51+t67+t68+t70+t74+t77+t85+t107
    t115 = INERTIA3+t70+t85+t111
    t116 = INERTIA3+t70+t111
    MassMatrix = np.array([t17,0.0,t30,-t86-t87-t88,-t88-t13*t28,-t88,
                           0.0,t17,t33,-t89-t90-t91,-t91-t24*t28,-t91,
                           t30,t33,INERTIA+INERTIA1+INERTIA2+INERTIA3+t46+t51+t57+t64+t65+t66+t67+t68+t70+t71+t72+t74+t75+t76+t77+t59*(A1*B0*M1*2.0+A1*B0*M2*2.0+A1*B0*M3*2.0+B0*B1*M2*2.0+B0*B1*M3*2.0)+M1*t36+M2*t36+M3*t36-t61*(A2*B0*M2*2.0+A2*B0*M3*2.0+B0*B2*M3*2.0)-A3*B0*M3*t63*2.0,t99,INERTIA2+INERTIA3+t51+t67+t68+t70+t74+t77+t85+t107-t112-t113,INERTIA3+t70+t85+t111-t113,
                           -t7*t8-t13*t14-A3*M3*t16,-t8*t23-t14*t24-A3*M3*t25,t99,INERTIA1+INERTIA2+INERTIA3+t46+t51+t57+t64+t65+t66+t67+t68+t70+t71+t72+t74+t75+t76+t77,t114,t115,
                           -t13*t28-A3*M3*t16,-t24*t28-A3*M3*t25,INERTIA2+INERTIA3+t51+t67+t68+t70+t74+t77+t85+t107-t61*t81-A3*B0*M3*t63,t114,INERTIA2+INERTIA3+t51+t67+t68+t70+t74+t77,t116,
                           -A3*M3*t16,-A3*M3*t25,INERTIA3+t70+t85+t111-A3*B0*M3*t63,t115,t116,INERTIA3+t70]).reshape([6,6], order ='F') # default order is different from matlab
    return MassMatrix


def calculate_coriolis_matrix(chaser_state, t, parameters):
    # Unpacking the chaser properties from the chaser_state
    x, y, theta, theta_1, theta_2, theta_3, x_dot, y_dot, theta_dot, theta_1_dot, theta_2_dot, theta_3_dot = chaser_state

    control_effort, LENGTH, PHI, B0, \
    MASS, M1, M2, M3, \
    A1, B1, A2, B2, A3, B3, \
    INERTIA, INERTIA1, INERTIA2, INERTIA3 = parameters # Unpacking parameters

    # Generating coriolis matrix using Alex's equations from CoriolisFinc3LINK
    t2 = A1+B1
    t3 = A1*M1
    t4 = M2*t2
    t5 = M3*t2
    t6 = t3+t4+t5
    t7 = A2*M2
    t8 = A2+B2
    t9 = M3*t8
    t10 = t7+t9
    t11 = np.pi*(10/20)
    t12 = theta_dot*t6
    t13 = theta_1_dot*t6
    t14 = theta+theta_1+t11
    t15 = np.cos(t14)
    t16 = t12+t13
    t17 = theta_dot*t10
    t18 = theta_1_dot*t10
    t19 = theta_2_dot*t10
    t20 = theta+theta_1+theta_2+t11
    t21 = np.cos(t20)
    t22 = t17+t18+t19
    t23 = A3*M3*theta_dot
    t24 = A3*M3*theta_1_dot
    t25 = A3*M3*theta_2_dot
    t26 = A3*M3*theta_3_dot
    t27 = theta+theta_1+theta_2+theta_3+t11
    t28 = np.cos(t27)
    t29 = t23+t24+t25+t26
    t30 = B0*M1
    t31 = B0*M2
    t32 = B0*M3
    t33 = t30+t31+t32
    t34 = PHI+theta
    t35 = np.sin(t14)
    t36 = np.sin(t20)
    t37 = np.sin(t27)
    t38 = theta_dot+theta_1_dot+theta_2_dot+theta_3_dot
    t39 = theta+theta_1+theta_2+theta_3
    t40 = A1*B0*M1*theta_1_dot
    t41 = A1*B0*M2*theta_1_dot
    t42 = A1*B0*M3*theta_1_dot
    t43 = B0*B1*M2*theta_1_dot
    t44 = B0*B1*M3*theta_1_dot
    t45 = PHI-theta_1
    t46 = np.cos(t45)
    t47 = A2*B0*M2*theta_1_dot
    t48 = A2*B0*M2*theta_2_dot
    t49 = A2*B0*M3*theta_1_dot
    t50 = A2*B0*M3*theta_2_dot
    t51 = B0*B2*M3*theta_1_dot
    t52 = B0*B2*M3*theta_2_dot
    t53 = -PHI+theta_1+theta_2
    t54 = np.cos(t53)
    t55 = A3*B0*M3*theta_1_dot
    t56 = A3*B0*M3*theta_2_dot
    t57 = A3*B0*M3*theta_3_dot
    t58 = -PHI+theta_1+theta_2+theta_3
    t59 = np.cos(t58)
    t60 = A2*B1*M2*theta_2_dot
    t61 = A1*B2*M3*theta_2_dot
    t62 = A2*B1*M3*theta_2_dot
    t63 = B1*B2*M3*theta_2_dot
    t64 = A1*A2*M2*theta_2_dot
    t65 = A1*A2*M3*theta_2_dot
    t66 = np.sin(theta_2)
    t67 = t60+t61+t62+t63+t64+t65
    t68 = A3*B2*M3*theta_3_dot
    t69 = A2*A3*M3*theta_3_dot
    t70 = np.sin(theta_3)
    t71 = t68+t69
    t72 = A3*B1*M3*theta_2_dot
    t73 = A3*B1*M3*theta_3_dot
    t74 = A1*A3*M3*theta_2_dot
    t75 = A1*A3*M3*theta_3_dot
    t76 = theta_2+theta_3
    t77 = np.sin(t76)
    t78 = t72+t73+t74+t75
    t79 = A2*B0*M2*theta_dot
    t80 = A2*B0*M3*theta_dot
    t81 = B0*B2*M3*theta_dot
    t82 = t47+t48+t49+t50+t51+t52+t79+t80+t81
    t83 = A3*B0*M3*theta_dot
    t84 = t55+t56+t57+t83
    t85 = A1*B0*M1*theta_dot
    t86 = A1*B0*M2*theta_dot
    t87 = A1*B0*M3*theta_dot
    t88 = B0*B1*M2*theta_dot
    t89 = B0*B1*M3*theta_dot
    t90 = A2*B1*M2*theta_dot
    t91 = A1*B2*M3*theta_dot
    t92 = A2*B1*M2*theta_1_dot
    t93 = A2*B1*M3*theta_dot
    t94 = A1*B2*M3*theta_1_dot
    t95 = A2*B1*M3*theta_1_dot
    t96 = B1*B2*M3*theta_dot
    t97 = B1*B2*M3*theta_1_dot
    t98 = A1*A2*M2*theta_dot
    t99 = A1*A2*M2*theta_1_dot
    t100 = A1*A2*M3*theta_dot
    t101 = A1*A2*M3*theta_1_dot
    t102 = t60+t61+t62+t63+t64+t65+t90+t91+t92+t93+t94+t95+t96+t97+t98+t99+t100+t101
    t103 = A3*B1*M3*theta_dot
    t104 = A3*B1*M3*theta_1_dot
    t105 = A1*A3*M3*theta_dot
    t106 = A1*A3*M3*theta_1_dot
    t107 = t72+t73+t74+t75+t103+t104+t105+t106
    t108 = t79+t80+t81
    t109 = t54*t108
    t110 = t59*t83
    t111 = t90+t91+t92+t93+t94+t95+t96+t97+t98+t99+t100+t101
    t112 = t66*t111
    t113 = t103+t104+t105+t106
    t114 = t77*t113
    t115 = A2*theta_dot
    t116 = A2*theta_1_dot
    t117 = A2*theta_2_dot
    t118 = B2*theta_dot
    t119 = B2*theta_1_dot
    t120 = B2*theta_2_dot
    t121 = t115+t116+t117+t118+t119+t120
    t122 = A3*M3*t70*t121
    t123 = A1*theta_dot
    t124 = A1*theta_1_dot
    t125 = B1*theta_dot
    t126 = B1*theta_1_dot
    t127 = t123+t124+t125+t126
    t128 = A3*M3*t77*t127

    # Assembling the matrix
    CoriolisMatrix = np.array([0.0,0.0,0.0,0.0,0.0,0.0,
                               0.0,0.0,0.0,0.0,0.0,0.0,
                               -t15*t16-t21*t22-t28*t29-theta_dot*t33*np.cos(t34),-t16*t35-t22*t36-t29*t37-theta_dot*t33*np.sin(t34),-t66*t67-t70*t71-t77*t78-t46*(t40+t41+t42+t43+t44)-t59*(t55+t56+t57)-t54*(t47+t48+t49+t50+t51+t52),t109+t110-t66*t67-t70*t71-t77*t78+t46*(t85+t86+t87+t88+t89),t109+t110+t112+t114-t70*t71,t110+t122+t128,
                               -t15*t16-t21*t22-t28*t29,-t16*t35-t22*t36-t29*t37,-t66*t67-t54*t82-t70*t71-t59*t84-t77*t78-t46*(t40+t41+t42+t43+t44+t85+t86+t87+t88+t89),-t66*t67-t70*t71-t77*t78,t112+t114-t70*t71,t122+t128,
                               -t21*t22-t28*t29,-t22*t36-t29*t37,-t54*t82-t70*t71-t59*t84-t66*t102-t77*t107,-t70*t71-t66*t102-t77*t107,-A3*M3*theta_3_dot*t8*t70,A3*M3*t8*t70*(theta_dot+theta_1_dot+theta_2_dot),
                               A3*M3*t38*np.sin(t39),-A3*M3*t38*np.cos(t39),-A3*B0*M3*t38*t59-A3*M3*t8*t38*t70-A3*M3*t2*t38*t77,-A3*M3*t8*t38*t70-A3*M3*t2*t38*t77,-A3*M3*t8*t38*t70,00]).reshape([6,6], order='F') # default order is different than matlab
    return CoriolisMatrix


##########################################
##### Function to animate the motion #####
##########################################
def render(states, actions, instantaneous_reward_log, cumulative_reward_log, critic_distributions, target_critic_distributions, projected_target_distribution, bins, loss_log, episode_number, filename, save_directory, time_log, timestep_where_docking_occurred = -1):

    # Load in a temporary environment, used to grab the physical parameters
    temp_env = Environment()
    temp_env.reset(False)

    # Checking if we want the additional reward and value distribution information
    extra_information = temp_env.ADDITIONAL_VALUE_INFO

    # Unpacking state from TOTAL_STATE 
    """
    [chaser_x, chaser_y, chaser_theta, chaser_x_dot, chaser_y_dot, chaser_theta_dot, 
     shoulder_theta, elbow_theta, wrist_theta, shoulder_theta_dot, elbow_theta_dot, wrist_theta_dot, 
     target_x, target_y, target_theta, target_x_dot, target_y_dot, target_theta_dot, ee_x_I, ee_y_I, ee_x_dot_I, ee_y_dot_I, relative_x_b, relative_y_b, relative_theta, ee_x_b, ee_y_b, ee_x_dot_b, ee_y_dot_b]
    """
    # Chaser positions
    chaser_x, chaser_y, chaser_theta, theta_1, theta_2, theta_3 = states[:,0], states[:,1], states[:,2], states[:,6], states[:,7], states[:,8]
    
    # Target positions
    target_x, target_y, target_theta = states[:,12], states[:,13], states[:,14]
    
    # Target initial angular velocity
    target_initial_omega = states[0,17]*180/np.pi
    
    # Chaser final velocities
    chaser_final_vx, chaser_final_vy, chaser_final_omega = states[-1,3], states[-1,4], states[-1,5]
    
    # Manipulator final angular velocities
    shoulder_final_theta_dot, elbow_final_theta_dot, wrist_final_theta_dot = states[-1,9], states[-1,10], states[-1,11]
    
    # Target final velocities
    target_final_vx, target_final_vy, target_final_omega = states[-1,15], states[-1,16], states[-1,17]

    # Extracting physical properties
    LENGTH = temp_env.LENGTH
    PHI    = temp_env.PHI
    B0     = temp_env.B0
    A1     = temp_env.A1
    B1     = temp_env.B1
    A2     = temp_env.A2
    B2     = temp_env.B2
    A3     = temp_env.A3
    B3     = temp_env.B3
    DOCKING_PORT_MOUNT_POSITION = temp_env.DOCKING_PORT_MOUNT_POSITION
    DOCKING_PORT_CORNER1_POSITION = temp_env.DOCKING_PORT_CORNER1_POSITION
    DOCKING_PORT_CORNER2_POSITION = temp_env.DOCKING_PORT_CORNER2_POSITION
    

    #################################################
    ### Calculating chaser locations through time ###
    #################################################
    
    ##############################################
    ### Manipulator Joint Locations (Inertial) ###
    ##############################################
    # Shoulder
    shoulder_x = chaser_x + B0*np.cos(chaser_theta + PHI)
    shoulder_y = chaser_y + B0*np.sin(chaser_theta + PHI)

    # Elbow
    elbow_x = shoulder_x + (A1 + B1)*np.cos(np.pi/2 + chaser_theta + theta_1)
    elbow_y = shoulder_y + (A1 + B1)*np.sin(np.pi/2 + chaser_theta + theta_1)

    # Wrist
    wrist_x = elbow_x + (A2 + B2)*np.cos(np.pi/2 + chaser_theta + theta_1 + theta_2)
    wrist_y = elbow_y + (A2 + B2)*np.sin(np.pi/2 + chaser_theta + theta_1 + theta_2)

    # End-effector
    end_effector_x = wrist_x + (A3 + B3)*np.cos(np.pi/2 + chaser_theta + theta_1 + theta_2 + theta_3)
    end_effector_y = wrist_y + (A3 + B3)*np.sin(np.pi/2 + chaser_theta + theta_1 + theta_2 + theta_3)

    ###############################
    ### Chaser corner locations ###
    ###############################

    # All the points to draw of the chaser (except the front-face)    
    chaser_points_body = np.array([[ LENGTH/2,-LENGTH/2],
                                   [-LENGTH/2,-LENGTH/2],
                                   [-LENGTH/2, LENGTH/2],
                                   [ LENGTH/2, LENGTH/2]]).T
    
    # The front-face points on the target
    chaser_front_face_body = np.array([[[ LENGTH/2],[ LENGTH/2]],
                                       [[ LENGTH/2],[-LENGTH/2]]]).squeeze().T

    # Rotation matrix (body -> inertial)
    C_Ib_chaser = np.moveaxis(np.array([[np.cos(chaser_theta), -np.sin(chaser_theta)],
                                        [np.sin(chaser_theta),  np.cos(chaser_theta)]]), source = 2, destination = 0) # [NUM_TIMESTEPS, 2, 2]
    
    # Rotating body frame coordinates to inertial frame    
    chaser_body_inertial       = np.matmul(C_Ib_chaser, chaser_points_body)     + np.array([chaser_x, chaser_y]).T.reshape([-1,2,1])
    chaser_front_face_inertial = np.matmul(C_Ib_chaser, chaser_front_face_body) + np.array([chaser_x, chaser_y]).T.reshape([-1,2,1])

    #############################
    ### Target Body Locations ###
    #############################
    # All the points to draw of the target (except the front-face)     
    target_points_body = np.array([[ LENGTH/2,-LENGTH/2],
                                   [-LENGTH/2,-LENGTH/2],
                                   [-LENGTH/2, LENGTH/2],
                                   [ LENGTH/2, LENGTH/2],
                                   [DOCKING_PORT_MOUNT_POSITION[0], LENGTH/2], # artificially adding this to make the docking cone look better 
                                   [DOCKING_PORT_MOUNT_POSITION[0],DOCKING_PORT_MOUNT_POSITION[1]],
                                   [DOCKING_PORT_CORNER1_POSITION[0],DOCKING_PORT_CORNER1_POSITION[1]],
                                   [DOCKING_PORT_CORNER2_POSITION[0],DOCKING_PORT_CORNER2_POSITION[1]],
                                   [DOCKING_PORT_MOUNT_POSITION[0],DOCKING_PORT_MOUNT_POSITION[1]]]).T
    
    # The front-face points on the target
    target_front_face_body = np.array([[[ LENGTH/2],[ LENGTH/2]],
                                       [[ LENGTH/2],[-LENGTH/2]]]).squeeze().T

    # Rotation matrix (body -> inertial)
    C_Ib_target = np.moveaxis(np.array([[np.cos(target_theta), -np.sin(target_theta)],
                                        [np.sin(target_theta),  np.cos(target_theta)]]), source = 2, destination = 0) # [NUM_TIMESTEPS, 2, 2]
    
    # Rotating body frame coordinates to inertial frame
    target_body_inertial       = np.matmul(C_Ib_target, target_points_body)     + np.array([target_x, target_y]).T.reshape([-1,2,1])
    target_front_face_inertial = np.matmul(C_Ib_target, target_front_face_body) + np.array([target_x, target_y]).T.reshape([-1,2,1])
    
    # Calculating the accelerations for each state through time
    velocities = np.concatenate([states[:,3:6],states[:,9:12]], axis = 1) # Velocities measured in the inertial frame
    # Numerically differentiating to approximate the derivative
    accelerations = np.diff(velocities, axis = 0)/temp_env.TIMESTEP
    # Add a row of zeros initially to the current acceleartions
    accelerations = np.concatenate([np.zeros([1,temp_env.ACTION_SIZE]), accelerations])
        
    # Adding a row of zeros to the actions for the first timestep
    actions = np.concatenate([np.zeros([1,temp_env.ACTION_SIZE]), actions])
    
    # Calculating the final combined angular momentum
    temp_env.chaser_position = np.array([chaser_x[timestep_where_docking_occurred], chaser_y[timestep_where_docking_occurred], chaser_theta[timestep_where_docking_occurred]])
    temp_env.arm_angles = np.array([theta_1[timestep_where_docking_occurred], theta_2[timestep_where_docking_occurred], theta_3[timestep_where_docking_occurred]])
    temp_env.chaser_velocity = np.array([chaser_final_vx, chaser_final_vy, chaser_final_omega])
    temp_env.arm_angular_rates = np.array([shoulder_final_theta_dot, elbow_final_theta_dot, wrist_final_theta_dot])
    temp_env.target_position = np.array([target_x[timestep_where_docking_occurred], target_y[timestep_where_docking_occurred], target_theta[timestep_where_docking_occurred]])
    temp_env.target_velocity = np.array([target_final_vx, target_final_vy, target_final_omega])
    temp_env.update_end_effector_and_docking_locations()
    temp_env.update_end_effector_location_body_frame()
    temp_env.update_relative_pose_body_frame()
    temp_env.check_collisions()
    # Check if we docked
    docked = temp_env.docked
    
    combined_total_angular_momentum, combined_angular_velocity = temp_env.combined_angular_momentum()

    #######################
    ### Plotting Motion ###
    #######################
    
    # Generating figure window
    figure = plt.figure(constrained_layout = True)
    figure.set_size_inches(5, 4, True)

    if extra_information:
        grid_spec = gridspec.GridSpec(nrows = 2, ncols = 3, figure = figure)
        subfig1 = figure.add_subplot(grid_spec[0,0], aspect = 'equal', autoscale_on = False, xlim = (0, 3.5), ylim = (0, 2.4))
        subfig2 = figure.add_subplot(grid_spec[0,1], xlim = (np.min([np.min(instantaneous_reward_log), 0]) - (np.max(instantaneous_reward_log) - np.min(instantaneous_reward_log))*0.02, np.max([np.max(instantaneous_reward_log), 0]) + (np.max(instantaneous_reward_log) - np.min(instantaneous_reward_log))*0.02), ylim = (-0.5, 0.5))
        subfig3 = figure.add_subplot(grid_spec[0,2], xlim = (np.min(loss_log)-0.01, np.max(loss_log)+0.01), ylim = (-0.5, 0.5))
        subfig4 = figure.add_subplot(grid_spec[1,0], ylim = (0, 1.02))
        subfig5 = figure.add_subplot(grid_spec[1,1], ylim = (0, 1.02))
        subfig6 = figure.add_subplot(grid_spec[1,2], ylim = (0, 1.02))

        # Setting titles
        subfig1.set_xlabel("X Position (m)",    fontdict = {'fontsize': 8})
        subfig1.set_ylabel("Y Position (m)",    fontdict = {'fontsize': 8})
        subfig2.set_title("Timestep Reward",    fontdict = {'fontsize': 8})
        subfig3.set_title("Current loss",       fontdict = {'fontsize': 8})
        subfig4.set_title("Q-dist",             fontdict = {'fontsize': 8})
        subfig5.set_title("Target Q-dist",      fontdict = {'fontsize': 8})
        subfig6.set_title("Bellman projection", fontdict = {'fontsize': 8})

        # Changing around the axes
        subfig1.tick_params(labelsize = 8)
        subfig2.tick_params(which = 'both', left = False, labelleft = False, labelsize = 8)
        subfig3.tick_params(which = 'both', left = False, labelleft = False, labelsize = 8)
        subfig4.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = False, labelsize = 8)
        subfig5.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = False, labelsize = 8)
        subfig6.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = True, labelsize = 8)

        # Adding the grid
        subfig4.grid(True)
        subfig5.grid(True)
        subfig6.grid(True)

        # Setting appropriate axes ticks
        subfig2.set_xticks([np.min(instantaneous_reward_log), 0, np.max(instantaneous_reward_log)] if np.sign(np.min(instantaneous_reward_log)) != np.sign(np.max(instantaneous_reward_log)) else [np.min(instantaneous_reward_log), np.max(instantaneous_reward_log)])
        subfig3.set_xticks([np.min(loss_log), np.max(loss_log)])
        subfig4.set_xticks([bins[i*5] for i in range(round(len(bins)/5) + 1)])
        subfig4.tick_params(axis = 'x', labelrotation = -90)
        subfig4.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
        subfig5.set_xticks([bins[i*5] for i in range(round(len(bins)/5) + 1)])
        subfig5.tick_params(axis = 'x', labelrotation = -90)
        subfig5.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
        subfig6.set_xticks([bins[i*5] for i in range(round(len(bins)/5) + 1)])
        subfig6.tick_params(axis = 'x', labelrotation = -90)
        subfig6.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])

    else:
        subfig1 = figure.add_subplot(1, 1, 1, aspect = 'equal', autoscale_on = False, xlim = (0, temp_env.MAX_X_POSITION), ylim = (0, temp_env.MAX_Y_POSITION), xlabel = 'X Position (m)', ylabel = 'Y Position (m)')

    # Defining plotting objects that change each frame
    chaser_body,       = subfig1.plot([], [], color = 'r', linestyle = '-', linewidth = 2) # Note, the comma is needed
    chaser_front_face, = subfig1.plot([], [], color = 'k', linestyle = '-', linewidth = 2) # Note, the comma is needed
    target_body,       = subfig1.plot([], [], color = 'g', linestyle = '-', linewidth = 2)
    target_front_face, = subfig1.plot([], [], color = 'k', linestyle = '-', linewidth = 2)
    manipulator,       = subfig1.plot([], [], color = 'r', linestyle = '-', linewidth = 2) # Note, the comma is needed

    if extra_information:
        reward_bar           = subfig2.barh(y = 0, height = 0.2, width = 0)
        loss_bar             = subfig3.barh(y = 0, height = 0.2, width = 0)
        q_dist_bar           = subfig4.bar(x = bins, height = np.zeros(shape = len(bins)), width = bins[1]-bins[0])
        target_q_dist_bar    = subfig5.bar(x = bins, height = np.zeros(shape = len(bins)), width = bins[1]-bins[0])
        projected_q_dist_bar = subfig6.bar(x = bins, height = np.zeros(shape = len(bins)), width = bins[1]-bins[0])
        time_text            = subfig1.text(x = 0.2, y = 0.91, s = '', fontsize = 8, transform=subfig1.transAxes)
        reward_text          = subfig1.text(x = 0.0,  y = 1.02, s = '', fontsize = 8, transform=subfig1.transAxes)
    else:
        time_text         = subfig1.text(x = 0.03, y = 0.96, s = '', fontsize = 8, transform=subfig1.transAxes)
        reward_text       = subfig1.text(x = 0.62, y = 0.96, s = '', fontsize = 8, transform=subfig1.transAxes)
        angular_rate_text = subfig1.text(x = 0.55, y = 0.90, s = '', fontsize = 8, transform=subfig1.transAxes)
        angular_rate_text.set_text('Target angular rate = %.2f deg/s' %target_initial_omega)
        episode_text      = subfig1.text(x = 0.40, y = 1.02, s = '', fontsize = 8, transform=subfig1.transAxes)
        episode_text.set_text('Episode ' + str(episode_number))
        control1_text     = subfig1.text(x = 0.01, y = 0.90, s = '', fontsize = 6, transform=subfig1.transAxes)
        control2_text     = subfig1.text(x = 0.01, y = 0.85, s = '', fontsize = 6, transform=subfig1.transAxes)
        control3_text     = subfig1.text(x = 0.01, y = 0.80, s = '', fontsize = 6, transform=subfig1.transAxes)
        control4_text     = subfig1.text(x = 0.01, y = 0.75, s = '', fontsize = 6, transform=subfig1.transAxes)
        control5_text     = subfig1.text(x = 0.01, y = 0.70, s = '', fontsize = 6, transform=subfig1.transAxes)
        control6_text     = subfig1.text(x = 0.01, y = 0.65, s = '', fontsize = 6, transform=subfig1.transAxes)
        
        
        


    # Function called repeatedly to draw each frame
    def render_one_frame(frame, *fargs):

        # Draw the chaser body
        chaser_body.set_data(chaser_body_inertial[frame,0,:], chaser_body_inertial[frame,1,:])

        # Draw the front face of the chaser body in a different colour
        chaser_front_face.set_data(chaser_front_face_inertial[frame,0,:], chaser_front_face_inertial[frame,1,:])

        # Draw the target body
        target_body.set_data(target_body_inertial[frame,0,:], target_body_inertial[frame,1,:])

        # Draw the front face of the target body in a different colour
        target_front_face.set_data(target_front_face_inertial[frame,0,:], target_front_face_inertial[frame,1,:])

        # Draw the manipulator
        thisx = [shoulder_x[frame], elbow_x[frame], wrist_x[frame], end_effector_x[frame]]
        thisy = [shoulder_y[frame], elbow_y[frame], wrist_y[frame], end_effector_y[frame]]
        manipulator.set_data(thisx, thisy)

        
        
        # Update the control text
        control1_text.set_text('$\ddot{x}$ = %6.3f; true = %6.3f' %(actions[frame,0], accelerations[frame,0]))
        control2_text.set_text('$\ddot{y}$ = %6.3f; true = %6.3f' %(actions[frame,1], accelerations[frame,1]))
        control3_text.set_text(r'$\ddot{\theta}$ = %1.3f; true = %6.3f' %(actions[frame,2], accelerations[frame,2]))
        control4_text.set_text('$\ddot{q_0}$ = %6.3f; true = %6.3f' %(actions[frame,3], accelerations[frame,3]))
        control5_text.set_text('$\ddot{q_1}$ = %6.3f; true = %6.3f' %(actions[frame,4], accelerations[frame,4]))
        control6_text.set_text('$\ddot{q_2}$ = %6.3f; true = %6.3f' %(actions[frame,5], accelerations[frame,5]))
        
        # Update the reward text
        reward_text.set_text('Total reward = %.1f' %cumulative_reward_log[frame])
        
        # Update the time text
        time_text.set_text('Time = %.1f s' %(time_log[frame]))
        
        # If we're on the last frame, update the angular rate text
        if ((frame == timestep_where_docking_occurred) or (frame == (len(time_log)-1))) and docked:        
            angular_rate_text.set_text('Combined angular rate = %.2f deg/s' %combined_angular_velocity)

        if extra_information:
            # Updating the instantaneous reward bar graph
            reward_bar[0].set_width(instantaneous_reward_log[frame])
            # And colouring it appropriately
            if instantaneous_reward_log[frame] < 0:
                reward_bar[0].set_color('r')
            else:
                reward_bar[0].set_color('g')

            # Updating the loss bar graph
            loss_bar[0].set_width(loss_log[frame])

            # Updating the q-distribution plot
            for this_bar, new_value in zip(q_dist_bar, critic_distributions[frame,:]):
                this_bar.set_height(new_value)

            # Updating the target q-distribution plot
            for this_bar, new_value in zip(target_q_dist_bar, target_critic_distributions[frame, :]):
                this_bar.set_height(new_value)

            # Updating the projected target q-distribution plot
            for this_bar, new_value in zip(projected_q_dist_bar, projected_target_distribution[frame, :]):
                this_bar.set_height(new_value)

        # Since blit = True, must return everything that has changed at this frame
        return time_text, reward_text, chaser_body, chaser_front_face, target_body, target_front_face, manipulator

    # Generate the animation!
    #fargs = [temp_env] # bundling additional arguments
    animator = animation.FuncAnimation(figure, render_one_frame, frames = np.linspace(0, len(states)-1, len(states)).astype(int),
                                       blit = False)#, fargs = fargs)
    """
    frames = the int that is passed to render_one_frame. I use it to selectively plot certain data
    fargs = additional arguments for render_one_frame
    interval = delay between frames in ms
    """

    # Save the animation!
    if temp_env.SKIP_FAILED_ANIMATIONS:
        try:
            if temp_env.ON_CEDAR:
                # Save it to the working directory [have to], then move it to the proper folder
                animator.save(filename = os.environ['SLURM_TMPDIR'] + '/' + 'episode_' + str(episode_number) + '.mp4', fps = 30, dpi = 100)
                # Make directory if it doesn't already exist
                os.makedirs(os.path.dirname(save_directory + filename + '/videos/'), exist_ok=True)
                # Move animation to the proper directory                
                shutil.move(os.environ['SLURM_TMPDIR'] + '/' + 'episode_' + str(episode_number) + '.mp4', save_directory + filename + '/videos/episode_' + str(episode_number) + '.mp4')
                print("Done!")
            else:
                # Save it to the working directory [have to], then move it to the proper folder
                animator.save(filename = filename + '_episode_' + str(episode_number) + '.mp4', fps = 30, dpi = 100)
                # Make directory if it doesn't already exist
                os.makedirs(os.path.dirname(save_directory + filename + '/videos/'), exist_ok=True)
                # Move animation to the proper directory
                os.rename(filename + '_episode_' + str(episode_number) + '.mp4', save_directory + filename + '/videos/episode_' + str(episode_number) + '.mp4')
        except:
            ("Skipping animation for episode %i due to an error" %episode_number)
            # Try to delete the partially completed video file
            try:
                os.remove(filename + '_episode_' + str(episode_number) + '.mp4')
            except:
                pass
    else:
        # Save it to the working directory [have to], then move it to the proper folder
        animator.save(filename = filename + '_episode_' + str(episode_number) + '.mp4', fps = 30, dpi = 100)
        # Make directory if it doesn't already exist
        os.makedirs(os.path.dirname(save_directory + filename + '/videos/'), exist_ok=True)
        # Move animation to the proper directory
        os.rename(filename + '_episode_' + str(episode_number) + '.mp4', save_directory + filename + '/videos/episode_' + str(episode_number) + '.mp4')

    del temp_env
    plt.close(figure)