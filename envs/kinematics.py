import numpy as np


class Kinematics:
    def __init__(self):
        self.L = 0.350 #length of robot
        self.W = 0.188 #width of robot
        self.l1 = 0.09
        self.l2 = 0.162
        self.height = 0.16
        #body frame to coxa frame vector
        self.bodytoFR0 = np.array([ self.L/2, -self.W/2 , 0])
        self.bodytoFL0 = np.array([ self.L/2,  self.W/2 , 0])
        self.bodytoBR0 = np.array([-self.L/2, -self.W/2 , 0])
        self.bodytoBL0 = np.array([-self.L/2,  self.W/2 , 0])
        #body frame to foot frame vector
        self.bodytoFR4 = np.array([ self.L/2, -self.W/2, -self.height])
        self.bodytoFL4 = np.array([ self.L/2, -self.W/2 , -self.height])
        self.bodytoBR4 = np.array([ self.L/2, -self.W/2 , -self.height])
        self.bodytoBL4 = np.array([ self.L/2, -self.W/2 , -self.height])
        self._frames = np.asmatrix([[self.L / 2, -self.W / 2, -self.height],
                                    [self.L / 2, self.W / 2, -self.height],
                                    [-self.L / 2, -self.W / 2, -self.height],
                                    [-self.L / 2, self.W / 2, -self.height]])

    @staticmethod
    def get_Rx(x):
        return np.asmatrix([[1, 0, 0, 0],
                            [0, np.cos(x), -np.sin(x), 0],
                            [0, np.sin(x), np.cos(x), 0],
                            [0, 0, 0, 1]])

    @staticmethod
    def get_Ry(y):
        return np.asmatrix([[np.cos(y), 0, np.sin(y), 0],
                            [0, 1, 0, 0],
                            [-np.sin(y), 0, np.cos(y), 0],
                            [0, 0, 0, 1]])

    @staticmethod
    def get_Rz(z):
        return np.asmatrix([[np.cos(z), -np.sin(z), 0, 0],
                            [np.sin(z), np.cos(z), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

    def get_Rxyz(self, x, y, z):
        if x != 0 or y != 0 or z != 0:
            R = self.get_Rx(x) * self.get_Ry(y) * self.get_Rz(z)
            return R
        else:
            return np.identity(4)

    def get_RT(self, orientation, position):
        roll = orientation[0]
        pitch = orientation[1]
        yaw = orientation[2]
        x0 = position[0]
        y0 = position[1]
        z0 = position[2]

        translation = np.asmatrix([[1, 0, 0, x0],
                                   [0, 1, 0, y0],
                                   [0, 0, 1, z0],
                                   [0, 0, 0, 1]])
        rotation = self.get_Rxyz(roll, pitch, yaw)
        return rotation * translation

    def transform(self, coord, rotation, translation):
        vector = np.array([[coord[0]],
                           [coord[1]],
                           [coord[2]],
                           [1]])

        transform_vector = self.get_RT(rotation, translation) * vector
        return np.array([transform_vector[0, 0], transform_vector[1, 0], transform_vector[2, 0]])

    @staticmethod
    def check_domain(domain):
        if domain > 1 or domain < -1:
            if domain > 1:
                domain = 0.99
            else:
                domain = -0.99
        return domain

    def _solve_IK(self, coord, leg_direction):
        L = np.sqrt(coord[0] ** 2 + coord[2] ** 2)
        cos_param = (self.l1**2 + L**2 - self.l2**2) / (2.0*self.l1*L)
        theta = np.arctan2(leg_direction * coord[0], -coord[2])
        if(cos_param>=1 or cos_param<=-1):
            raise ValueError("cos_param is out of bounds.")
        gamma = np.arccos(cos_param)
        angles = np.array([theta, gamma])
        return angles

    def solve_K(self,angle):
        theta=angle[0]
        gamma=angle[1]
        L=self.l1*np.cos(gamma)+np.sqrt(pow(self.l1*np.cos(gamma),2)-(pow(self.l1,2)-pow(self.l2,2)))
        x=np.sin(theta)*L
        y=np.cos(theta)*L
        return x,y

    def solve(self, orientation, position, frames=None):

        if frames is not None:
            self._frames = frames
        bodytoFR4 = np.asarray([self._frames[0, 0], self._frames[0, 1], self._frames[0, 2]])
        bodytoFL4 = np.asarray([self._frames[1, 0], self._frames[1, 1], self._frames[1, 2]])
        bodytoBR4 = np.asarray([self._frames[2, 0], self._frames[2, 1], self._frames[2, 2]])
        bodytoBL4 = np.asarray([self._frames[3, 0], self._frames[3, 1], self._frames[3, 2]])

        """defines 4 vertices which rotates with the body"""
        _bodytoFR0  = self.transform(self.bodytoFR0, orientation, position)
        _bodytoFL0  = self.transform(self.bodytoFL0, orientation, position)
        _bodytoBR0  = self.transform(self.bodytoBR0, orientation, position)
        _bodytoBL0  = self.transform(self.bodytoBL0, orientation, position)

        """defines coxa_frame to foot_frame leg vector neccesary for IK"""
        FRcoord = bodytoFR4 - _bodytoFR0
        FLcoord = bodytoFL4 - _bodytoFL0
        BRcoord = bodytoBR4 - _bodytoBR0
        BLcoord = bodytoBL4 - _bodytoBL0
        # print(FLcoord[0],BRcoord[0])
        """undo transformation of leg vector to keep feet still"""
        inv_orientation = -orientation
        inv_position = -position
        _FRcoord = self.transform(FRcoord, inv_orientation, inv_position)
        _FLcoord = self.transform(FLcoord, inv_orientation, inv_position)
        _BRcoord = self.transform(BRcoord, inv_orientation, inv_position)
        _BLcoord = self.transform(BLcoord, inv_orientation, inv_position)
        
        """solve IK"""
        leg_direction=-1        
        FR_angles  = self._solve_IK(_FRcoord, leg_direction)
        BL_angles  = self._solve_IK(_BLcoord, leg_direction)
        leg_direction=-1            
        FL_angles  = self._solve_IK(_FLcoord, leg_direction)
        BR_angles  = self._solve_IK(_BRcoord, leg_direction)
        # print(FL_angles[1],BR_angles[1])        

        _bodytofeetFR = _bodytoFR0 + _FRcoord
        _bodytofeetFL = _bodytoFL0 + _FLcoord
        _bodytofeetBR = _bodytoBR0 + _BRcoord
        _bodytofeetBL = _bodytoBL0 + _BLcoord
        _bodytofeet = np.matrix([[_bodytofeetFR[0] , _bodytofeetFR[1] , _bodytofeetFR[2]],
                                 [_bodytofeetFL[0] , _bodytofeetFL[1] , _bodytofeetFL[2]],
                                 [_bodytofeetBR[0] , _bodytofeetBR[1] , _bodytofeetBR[2]],
                                 [_bodytofeetBL[0] , _bodytofeetBL[1] , _bodytofeetBL[2]]])
                                 
        return FR_angles, FL_angles, BR_angles, BL_angles, _bodytofeet
