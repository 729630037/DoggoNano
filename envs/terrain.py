# Original script: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/heightfield.py
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
import pybullet_data as pd
from envs.model import plane
import pybullet as p
import random
import time
from envs.tools import bullet_client as bc
from envs import model

FLAG_TO_FILENAME = {
    'mounts': "heightmaps/wm_height_out.png",
    'maze': "heightmaps/Maze.png"
}

ROBOT_INIT_POSITION = {
    'mounts': [0, 0, .85],
    'plane': [0, 0, 0.21],
    'hills': [0, 0, 1.98],
    'maze': [0, 0, 0.21],
    'random': [0, 0, 0.21]
}


class Terrain:

    def __init__(self, terrain_source, terrain_id, columns=256, rows=256):
        random.seed(10)
        self.terrain_source = terrain_source
        self.terrain_id = terrain_id
        self.columns = columns
        self.rows = rows

    def generate_terrain(self, env, height_perturbation_range=0.05):
        env.pybullet_client=env
        env.pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        env.pybullet_client.configureDebugVisualizer(env.pybullet_client.COV_ENABLE_RENDERING, 0)
        height_perturbation_range = height_perturbation_range
        terrain_data = [0] * self.columns * self.rows
        if self.terrain_source == 'random':
            for j in range(int(self.columns / 2)):
                for i in range(int(self.rows / 2)):
                    height = random.uniform(0, height_perturbation_range)
                    terrain_data[2 * i + 2 * j * self.rows] = height
                    terrain_data[2 * i + 1 + 2 * j * self.rows] = height
                    terrain_data[2 * i + (2 * j + 1) * self.rows] = height
                    terrain_data[2 * i + 1 + (2 * j + 1) * self.rows] = height
            terrain_shape = env.pybullet_client.createCollisionShape(
                shapeType=env.pybullet_client.GEOM_HEIGHTFIELD,
                meshScale=[.05, .05, 1],
                heightfieldTextureScaling=(self.rows - 1) / 2,
                heightfieldData=terrain_data,
                numHeightfieldRows=self.rows,
                numHeightfieldColumns=self.columns)
            terrain = env.pybullet_client.createMultiBody(0, terrain_shape)
            env.pybullet_client.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])

        if self.terrain_source == 'csv':
            terrain_shape = env.pybullet_client.createCollisionShape(
                shapeType=env.pybullet_client.GEOM_HEIGHTFIELD,
                meshScale=[.5, .5, .5],
                fileName="heightmaps/ground0.txt",
                heightfieldTextureScaling=128)
            terrain = env.pybullet_client.createMultiBody(0, terrain_shape)
            textureId = env.pybullet_client.loadTexture(f"{plane.getDataPath()}/grass.png")
            env.pybullet_client.changeVisualShape(terrain, -1, textureUniqueId=textureId)
            env.pybullet_client.resetBasePositionAndOrientation(terrain, [1, 0, 2], [0, 0, 0, 1])

        # TODO do this better..
        if self.terrain_source == 'png':
            terrain_shape = env.pybullet_client.createCollisionShape(
                shapeType=env.pybullet_client.GEOM_HEIGHTFIELD,
                meshScale=[.1, .1, 24 if self.terrain_id == "mounts" else 1],
                fileName="heightmaps/wm_height_out.png")
            terrain = env.pybullet_client.createMultiBody(0, terrain_shape)
            if self.terrain_id == "mounts":
                textureId = env.pybullet_client.loadTexture("heightmaps/gimp_overlay_out.png")
                env.pybullet_client.changeVisualShape(terrain, -1, textureUniqueId=textureId)
                env.pybullet_client.resetBasePositionAndOrientation(terrain, [0, 0, 2], [0, 0, 0, 1])
            else:
                env.pybullet_client.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])

        self.terrain_shape = terrain_shape
        env.pybullet_client.changeVisualShape(terrain, -1, rgbaColor=[1, 1, 1, 1])
        # env.pybullet_client.configureDebugVisualizer(env.pybullet_client.COV_ENABLE_RENDERING, 1)

    def update_terrain(self, height_perturbation_range=0.05):
        if self.terrain_source == 'random':
            terrain_data = [0] * self.columns * self.rows
            for j in range(int(self.columns / 2)):
                for i in range(int(self.rows / 2)):
                    height = random.uniform(0, height_perturbation_range)
                    terrain_data[2 * i + 2 * j * self.rows] = height
                    terrain_data[2 * i + 1 + 2 * j * self.rows] = height
                    terrain_data[2 * i + (2 * j + 1) * self.rows] = height
                    terrain_data[2 * i + 1 + (2 * j + 1) * self.rows] = height
            # GEOM_CONCAVE_INTERNAL_EDGE may help avoid getting stuck at an internal (shared) edge of
            # the triangle/heightfield. GEOM_CONCAVE_INTERNAL_EDGE is a bit slower to build though.
            flags = p.GEOM_CONCAVE_INTERNAL_EDGE
            # flags = 0
            self.terrain_shape = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                flags=flags,
                meshScale=[.05, .05, 1],
                heightfieldTextureScaling=(self.rows - 1) / 2,
                heightfieldData=terrain_data,
                numHeightfieldRows=self.rows,
                numHeightfieldColumns=self.columns,
                replaceHeightfieldIndex=self.terrain_shape)




if __name__=='__main__':
    tt=Terrain("random",None)
    bbc=bc.BulletClient(connection_mode=p.GUI)
    bbc.setPhysicsEngineParameter(enableConeFriction=0)
    bbc.configureDebugVisualizer(bbc.COV_ENABLE_RENDERING, 0)
    bbc.resetSimulation()
    bbc.setPhysicsEngineParameter(numSolverIterations=int(300))
    bbc.setTimeStep(0.01)
    urdf_root=model.getDataPath()
    # ground_id = bbc.loadURDF("%s/plane/plane.urdf" % urdf_root)
    # bbc.changeVisualShape(ground_id, -1, rgbaColor=[1, 1, 1, 0.8])
    # bbc.configureDebugVisualizer(
    #     bbc.COV_ENABLE_PLANAR_REFLECTION, ground_id)
    bbc.setGravity(0, 0, -10) 
    bbc.setPhysicsEngineParameter(enableConeFriction=0)   
    tt.generate_terrain(p)      
    bbc.configureDebugVisualizer(bbc.COV_ENABLE_RENDERING, 1)     
    while True:
        keys = p.getKeyboardEvents()        
        pass