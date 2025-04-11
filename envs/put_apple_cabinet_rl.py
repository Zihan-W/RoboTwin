from .base_task import Base_task
from .utils import *
import sapien
import open3d as o3d
from .utils import *


class StageTracker:
    def __init__(self, cabinet_pos, apple_pose):
        # Initialize the stage
        self.stage = 0

        # Initialize robot TCP poses
        self.left_endpose = 0
        self.right_endpose = 0

        # Initialize cabinet_pos
        self.cabinet_pos = cabinet_pos

        # Initialize target poses
        self.left_target_pose = list(cabinet_pos + [-0.054, -0.37, -0.09]) + [0.5, 0.5, 0.5, 0.5]
        self.right_target_pose = list(apple_pose + [0, 0, 0.17]) + [-0.5, 0.5, -0.5, -0.5]

    def stage_complete(self, threshold):
        '''
        判断当前阶段是否完成，并步进到下一个stage
        一共有0~9共10个状态
        '''
        left_endpose_distance = np.linalg.norm(np.array(self.left_target_pose[:3]) - np.array(self.left_endpose.p))
        right_endpose_distance = np.linalg.norm(np.array(self.right_target_pose[:3]) - np.array(self.right_endpose.p))
        if left_endpose_distance < threshold and right_endpose_distance < threshold:
            self.stage += 1
            return True
        else:
            return False

    def update_stage(self, left_endpose, right_endpose,left_gripper_val,right_gripper_val):
        self.left_endpose = left_endpose
        self.right_endpose = right_endpose
        if self.stage == 0:
            # 左tcp移动到柜子前，右tcp移动到苹果上方，则认为阶段0结束
            # 阶段0进入到阶段1，并更新target_pose
            if self.stage_complete(threshold=0.03):
                self.left_target_pose[1] += 0.09
                self.right_target_pose[2] -= 0.05
        elif self.stage == 1:
            # 左右tcp夹紧，则认为阶段1结束
            # 阶段1进入到阶段2，并更新target_pose
            if left_gripper_val <= -0.02 and right_gripper_val <= 0:
                self.stage += 1
                self.left_target_pose[1] -= 0.18
                self.right_target_pose[2] += 0.18
        elif self.stage == 2:
            # 左tcp将抽屉拉开，右tcp将苹果举起来，则认为阶段2结束
            # 这里只判断位置，是否拉开或苹果是否被举起，由reward参与判断
            # 阶段2进入到阶段3，并更新target_pose
            if self.stage_complete(threshold=0.03):
                self.right_target_pose[1] = self.cabinet_pos[1] - 0.216
        elif self.stage == 3:
            # 右tcp平移到一定位置（向前），则认为阶段3结束
            # 阶段3进入到阶段4，并更新target_pose
            print("self.cabinet_pos:", self.cabinet_pos)
            print("self.right_target_pose:",self.right_target_pose)
            if self.stage_complete(threshold=0.03):
                self.right_target_pose = list(self.cabinet_pos+[0.036,-0.216,0.078]) + [-0.5,0.5,-0.5,-0.5]
        elif self.stage == 4:
            # 右tcp平移到一定位置（靠近抽屉），则认为阶段4结束
            # 阶段4进入到阶段5，并更新target_pose
            if self.stage_complete(threshold=0.03):
                pass
        elif self.stage == 5:
            # 右tcp夹爪打开，则认为阶段5结束
            # 阶段5进入到阶段6，并更新target_pose（实际上是回到状态4的初始位置，即状态3的末位置）
            if right_gripper_val >= 0.045:
                self.stage += 1
                self.right_target_pose = list(self.cabinet_pos+[0.036,-0.216,0.078]) + [-0.5,0.5,-0.5,-0.5]
        elif self.stage == 6:
            # 右tcp平移到一定位置（远离抽屉），则认为阶段6结束
            # 阶段6进入到阶段7，并更新target_pose
            if self.stage_complete(threshold=0.03):
                self.left_target_pose[1] += 0.18
                self.right_target_pose = [0.3,-0.32,0.935,1,0,0,1]  # right_original_pose

        elif self.stage == 7:
            # 左tcp平移到一定为止，右tcp回到初始位置，则认为阶段7结束
            if self.stage_complete(threshold=0.03):
                pass

        return self.stage

class put_apple_cabinet_rl(Base_task):
    def setup_demo(self,**kwags):
        super()._init(**kwags,table_static=False)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera()
        self.pre_move()
        self.load_actors()
        self.step_lim = 600
        _cabinet_pos = self.cabinet.get_pose().p
        _apple_pose = self.apple.get_pose().p
        self.stage_tracker = StageTracker(_cabinet_pos, _apple_pose)

    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq
    
    def load_actors(self):
        self.cabinet, _ = rand_create_urdf_obj(
            self.scene,
            modelname="036_cabine",
            xlim=[0,0],
            ylim=[0.155,0.155],
            zlim=[0.96],
            rotate_rand=False,
            qpos=[1,0,0,1],
            scale=0.27
        )

        self.cabinet_active_joints = self.cabinet.get_active_joints()
        for joint in self.cabinet_active_joints:
            # set stiffness to 0 to avoid restoring elastic force
            joint.set_drive_property(stiffness=0, damping=5, force_limit=1000, mode="force")
        self.cabinet_all_joints = self.cabinet.get_joints()

        self.apple,_ = rand_create_obj(
            self.scene,
            xlim=[0.2,0.32],
            ylim=[-0.2,-0.1],
            zlim=[0.78],
            modelname="035_apple",
            rotate_rand=False,
            convex=True
        )
        self.apple.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        
    def play_once(self):
        # 执行运动相关的函数，together_follow_path函数与set_gripper函数与left_follow_path与right_follow_path函数会被调用
        # 这些函数被调用时，会执行_take_picture函数，保存pkl格式数据集
        # 现有数据集结构:
        # ├─ data [Group]
        #   ├─ action [Array] (6190, 14) float32    ## data['joint_action']，是agent的state记录
        #   ├─ head_camera [Array] (6190, 3, 240, 320) uint8    ## data['observation']['head_camera']['rgb']
        #   ├─ state [Array] (6190, 14) float32     ## data['joint_action']，是agent的state记录
        #   ├─ tcp_action [Array] (6190, 14) float32      ## data['endpose']，是agent的tcp的state记录
        # ├─ meta [Group]
        #   ├─ episode_ends [Array] (20,) int64
        # 该数据集训练的是下一步机器人应该到哪个pos，TODO：或许可以改为增量输出，则数据集中应该有基于当前state采取的action数据
        # TODO: 增加并设计compute_reward函数，以计算reward
        # TODO: 修改_take_picture函数，以保存part_poses(包括cabinet和apple的pose)、reward（调用compute_reward函数计算）、dones（每个step都记录）
        # TODO: 修改script/pkl2zarr_rl.py，以使转换后的数据集含有reward和dones数据
        pose0 = list(self.cabinet.get_pose().p+[-0.054,-0.37,-0.09])+[0.5,0.5,0.5,0.5]
        pose1 = list(self.apple.get_pose().p+[0,0,0.17])+[-0.5,0.5,-0.5,-0.5]
        self.together_move_to_pose_with_screw(left_target_pose=pose0,right_target_pose=pose1)
        pose0[1] +=0.09
        pose1[2] -= 0.05
        self.together_move_to_pose_with_screw(left_target_pose=pose0,right_target_pose=pose1)

        self.together_close_gripper(left_pos=-0.02)
        pose0[1]-=0.18
        pose1[2]+=0.18
        self.together_move_to_pose_with_screw(left_target_pose=pose0,right_target_pose=pose1)
        pose2 = list(self.cabinet.get_pose().p+[0.036,-0.216,0.078]) + [-0.5,0.5,-0.5,-0.5]
        pose1[1] = pose2[1]
        self.right_move_to_pose_with_screw(pose1)
        self.right_move_to_pose_with_screw(pose2)

        self.open_right_gripper()
        pose2[2]+=0.082
        self.right_move_to_pose_with_screw(pose1)
        pose0[1]+=0.18

        self.together_move_to_pose_with_screw(left_target_pose=pose0,right_target_pose=self.right_original_pose)
        
    def check_success(self):
        cabinet_pos = self.cabinet.get_pose().p
        apple_pose = self.apple.get_pose().p
        left_endpose = self.get_left_endpose_pose()
        target_pose = (cabinet_pos + np.array([-0.05,-0.27,-0.09])).tolist() + [0.5, -0.5, -0.5, 0.5]
        eps1 = 0.03
        tag = np.all(abs(apple_pose[:2] - np.array([0.01, 0.1])) < np.array([0.015,0.015]))
        return np.abs(apple_pose[2] - 0.797) < 0.015 and tag and\
               np.all(abs(np.array(left_endpose.p.tolist() + left_endpose.q.tolist()) - target_pose) < eps1)

    def compute_reward(self):
        # cabinet的pose不会改变，而apple的pose会随着apple的运动而改变
        # get_left_endpose_pose得到的格式：Pose([x,y,z],[q1,q2,q3,q4])
        # 输入控制器的pose格式：[x,y,z,q1,q2,q3,q4]
        cabinet_pos = self.cabinet.get_pose().p
        apple_pose = self.apple.get_pose().p
        left_endpose = self.get_left_endpose_pose()
        right_endpose = self.get_right_endpose_pose()
        left_gripper_val = self.left_gripper_val
        right_gripper_val = self.right_gripper_val

        # 更新self.state
        state = self.stage_tracker.update_stage(left_endpose,right_endpose,left_gripper_val,right_gripper_val)
        print("current_state:",state)

        # 通过任务阶段标志动态控制奖励项的激活，引导agent按顺序学习
        if state == 0:
            # Initial approach to cabinet
            target_pose = (cabinet_pos + np.array([-0.054,-0.37,-0.09])).tolist() + [0.5,0.5,0.5,0.5]
            dist = np.linalg.norm(np.array(left_endpose.p.tolist()) - np.array(target_pose[:3]))
            reward_dist = -dist
            
        elif state == 1:
            # Grasping cabinet handle
            target_pose = (cabinet_pos + np.array([-0.054,-0.28,-0.09])).tolist() + [0.5,0.5,0.5,0.5]
            dist = np.linalg.norm(np.array(left_endpose.p.tolist()) - np.array(target_pose[:3]))
            reward_dist = -dist
            if dist < 0.02:
                reward_bonus = 1.0

        elif state == 2:
            # Opening cabinet
            target_pose = (cabinet_pos + np.array([-0.054,-0.1,-0.09])).tolist() + [0.5,0.5,0.5,0.5]
            dist = np.linalg.norm(np.array(left_endpose.p.tolist()) - np.array(target_pose[:3]))
            reward_dist = -dist

        elif state == 3:
            # Approaching apple
            target_pose = (apple_pose + np.array([0,0,0.17])).tolist() + [-0.5,0.5,-0.5,-0.5]
            dist = np.linalg.norm(np.array(right_endpose.p.tolist()) - np.array(target_pose[:3]))
            reward_dist = -dist

        elif state == 4:
            # Grasping apple
            target_pose = (apple_pose + np.array([0,0,0.12])).tolist() + [-0.5,0.5,-0.5,-0.5]
            dist = np.linalg.norm(np.array(right_endpose.p.tolist()) - np.array(target_pose[:3]))
            reward_dist = -dist
            if dist < 0.02:
                reward_bonus = 1.0

        elif state == 5:
            # Lifting apple
            target_pose = (apple_pose + np.array([0,0,0.3])).tolist() + [-0.5,0.5,-0.5,-0.5]
            dist = np.linalg.norm(np.array(right_endpose.p.tolist()) - np.array(target_pose[:3]))
            reward_dist = -dist

        elif state == 6:
            # Moving apple to cabinet
            target_pose = (cabinet_pos + np.array([0.036,-0.216,0.078])).tolist() + [-0.5,0.5,-0.5,-0.5]
            dist = np.linalg.norm(np.array(right_endpose.p.tolist()) - np.array(target_pose[:3]))
            reward_dist = -dist
            if dist < 0.02:
                reward_bonus = 1.0

        elif state == 7:
            # Return to home position
            target_pose = self.right_original_pose
            dist = np.linalg.norm(np.array(right_endpose.p.tolist()) - np.array(target_pose[:3]))
            reward_dist = -dist
            if dist < 0.02:
                reward_bonus = 1.0
        
        elif state == 8:
            # Return to home position
            target_pose = self.right_original_pose
            dist = np.linalg.norm(np.array(right_endpose.p.tolist()) - np.array(target_pose[:3]))
            reward_dist = -dist
            if dist < 0.02:
                reward_bonus = 1.0
        
        elif state == 9:
            # Return to home position
            target_pose = self.right_original_pose
            dist = np.linalg.norm(np.array(right_endpose.p.tolist()) - np.array(target_pose[:3]))
            reward_dist = -dist
            if dist < 0.02:
                reward_bonus = 1.0

        elif state == 10:
            # Return to home position
            target_pose = self.right_original_pose
            dist = np.linalg.norm(np.array(right_endpose.p.tolist()) - np.array(target_pose[:3]))
            reward_dist = -dist
            if dist < 0.02:
                reward_bonus = 1.0

        # Combine rewards
        # reward = alpha_dist * reward_dist + alpha_dir * reward_dir + alpha_align * reward_align \
        #       + alpha_bonus * reward_bonus + alpha_penalty * reward_penalty

        # # 参数
        # alpha_dist = 1
        # alpha_dir = 1
        # alpha_align = 1
        # alpha_bonus = 1
        # alpha_penalty = 1

        # # reward计算
        # reward_dist = 1
        # reward_dir = 1
        # reward_align = 1
        # reward_bonus = 1
        # reward_penalty = 1

        # reward = alpha_dist * reward_dist + alpha_dir * reward_dir + alpha_align * reward_align \
        #       + alpha_bonus * reward_bonus + alpha_penalty * reward_penalty
        
        # return reward

    def _take_picture(self): # Save data
        '''
        overwirte: for recording reward and loss
        '''
        if not self.is_save:
            return

        self.compute_reward()
        print('saving: episode = ', self.ep_num, ' index = ',self.PCD_INDEX, end='\r')
        self._update_render()
        self.left_camera.take_picture()
        self.right_camera.take_picture()
        self.head_camera.take_picture()
        self.observer_camera.take_picture()
        self.front_camera.take_picture()
        
        if self.PCD_INDEX==0:
            self.file_path ={
                "observer_color" : f"{self.save_dir}/episode{self.ep_num}/camera/color/observer/",

                "l_color" : f"{self.save_dir}/episode{self.ep_num}/camera/color/left/",
                "l_depth" : f"{self.save_dir}/episode{self.ep_num}/camera/depth/left/",
                "l_pcd" : f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/left/",

                "f_color" : f"{self.save_dir}/episode{self.ep_num}/camera/color/front/",
                "f_depth" : f"{self.save_dir}/episode{self.ep_num}/camera/depth/front/",
                "f_pcd" : f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/front/",

                "r_color" : f"{self.save_dir}/episode{self.ep_num}/camera/color/right/",
                "r_depth" : f"{self.save_dir}/episode{self.ep_num}/camera/depth/right/",
                "r_pcd" : f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/right/",

                "t_color" : f"{self.save_dir}/episode{self.ep_num}/camera/color/head/",
                "t_depth" : f"{self.save_dir}/episode{self.ep_num}/camera/depth/head/",
                "t_pcd" : f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/head/",

                "f_seg_mesh" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/front/mesh/",
                "l_seg_mesh" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/left/mesh/",
                "r_seg_mesh" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/right/mesh/",
                "t_seg_mesh" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/top/mesh/",

                "f_seg_actor" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/front/actor/",
                "l_seg_actor" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/left/actor/",
                "r_seg_actor" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/right/actor/",
                "t_seg_actor" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/head/actor/",

                "f_camera" : f"{self.save_dir}/episode{self.ep_num}/camera/model_camera/front/",
                "t_camera" : f"{self.save_dir}/episode{self.ep_num}/camera/model_camera/head/",
                "l_camera" : f"{self.save_dir}/episode{self.ep_num}/camera/model_camera/left/",
                "r_camera" : f"{self.save_dir}/episode{self.ep_num}/camera/model_camera/right/",

                "ml_ep" : f"{self.save_dir}/episode{self.ep_num}/arm/endPose/masterLeft/",
                "mr_ep" : f"{self.save_dir}/episode{self.ep_num}/arm/endPose/masterRight/",
                "pl_ep" : f"{self.save_dir}/episode{self.ep_num}/arm/endPose/puppetLeft/",
                "pr_ep" : f"{self.save_dir}/episode{self.ep_num}/arm/endPose/puppetRight/",
                "pl_joint" : f"{self.save_dir}/episode{self.ep_num}/arm/jointState/puppetLeft/",
                "pr_joint" : f"{self.save_dir}/episode{self.ep_num}/arm/jointState/puppetRight/",
                "ml_joint" : f"{self.save_dir}/episode{self.ep_num}/arm/jointState/masterLeft/",
                "mr_joint" : f"{self.save_dir}/episode{self.ep_num}/arm/jointState/masterRight/",
                "pkl" : f"{self.save_dir}_pkl/episode{self.ep_num}/",
                "conbine_pcd" : f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/conbine/",
            }

            for directory in self.file_path.values():
                if os.path.exists(directory):
                    file_list = os.listdir(directory)
                    for file in file_list:
                        os.remove(directory + file)

        pkl_dic = {
            "observation":{
                "head_camera":{},   # rbg , mesh_seg , actior_seg , depth , intrinsic_cv , extrinsic_cv , cam2world_gl(model_matrix)
                "left_camera":{},
                "right_camera":{},
                "front_camera":{}
            },
            "pointcloud":[],   # conbinet pcd
            "joint_action":[],
            "endpose":[]
        }
        
        head_camera_intrinsic_cv = self.head_camera.get_intrinsic_matrix()
        head_camera_extrinsic_cv = self.head_camera.get_extrinsic_matrix()
        head_camera_model_matrix = self.head_camera.get_model_matrix()

        pkl_dic["observation"]["head_camera"] = {
            "intrinsic_cv" : head_camera_intrinsic_cv,
            "extrinsic_cv" : head_camera_extrinsic_cv,
            "cam2world_gl" : head_camera_model_matrix
        }

        front_camera_intrinsic_cv = self.front_camera.get_intrinsic_matrix()
        front_camera_extrinsic_cv = self.front_camera.get_extrinsic_matrix()
        front_camera_model_matrix = self.front_camera.get_model_matrix()

        pkl_dic["observation"]["front_camera"] = {
            "intrinsic_cv" : front_camera_intrinsic_cv,
            "extrinsic_cv" : front_camera_extrinsic_cv,
            "cam2world_gl" : front_camera_model_matrix
        }

        left_camera_intrinsic_cv = self.left_camera.get_intrinsic_matrix()
        left_camera_extrinsic_cv = self.left_camera.get_extrinsic_matrix()
        left_camera_model_matrix = self.left_camera.get_model_matrix()

        pkl_dic["observation"]["left_camera"] = {
            "intrinsic_cv" : left_camera_intrinsic_cv,
            "extrinsic_cv" : left_camera_extrinsic_cv,
            "cam2world_gl" : left_camera_model_matrix
        }

        right_camera_intrinsic_cv = self.right_camera.get_intrinsic_matrix()
        right_camera_extrinsic_cv = self.right_camera.get_extrinsic_matrix()
        right_camera_model_matrix = self.right_camera.get_model_matrix()

        pkl_dic["observation"]["right_camera"] = {
            "intrinsic_cv" : right_camera_intrinsic_cv,
            "extrinsic_cv" : right_camera_extrinsic_cv,
            "cam2world_gl" : right_camera_model_matrix
        }

        # # ---------------------------------------------------------------------------- #
        # # RGBA
        # # ---------------------------------------------------------------------------- #
        if self.data_type.get('rgb', False):
            front_rgba = self._get_camera_rgba(self.front_camera)
            head_rgba = self._get_camera_rgba(self.head_camera)
            left_rgba = self._get_camera_rgba(self.left_camera)
            right_rgba = self._get_camera_rgba(self.right_camera)

            if self.save_type.get('raw_data', True):
                if self.data_type.get('observer', False):
                    observer_rgba = self._get_camera_rgba(self.observer_camera)
                    self.save_img(self.file_path["observer_color"]+f"{self.PCD_INDEX}.png",observer_rgba)
                self.save_img(self.file_path["t_color"]+f"{self.PCD_INDEX}.png",head_rgba)
                self.save_img(self.file_path["f_color"]+f"{self.PCD_INDEX}.png",front_rgba)
                self.save_img(self.file_path["l_color"]+f"{self.PCD_INDEX}.png",left_rgba)
                self.save_img(self.file_path["r_color"]+f"{self.PCD_INDEX}.png",right_rgba)

            if self.save_type.get('pkl' , True):
                if self.data_type.get('observer', False):
                    observer_rgba = self._get_camera_rgba(self.observer_camera)
                    pkl_dic["observation"]["observer_camera"] = {"rgb": observer_rgba[:,:,:3]}
                pkl_dic["observation"]["head_camera"]["rgb"] = head_rgba[:,:,:3]
                pkl_dic["observation"]["front_camera"]["rgb"] = front_rgba[:,:,:3]
                pkl_dic["observation"]["left_camera"]["rgb"] = left_rgba[:,:,:3]
                pkl_dic["observation"]["right_camera"]["rgb"] = right_rgba[:,:,:3]
        # # ---------------------------------------------------------------------------- #
        # # mesh_segmentation
        # # ---------------------------------------------------------------------------- # 
        if self.data_type.get('mesh_segmentation', False):
            head_seg = self._get_camera_segmentation(self.head_camera,level="mesh")
            left_seg = self._get_camera_segmentation(self.left_camera,level="mesh")
            right_seg = self._get_camera_segmentation(self.right_camera,level="mesh")
            front_seg = self._get_camera_segmentation(self.front_camera,level="mesh")

            if self.save_type.get('raw_data', True):
                self.save_img(self.file_path["t_seg_mesh"]+f"{self.PCD_INDEX}.png", head_seg)
                self.save_img(self.file_path["l_seg_mesh"]+f"{self.PCD_INDEX}.png", left_seg)
                self.save_img(self.file_path["r_seg_mesh"]+f"{self.PCD_INDEX}.png", right_seg)
                self.save_img(self.file_path["f_seg_mesh"]+f"{self.PCD_INDEX}.png", front_seg)

            if self.save_type.get('pkl' , True):
                pkl_dic["observation"]["head_camera"]["mesh_segmentation"] = head_seg
                pkl_dic["observation"]["front_camera"]["mesh_segmentation"] = front_seg
                pkl_dic["observation"]["left_camera"]["mesh_segmentation"] = left_seg
                pkl_dic["observation"]["right_camera"]["mesh_segmentation"] = right_seg
        # # ---------------------------------------------------------------------------- #
        # # actor_segmentation
        # # --------------------------------------------------------------------------- # 
        if self.data_type.get('actor_segmentation', False):
            head_seg = self._get_camera_segmentation(self.head_camera,level="actor")
            left_seg = self._get_camera_segmentation(self.left_camera,level="actor")
            right_seg = self._get_camera_segmentation(self.right_camera,level="actor")
            front_seg = self._get_camera_segmentation(self.front_camera,level="actor")

            if self.save_type.get('raw_data', True):
                self.save_img(self.file_path["t_seg_actor"]+f"{self.PCD_INDEX}.png", head_seg)
                self.save_img(self.file_path["l_seg_actor"]+f"{self.PCD_INDEX}.png", left_seg)
                self.save_img(self.file_path["r_seg_actor"]+f"{self.PCD_INDEX}.png", right_seg)
                self.save_img(self.file_path["f_seg_actor"]+f"{self.PCD_INDEX}.png", front_seg)
            if self.save_type.get('pkl' , True):
                pkl_dic["observation"]["head_camera"]["actor_segmentation"] = head_seg
                pkl_dic["observation"]["left_camera"]["actor_segmentation"] = left_seg
                pkl_dic["observation"]["right_camera"]["actor_segmentation"] = right_seg
                pkl_dic["observation"]["front_camera"]["actor_segmentation"] = front_seg
        # # ---------------------------------------------------------------------------- #
        # # DEPTH
        # # ---------------------------------------------------------------------------- #
        if self.data_type.get('depth', False):
            front_depth = self._get_camera_depth(self.front_camera)
            head_depth = self._get_camera_depth(self.head_camera)
            left_depth = self._get_camera_depth(self.left_camera)
            right_depth = self._get_camera_depth(self.right_camera)
            
            if self.save_type.get('raw_data', True):
                self.save_img(self.file_path["t_depth"]+f"{self.PCD_INDEX}.png", head_depth.astype(np.uint16))
                self.save_img(self.file_path["f_depth"]+f"{self.PCD_INDEX}.png", front_depth.astype(np.uint16))
                self.save_img(self.file_path["l_depth"]+f"{self.PCD_INDEX}.png", left_depth.astype(np.uint16))
                self.save_img(self.file_path["r_depth"]+f"{self.PCD_INDEX}.png", right_depth.astype(np.uint16))
            if self.save_type.get('pkl' , True):
                pkl_dic["observation"]["head_camera"]["depth"] = head_depth
                pkl_dic["observation"]["front_camera"]["depth"] = front_depth
                pkl_dic["observation"]["left_camera"]["depth"] = left_depth
                pkl_dic["observation"]["right_camera"]["depth"] = right_depth
        # # ---------------------------------------------------------------------------- #
        # # endpose JSON
        # # ---------------------------------------------------------------------------- #
        if self.data_type.get('endpose', False):
            left_endpose = self.endpose_transform(self.all_joints[42], self.left_gripper_val)
            right_endpose = self.endpose_transform(self.all_joints[43], self.right_gripper_val)

            if self.save_type.get('raw_data', True):
                self.save_json(self.file_path["ml_ep"]+f"{self.PCD_INDEX}.json", left_endpose)
                self.save_json(self.file_path["pl_ep"]+f"{self.PCD_INDEX}.json", left_endpose)
                self.save_json(self.file_path["mr_ep"]+f"{self.PCD_INDEX}.json", right_endpose)
                self.save_json(self.file_path["pr_ep"]+f"{self.PCD_INDEX}.json", right_endpose)

            if self.save_type.get('pkl' , True):
                if self.dual_arm:
                    pkl_dic["endpose"] = np.array([left_endpose["x"],left_endpose["y"],left_endpose["z"],left_endpose["roll"],
                                                left_endpose["pitch"],left_endpose["yaw"],left_endpose["gripper"],
                                                right_endpose["x"],right_endpose["y"],right_endpose["z"],right_endpose["roll"],
                                                right_endpose["pitch"],right_endpose["yaw"],right_endpose["gripper"],])
                else:
                    pkl_dic["endpose"] = np.array([right_endpose["x"],right_endpose["y"],right_endpose["z"],right_endpose["roll"],
                                                    right_endpose["pitch"],right_endpose["yaw"],right_endpose["gripper"],])
        # # ---------------------------------------------------------------------------- #
        # # JointState JSON
        # # ---------------------------------------------------------------------------- #
        if self.data_type.get('qpos', False):
            left_jointstate = {
                "effort" : [ 0, 0, 0, 0, 0, 0, 0 ],
                "position" : self.get_left_arm_jointState(),
                "velocity" : [ 0, 0, 0, 0, 0, 0, 0 ]
            }
            right_jointstate = {
                "effort" : [ 0, 0, 0, 0, 0, 0, 0 ],
                "position" : self.get_right_arm_jointState(),
                "velocity" : [ 0, 0, 0, 0, 0, 0, 0 ]
            }

            if self.save_type.get('raw_data', True):
                self.save_json(self.file_path["ml_joint"]+f"{self.PCD_INDEX}.json", left_jointstate)
                self.save_json(self.file_path["pl_joint"]+f"{self.PCD_INDEX}.json", left_jointstate)
                self.save_json(self.file_path["mr_joint"]+f"{self.PCD_INDEX}.json", right_jointstate)
                self.save_json(self.file_path["pr_joint"]+f"{self.PCD_INDEX}.json", right_jointstate)

            if self.save_type.get('pkl' , True):
                if self.dual_arm:
                    pkl_dic["joint_action"] = np.array(left_jointstate["position"]+right_jointstate["position"])
                else:
                    pkl_dic["joint_action"] = np.array(right_jointstate["position"])
        # # ---------------------------------------------------------------------------- #
        # # PointCloud
        # # ---------------------------------------------------------------------------- #
        if self.data_type.get('pointcloud', False):
            head_pcd = self._get_camera_pcd(self.head_camera, point_num=0)
            front_pcd = self._get_camera_pcd(self.front_camera, point_num=0)
            left_pcd = self._get_camera_pcd(self.left_camera, point_num=0)
            right_pcd = self._get_camera_pcd(self.right_camera, point_num=0) 

            # Merge pointcloud
            if self.data_type.get("conbine", False):
                conbine_pcd = np.vstack((head_pcd , left_pcd , right_pcd, front_pcd))
            else:
                conbine_pcd = head_pcd
            
            pcd_array,index = conbine_pcd[:,:3], np.array(range(len(conbine_pcd)))
            if self.pcd_down_sample_num > 0:
                pcd_array,index = fps(conbine_pcd[:,:3],self.pcd_down_sample_num)
                index = index.detach().cpu().numpy()[0]

            if self.save_type.get('raw_data', True):
                self.ensure_dir(self.file_path["t_pcd"] + f"{self.PCD_INDEX}.pcd")
                o3d.io.write_point_cloud(self.file_path["t_pcd"] + f"{self.PCD_INDEX}.pcd", self.arr2pcd(head_pcd[:,:3], head_pcd[:,3:])) 
                self.ensure_dir(self.file_path["l_pcd"] + f"{self.PCD_INDEX}.pcd")
                o3d.io.write_point_cloud(self.file_path["l_pcd"] + f"{self.PCD_INDEX}.pcd", self.arr2pcd(left_pcd[:,:3], left_pcd[:,3:]))
                self.ensure_dir(self.file_path["r_pcd"] + f"{self.PCD_INDEX}.pcd")
                o3d.io.write_point_cloud(self.file_path["r_pcd"] + f"{self.PCD_INDEX}.pcd", self.arr2pcd(right_pcd[:,:3], right_pcd[:,3:]))
                self.ensure_dir(self.file_path["f_pcd"] + f"{self.PCD_INDEX}.pcd")
                o3d.io.write_point_cloud(self.file_path["f_pcd"] + f"{self.PCD_INDEX}.pcd", self.arr2pcd(front_pcd[:,:3], front_pcd[:,3:]))
                if self.data_type.get("conbine", False):
                    self.ensure_dir(self.file_path["conbine_pcd"] + f"{self.PCD_INDEX}.pcd")
                    o3d.io.write_point_cloud(self.file_path["conbine_pcd"] + f"{self.PCD_INDEX}.pcd", self.arr2pcd(pcd_array, conbine_pcd[index,3:]))

            if self.save_type.get('pkl' , True):
                pkl_dic["pointcloud"] = conbine_pcd[index]
        #===========================================================#
        if self.save_type.get('pkl' , True):
            save_pkl(self.file_path["pkl"]+f"{self.PCD_INDEX}.pkl", pkl_dic)

        self.PCD_INDEX +=1