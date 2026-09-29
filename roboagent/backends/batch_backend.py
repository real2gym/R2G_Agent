#!/usr/bin/env python3
import base64, io, json, pathlib, sys, contextlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import imageio.v2 as imageio
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import builtins
from PIL import Image

from roboagent.backends.routing import load_backend_config
CFG = BACKEND_CONFIG if "BACKEND_CONFIG" in globals() else load_backend_config(sys.argv[1])
from roboagent.perception import PerceptionClient
PERCEPTION = PerceptionClient()
ROOT = pathlib.Path(CFG['scene_dir'])
TASK_ID = CFG['scene_id']
if TASK_ID not in ('D09','D10','D11','D12','E01','E02'): raise ValueError('Unsupported DROID task')
XML = ROOT / 'scene.xml'
MODEL = mujoco.MjModel.from_xml_path(str(XML))
DATA = mujoco.MjData(MODEL)
from roboagent.backends.batch_profile import BatchTask
PROFILE = BatchTask(MODEL, DATA, ROOT, CFG)
QA,VA,AA=PROFILE.qa,PROFILE.va,PROFILE.aa
GA,GQ=PROFILE.ga,PROFILE.gq
GRAV=PROFILE.grav
HOME_Q=DATA.qpos[QA].copy()
BASE=MODEL.body('right_fr3_link0' if TASK_ID.startswith('E') else 'link0').id
TCP=MODEL.site('right_tcp' if TASK_ID.startswith('E') else 'rq_pinch').id
mujoco.mj_forward(MODEL,DATA)
SCOPE = {}
TRACE = []
SUBSTEPS = round(.02 / MODEL.opt.timestep)
BLOCK_START = 0
STEP_COUNT = 0
ARTIFACT_DIR = pathlib.Path(CFG.get('artifact_dir', str(ROOT / 'policy_artifacts')))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
WRITERS = {'agentview': imageio.get_writer(str(ARTIFACT_DIR / 'agentview.mp4'), fps=20, codec='libx264'), 'wrist': imageio.get_writer(str(ARTIFACT_DIR / 'wrist.mp4'), fps=20, codec='libx264')}
RENDERERS = {}
for name in ('ext1', 'wrist'):
    RENDERERS[name] = mujoco.Renderer(MODEL, height=256, width=384)

def image_array(camera):
    renderer = RENDERERS[camera]
    renderer.update_scene(DATA, camera=camera)
    return np.asarray(renderer.render(), dtype=np.uint8)

def _native_view(camera):
    rgb = image_array(camera)
    h, w = rgb.shape[:2]
    camera_id = MODEL.camera(camera).id
    focal = h / (2 * np.tan(np.deg2rad(MODEL.cam_fovy[camera_id]) / 2))
    K = np.array([[focal,0.,(w-1)/2],[0.,focal,(h-1)/2],[0.,0.,1.]])
    base_R = DATA.xmat[BASE].reshape(3,3)
    T = np.eye(4)
    T[:3,:3] = base_R.T @ DATA.cam_xmat[camera_id].reshape(3,3) @ np.diag([1.,-1.,-1.])
    T[:3,3] = base_R.T @ (DATA.cam_xpos[camera_id]-DATA.xpos[BASE])
    renderer = RENDERERS[camera]
    renderer.enable_depth_rendering()
    renderer.update_scene(DATA,camera=camera)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    # Support both the public ASPIRE nested schema and the dotted-key schema
    # used by existing DROID policies.
    return {'images': {'rgb': rgb, 'depth': depth}, 'images.rgb': rgb,
            'images.depth': depth, 'intrinsics': K, 'pose_mat': T, 'K': K, 'T': T}

def native_observation():
    return {'agentview': _native_view('ext1'), 'robot0_eye_in_hand': _native_view('wrist'),
            'robot_cartesian_pos': state()['robot_cartesian_pos'], 'robot_joint_pos': state()['robot_joint_pos']}

def record_observation(obs):
    WRITERS['agentview'].append_data(np.asarray(obs['images']['agentview']))
    WRITERS['wrist'].append_data(np.asarray(obs['images']['wrist']))

def image_b64(camera):
    renderer = RENDERERS[camera]
    renderer.update_scene(DATA, camera=camera)
    frame = np.asarray(renderer.render(), dtype=np.uint8)
    buf = io.BytesIO(); Image.fromarray(frame).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def state():
    base_R=DATA.xmat[BASE].reshape(3,3)
    pos=base_R.T @ (DATA.site_xpos[TCP]-DATA.xpos[BASE])
    rot=base_R.T @ DATA.site_xmat[TCP].reshape(3,3)
    quat=Rotation.from_matrix(rot).as_quat()[[3,0,1,2]]
    opening=float(np.clip(DATA.qpos[GQ]/.04 if TASK_ID=="E01" else 1-DATA.qpos[GQ]/.8,0,1))
    return {'robot_cartesian_pos':list(pos)+list(quat)+[opening],
            'robot_joint_pos':list(DATA.qpos[QA])+[opening], 'executed_steps':int(STEP_COUNT)}

def observation():
    s = state(); s['images'] = {'agentview': image_b64('ext1'), 'wrist': image_b64('wrist')}; return s

def step(n=1):
    global STEP_COUNT
    for dummy in range(int(n)):
        if STEP_COUNT >= int(CFG.get('max_steps',4000)) or STEP_COUNT-BLOCK_START >= 600:
            raise RuntimeError('control step budget reached; return for next observation')
        for sub in range(SUBSTEPS):
            DATA.qfrc_applied[GRAV]=DATA.qfrc_bias[GRAV]
            mujoco.mj_step(MODEL,DATA)
            if PROFILE is not None and (sub+1)%getattr(PROFILE,'sample_every',1)==0:
                mujoco.mj_forward(MODEL,DATA); PROFILE.sample()
        mujoco.mj_forward(MODEL,DATA)
        STEP_COUNT+=1
        TRACE.append((DATA.time,DATA.qpos.copy(),DATA.qvel.copy(),DATA.ctrl.copy()))
        if STEP_COUNT%5==0:
            record_observation({'images':{'agentview':image_array('ext1'),'wrist':image_array('wrist')}})

def run_code(code):
    target = {'q': np.asarray(DATA.qpos[QA], dtype=float).copy()}
    def get_observation(): return native_observation()
    def get_gripper_contact_feedback():
        if PROFILE is None: raise RuntimeError('Contact feedback not configured for this scene')
        left,right,support,penetration=PROFILE.contact_forces()
        return {'left_normal_force_N':left,'right_normal_force_N':right,'source':'native pad-object contact sensors; no goal/evaluation feedback'}
    def move_to_joints(joints):
        vals=np.asarray(joints,dtype=float).reshape(-1)
        if vals.size!=7 or not np.isfinite(vals).all(): raise ValueError('Expected seven finite arm joints; use gripper API separately')
        goal=np.clip(vals,MODEL.actuator_ctrlrange[AA,0],MODEL.actuator_ctrlrange[AA,1])
        start=DATA.qpos[QA].copy()
        n=max(20,int(np.max(np.abs(goal-start))/.012))
        for i in range(1,n+1):
            alpha=i/n; alpha=alpha*alpha*(3-2*alpha)
            DATA.ctrl[AA]=start+(goal-start)*alpha; step()
        for i in range(100):
            step()
            if np.max(np.abs(DATA.qpos[QA]-goal))<.012 and np.max(np.abs(DATA.qvel[VA]))<.05: break
        print('joint_error_rad',float(np.max(np.abs(DATA.qpos[QA]-goal))))
        return state()
    def goto_home_joint_position(): return move_to_joints(HOME_Q)
    def set_gripper_open_fraction(open_fraction):
        value=float(open_fraction)
        if not np.isfinite(value) or not 0<=value<=1:raise ValueError('open_fraction must be in [0,1]')
        DATA.ctrl[GA]=255*(value if TASK_ID=="E01" else 1-value); step(60); return state()
    def open_gripper():
        DATA.ctrl[GA]=255. if TASK_ID=="E01" else 0.; step(60); return state()
    def close_gripper():
        DATA.ctrl[GA]=0. if TASK_ID=="E01" else 255.; step(60); return state()
    def interpolate_segment(joints,steps=8): return move_to_joints(joints)
    segment_sam3_text_prompt = PERCEPTION.segment_sam3_text_prompt
    segment_sam3_point_prompt = PERCEPTION.segment_sam3_point_prompt
    plan_grasp = PERCEPTION.plan_grasp
    def solve_ik(position, quaternion_wxyz=None):
        if quaternion_wxyz is None: quaternion_wxyz=state()['robot_cartesian_pos'][3:7]
        pos=np.asarray(position,dtype=float).reshape(3)
        quat=np.asarray(quaternion_wxyz,dtype=float).reshape(4)
        base_R=DATA.xmat[BASE].reshape(3,3).copy()
        target_pos=base_R@pos+DATA.xpos[BASE]
        target_R=base_R@Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        kin=mujoco.MjData(MODEL);kin.qpos[:]=DATA.qpos
        seed=DATA.qpos[QA].copy()
        def residual(q):
            kin.qpos[QA]=q; mujoco.mj_kinematics(MODEL,kin)
            r=Rotation.from_matrix(target_R @ kin.site_xmat[TCP].reshape(3,3).T).as_rotvec()
            return np.r_[10*(kin.site_xpos[TCP]-target_pos),r,.005*(q-seed)]
        lo=MODEL.actuator_ctrlrange[AA,0]+1e-5; hi=MODEL.actuator_ctrlrange[AA,1]-1e-5
        best=None
        for initial in [seed,HOME_Q]:
            fit=least_squares(residual,np.clip(initial,lo,hi),bounds=(lo,hi),max_nfev=180,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            err=residual(fit.x)
            if best is None or np.linalg.norm(err[:6])<best[0]: best=(np.linalg.norm(err[:6]),fit.x,err)
            if np.linalg.norm(err[:3])<.02 and np.linalg.norm(err[3:6])<.04:break
        if np.linalg.norm(best[2][:3])>.04 or np.linalg.norm(best[2][3:6])>.08:
            raise RuntimeError('FR3 TCP IK target unreachable within 4mm/0.08rad; choose a closer pose')
        return best[1].copy()
    def goto_pose(position,quaternion_wxyz,z_approach=0.0):
        p=np.asarray(position,dtype=float).copy();p[2]+=z_approach
        return move_to_joints(solve_ik(p,quaternion_wxyz))
    def move_to_pose(position,quaternion_wxyz):return goto_pose(position,quaternion_wxyz)
    def mask_to_world_points(mask, depth, K, T):
        ys,xs=np.where(np.asarray(mask)); z=np.asarray(depth)[ys,xs].astype(float); valid=np.isfinite(z)&(z>0)
        pts=np.stack([(xs[valid]-K[0,2])*z[valid]/K[0,0],(ys[valid]-K[1,2])*z[valid]/K[1,1],z[valid]],axis=1)
        return np.einsum("ij,nj->ni",np.asarray(T)[:3,:3],pts)+np.asarray(T)[:3,3]
    def pixel_to_world_point(u,v,z,K,T):
        z=float(z); p=np.array([(float(u)-K[0,2])*z/K[0,0], (float(v)-K[1,2])*z/K[1,1], z])
        return (np.asarray(T)[:3,:3]@p)+np.asarray(T)[:3,3]
    def get_oriented_bounding_box_from_3d_points(points):
        pts=np.asarray(points); center=np.mean(pts,axis=0); return {'center':center,'extent':np.ptp(pts,axis=0),'R':np.eye(3)}
    def decompose_transform(T):
        T=np.asarray(T); return T[:3,3], Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]]
    api = {k:v for k,v in locals().items() if callable(v) and not k.startswith('_') and k not in ('post_json',)}
    api.update({'np': np, 'get_observation': get_observation, 'move_to_joints': move_to_joints,
                'get_gripper_contact_feedback': get_gripper_contact_feedback, 'goto_home_joint_position': goto_home_joint_position, 'open_gripper': open_gripper,
                'close_gripper': close_gripper, 'interpolate_segment': interpolate_segment,
                'move_to_pose': move_to_pose, 'goto_pose': goto_pose,
                'segment_sam3_text_prompt': segment_sam3_text_prompt,
                'segment_sam3_point_prompt': segment_sam3_point_prompt,
                'plan_grasp': plan_grasp, 'solve_ik': solve_ik, 'mask_to_world_points': mask_to_world_points, 'pixel_to_world_point': pixel_to_world_point, 'get_oriented_bounding_box_from_3d_points': get_oriented_bounding_box_from_3d_points, 'decompose_transform': decompose_transform})
    from roboagent.boundary import check_code
    check_code(code)
    safe = {name:getattr(builtins,name) for name in ('range','len','float','int','list','dict','tuple','set','print','isinstance','enumerate','zip','min','max','sum','abs','round','bool','any','all','sorted','reversed','str','Exception','ValueError','RuntimeError','KeyError','IndexError')}
    safe['__import__']=__import__
    SCOPE.update(api); SCOPE['__builtins__']=safe
    exec(compile(code,'<policy>','exec'),SCOPE,SCOPE)
    return state()

def emit(x): print(json.dumps(x, ensure_ascii=False), flush=True)

task = {'suite': 'droid', 'task_id': TASK_ID, 'init_id': 0, 'seed': 0,
        'language': CFG.get('language', 'Execute the configured task in MuJoCo')}
limits = {'max_steps': int(CFG.get('max_steps', 4000))}
_ready_obs = observation(); record_observation({'images': {'agentview': np.asarray(image_array('ext1')), 'wrist': np.asarray(image_array('wrist'))}})
emit({'ready': True, 'task': task, 'limits': limits, 'observation': _ready_obs,
      'capabilities': ['observe', 'state', 'capture_after', 'move_to_joints', 'open_gripper', 'close_gripper', 'get_observation',
                       'segment_sam3_text_prompt', 'plan_grasp', 'solve_ik']})
for line in sys.stdin:
    req = json.loads(line)
    if req.get('op') == 'exec':
        BLOCK_START=STEP_COUNT
        before = state(); feedback = {'stdout': '', 'stderr': '', 'sandbox_rc': 0, 'before': before, 'after': before}
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err): feedback['after'] = run_code(req['code'])
        except Exception as exc: feedback['stderr'] = type(exc).__name__ + ': ' + str(exc); feedback['sandbox_rc'] = 1
        feedback['after']=state()
        feedback['stdout'] = out.getvalue()[-16000:]
        feedback['stderr'] = (feedback['stderr'] + err.getvalue())[-5000:]
        _obs = observation(); record_observation({'images': {'agentview': np.asarray(image_array('ext1')), 'wrist': np.asarray(image_array('wrist'))}})
        answer = {'feedback': feedback}
        if req.get('capture_after', True): answer['observation'] = _obs
        emit(answer)
    elif req.get('op') == 'observe': emit({'observation': observation()})
    elif req.get('op') == 'state': emit({'state': state()})
    elif req.get('op') == 'finish':
        result = PROFILE.finish(STEP_COUNT, ARTIFACT_DIR)
        for w in WRITERS.values(): w.close()
        if TRACE: np.savez_compressed(ARTIFACT_DIR/'trajectory.npz',time=[t[0] for t in TRACE],qpos=[t[1] for t in TRACE],qvel=[t[2] for t in TRACE],ctrl=[t[3] for t in TRACE])
        np.save(ARTIFACT_DIR/'final_qpos.npy',DATA.qpos)
        (ARTIFACT_DIR/'native_result.json').write_text(json.dumps(result,indent=2))
        (ARTIFACT_DIR/'framework_manifest.json').write_text(json.dumps({'framework':'DROID_MUJOCO','scene':str(ROOT),'perception_services': PERCEPTION.service_urls,'control_dt':SUBSTEPS*MODEL.opt.timestep,'ik':'robot-only kinematic TCP solver', 'task_id': TASK_ID,'depth':'native metric render','policy_gt_access':False},indent=2))
        emit(dict(finished=True,**result));break
