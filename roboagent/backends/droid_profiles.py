"""Outer-only DROID scene initialization and task evaluation; never exposed to policy."""
import json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

class MarkerTask:
    def __init__(self, model, data, root, config):
        self.model,self.data=model,data
        self.object_name='marker';self.table_name='desk_top';self.pad_prefix='';self.task_id='D02'
        self.criteria_path=Path(config['success_criteria_path'])
        self.criteria_hash=config['success_criteria_sha256']
        if hashlib.sha256(self.criteria_path.read_bytes()).hexdigest()!=self.criteria_hash:raise ValueError('Criteria hash mismatch before execution')
        self.criteria=json.loads(self.criteria_path.read_text())
        self.evaluator_path=Path(__file__).with_name('droid_evaluator.py')
        self.evaluator_hash=hashlib.sha256(self.evaluator_path.read_bytes()).hexdigest()
        self.raw=[]
        self.spec=json.loads((Path(root)/'scenario.json').read_text())
        self.initial=json.loads((Path(root)/'initial_state.json').read_text())
        jid=model.joint('marker_free').id
        self.qa=int(model.jnt_qposadr[jid]);self.va=int(model.jnt_dofadr[jid])
        data.qpos[:7]=self.initial['joint_positions']
        data.qpos[self.qa:self.qa+3]=self.initial['marker_center']
        data.qpos[self.qa+3:self.qa+7]=self.initial['marker_quat_wxyz']
        for j in range(7,jid):
            sign=-1 if any(k in model.joint(j).name for k in ('coupler','follower')) else 1
            data.qpos[model.jnt_qposadr[j]]=sign*self.initial['driver_angle']
        mujoco.mj_forward(model,data)
        # The retained reconstruction stores the frame-zero wrist extrinsic in world coordinates.
        # Preserve that initial projection and rigidly mount it to the actual wrist thereafter.
        cam=model.camera('wrist').id;parent=model.body('link7').id
        R=data.xmat[parent].reshape(3,3);world_pos=data.cam_xpos[cam].copy();world_R=data.cam_xmat[cam].reshape(3,3).copy()
        model.cam_bodyid[cam]=parent
        model.cam_pos[cam]=R.T@(world_pos-data.xpos[parent])
        model.cam_quat[cam]=Rotation.from_matrix(R.T@world_R).as_quat()[[3,0,1,2]]
        mujoco.mj_forward(model,data)
        self.camera_audit={'method':'initial world camera extrinsic converted to rigid link7 mounting','parent':'link7','pos':model.cam_pos[cam].tolist(),'quat_wxyz':model.cam_quat[cam].tolist(),'initial_position_error_m':float(np.linalg.norm(data.cam_xpos[cam]-world_pos))}
        self.origin=data.body('marker').xpos.copy();self.rows=[];self.max_penetration=0.;self.finite=True
        self.sample()
    def contact_forces(self):
        m,d=self.model,self.data
        left=right=support=0.;penetration=0.;mid=m.body(self.object_name).id;table=m.geom(self.table_name).id
        for i,c in enumerate(d.contact):
            if mid not in (m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]):continue
            other=c.geom2 if m.geom_bodyid[c.geom1]==mid else c.geom1
            f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);force=max(float(f[0]),0.)
            name=m.geom(other).name or ''
            if name in (self.pad_prefix+'left_pad1',self.pad_prefix+'left_pad2'):left+=force
            elif name in (self.pad_prefix+'right_pad1',self.pad_prefix+'right_pad2'):right+=force
            elif other==table:support+=force
            penetration=max(penetration,-float(c.dist))
        return left,right,support,penetration
    def sample(self):
        m,d=self.model,self.data;pos=d.body(self.object_name).xpos.copy();axis=d.body(self.object_name).xmat.reshape(3,3)[:,2].copy()
        if self.task_id=='D04':axis=d.geom('target_green_display').xmat.reshape(3,3)[:,2].copy()
        left,right,support,penetration=self.contact_forces()
        table=m.geom(self.table_name).id;Rt=d.geom_xmat[table].reshape(3,3);center=d.geom_xpos[table]
        # Conservative cylinder envelope inside the actual tabletop box footprint.
        points=np.array([pos-axis*.057,pos+axis*.057]);local=(points-center)@Rt
        inside=bool(np.all(np.abs(local[:,:2])+.011<=m.geom_size[table,:2]))
        if self.task_id=='D04':
            gid=m.geom('target_green_display').id
            points=np.array([pos-axis*m.geom_size[gid,1],pos+axis*m.geom_size[gid,1]])
            local=(points-center)@Rt
            inside=bool(np.all(np.abs(local[:,:2])+m.geom_size[gid,0]<=m.geom_size[table,:2]))
        if self.task_id=='D03':
            corners=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*self.half_extent
            local=(corners@d.body(self.object_name).xmat.reshape(3,3).T+pos-center)@Rt
            inside=bool(np.all(np.abs(local[:,:2])<=m.geom_size[table,:2]))
        row={'time':float(d.time),'position':pos,'axis':axis,'linear_speed':float(np.linalg.norm(d.qvel[self.va:self.va+3])),'angular_speed':float(np.linalg.norm(d.qvel[self.va+3:self.va+6])),'left_force':left,'right_force':right,'support_force':support,'penetration':penetration,'opening':float(np.clip(1-d.qpos[7]/.8,0,1)),'inside_table':inside,'tcp_clearance':float(np.linalg.norm(d.site('rq_pinch' if self.task_id in ('D03','D04') else 'pinch').xpos-pos)),'warning_count':int(sum(w.number for w in d.warning)),'qpos':d.qpos.copy(),'qvel':d.qvel.copy(),'ctrl':d.ctrl.copy(),'qfrc_applied':d.qfrc_applied.copy()}
        self.raw.append(row)
    def finish(self,step_count,folder):
        m,d=self.model,self.data;folder=Path(folder)
        for i in range(round(self.criteria['thresholds']['settle_duration_s']/m.opt.timestep)):
            d.qfrc_applied[:7]=d.qfrc_bias[:7];mujoco.mj_step(m,d);mujoco.mj_forward(m,d);self.sample()
        arrays={k:np.array([r[k] for r in self.raw]) for k in self.raw[0]}
        arrays['table_normal']=np.array(self.spec['table_normal'])/np.linalg.norm(self.spec['table_normal'])
        trace=folder/'native_trace.npz';np.savez_compressed(trace,**arrays)
        if hashlib.sha256(self.evaluator_path.read_bytes()).hexdigest()!=self.evaluator_hash:raise ValueError('Evaluator changed during execution')
        result_path=folder/'evaluation.json'
        subprocess.run([sys.executable,str(self.evaluator_path),str(trace),str(self.criteria_path),self.criteria_hash,str(result_path)],check=True,capture_output=True,text=True)
        result=json.loads(result_path.read_text())
        result.update(task_id=self.task_id,executed_steps=step_count,simulation_time_s=float(d.time),evaluator_sha256=self.evaluator_hash,object_state_reset_after_initialization=False,object_attachments=False,source_trajectory_used=False)
        (folder/'camera_mount.json').write_text(json.dumps(self.camera_audit,indent=2))
        return result

class CartonTask(MarkerTask):
    def __init__(self,model,data,root,config):
        self.model,self.data=model,data
        self.object_name='milk_carton';self.table_name='table_top';self.pad_prefix='rq_';self.task_id='D03'
        self.criteria_path=Path(config['success_criteria_path']);self.criteria_hash=config['success_criteria_sha256']
        if hashlib.sha256(self.criteria_path.read_bytes()).hexdigest()!=self.criteria_hash:raise ValueError('Criteria hash mismatch')
        self.criteria=json.loads(self.criteria_path.read_text());self.evaluator_path=Path(__file__).with_name('droid_evaluator.py');self.evaluator_hash=hashlib.sha256(self.evaluator_path.read_bytes()).hexdigest()
        initial=np.load(Path(root)/'initial_state.npz')
        for k in ('qpos','qvel','ctrl'):getattr(data,k)[:]=initial[k]
        jid=model.joint('milk_free').id;self.qa=int(model.jnt_qposadr[jid]);self.va=int(model.jnt_dofadr[jid])
        mujoco.mj_forward(model,data)
        scenario=json.loads((Path(root)/'scenario.json').read_text());self.half_extent=np.array(scenario['parameters']['mechanical']['carton']['dimensions'])/2
        self.spec={'table_normal':[0,0,1]};self.raw=[];self.camera_audit={'method':'synthetic current-scene observation cameras; wrist rigidly mounted to fr3_link7; no goal RGB'};self.sample()

class GreenTask(MarkerTask):
    def __init__(self,model,data,root,config):
        self.model,self.data=model,data
        self.object_name='target_green';self.table_name='support_table_display';self.pad_prefix='rq_';self.task_id='D04'
        self.criteria_path=Path(config['success_criteria_path']);self.criteria_hash=config['success_criteria_sha256']
        if hashlib.sha256(self.criteria_path.read_bytes()).hexdigest()!=self.criteria_hash:raise ValueError('Criteria hash mismatch')
        self.criteria=json.loads(self.criteria_path.read_text());self.evaluator_path=Path(__file__).with_name('droid_evaluator.py');self.evaluator_hash=hashlib.sha256(self.evaluator_path.read_bytes()).hexdigest()
        initial=np.load(Path(root)/'initial_state.npz')
        for k in ('qpos','qvel','ctrl'):getattr(data,k)[:]=initial[k]
        jid=model.joint('target_free').id;self.qa=int(model.jnt_qposadr[jid]);self.va=int(model.jnt_dofadr[jid])
        mujoco.mj_forward(model,data)
        self.spec={'table_normal':[0,0,1]};self.raw=[];self.camera_audit={'method':'retained camera transforms; centered perspective intrinsics for native RGB-D; no goal RGB'};self.sample()
