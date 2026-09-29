"""D08 outer-only drawer initialization, native trace and independent evaluator."""
from pathlib import Path
import json,hashlib,subprocess,sys
import numpy as np
import mujoco
class KitchenTask:
    def __init__(self,m,d,root,cfg):
        self.m,self.d=m,d;self.rows=[];self.criteria_path=Path(cfg['success_criteria_path']);self.criteria_hash=cfg['success_criteria_sha256'];assert hashlib.sha256(self.criteria_path.read_bytes()).hexdigest()==self.criteria_hash
        self.criteria=json.loads(self.criteria_path.read_text());self.evaluator=Path(__file__);self.evaluator_hash=hashlib.sha256(self.evaluator.read_bytes()).hexdigest()
        z=np.load(Path(root)/'initial_state.npz');d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];mujoco.mj_forward(m,d)
        j=m.joint('door_hinge').id;self.qa=int(m.jnt_qposadr[j]);self.va=int(m.jnt_dofadr[j]);self.closed=float(1.1715178812453935);self.body=m.body('active_door').id;self.geoms=np.flatnonzero(m.geom_bodyid==self.body);self.sample()
    def contact_forces(self):
        m,d=self.m,self.d;left=right=other=pen=0.
        for i,c in enumerate(d.contact):
            bs=[int(m.geom_bodyid[g]) for g in c.geom]
            if self.body not in bs:continue
            b=bs[1] if bs[0]==self.body else bs[0];n=m.body(b).name
            if not (n.startswith('rq_') or n.startswith('link') or n=='flange_adapter'):continue
            f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);v=max(0.,float(f[0]));pen=max(pen,-float(c.dist))
            if 'left' in n:left+=v
            elif 'right' in n:right+=v
            else:other+=v
        return left,right,other,pen
    def sample(self):
        m,d=self.m,self.d;l,r,o,p=self.contact_forces();tcp=d.site('rq_pinch').xpos;dist=[]
        for g in self.geoms:
            local=d.geom_xmat[g].reshape(3,3).T@(tcp-d.geom_xpos[g]);dist.append(np.linalg.norm(np.maximum(np.abs(local)-m.geom_size[g],0)))
        self.rows.append(dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),qfrc_applied=d.qfrc_applied.copy(),residual=float(self.closed-d.qpos[self.qa]),slide_speed=float(abs(d.qvel[self.va])),hand_force=l+r+o,tcp_clearance=float(min(dist)),penetration=p,warning_count=int(sum(w.number for w in d.warning))))
    def finish(self,steps,folder):
        m,d=self.m,self.d
        for i in range(round(self.criteria['thresholds']['settle_duration_s']/m.opt.timestep)):
            d.qfrc_applied[:7]=d.qfrc_bias[:7];mujoco.mj_step(m,d);mujoco.mj_forward(m,d);self.sample()
        f=Path(folder);trace=f/'native_trace.npz';np.savez_compressed(trace,**{k:np.array([r[k] for r in self.rows]) for k in self.rows[0]});assert hashlib.sha256(self.evaluator.read_bytes()).hexdigest()==self.evaluator_hash
        out=f/'evaluation.json';subprocess.run([sys.executable,str(self.evaluator),str(trace),str(self.criteria_path),self.criteria_hash,str(out)],check=True,capture_output=True,text=True)
        r=json.loads(out.read_text());r.update(task_id='D08',executed_steps=steps,simulation_time_s=float(d.time),evaluator_sha256=self.evaluator_hash,object_state_reset_after_initialization=False,object_attachments=False,source_trajectory_used=False);return r

def evaluate(trace,criteria,expected):
    assert hashlib.sha256(Path(criteria).read_bytes()).hexdigest()==expected
    c=json.loads(Path(criteria).read_text());v=c['thresholds'];z=np.load(trace);t=z['time'];tail=t>=t[-1]-v['stable_duration_s']-1e-9;dt=np.r_[0.,np.diff(t)];contact=z['hand_force']>=v['min_contact_N'];duration=float(np.sum(dt*contact));progress=z['residual'][0]-z['residual'];under_contact=bool(np.any(contact&(progress>=v['min_contact_progress_rad'])))
    checks={'finite_states':bool(all(np.isfinite(z[k]).all() for k in ['qpos','qvel','ctrl','residual','hand_force'])),'zero_solver_warnings':bool(z['warning_count'].max()==0),'initially_open':bool(z['residual'][0]>=v['min_initial_opening_rad']),'native_push_contact':bool(duration>=v['min_contact_duration_s'] and under_contact),'closed':bool(np.all(np.abs(z['residual'][tail])<=v['max_angle_error_rad'])),'stable':bool(np.all(z['slide_speed'][tail]<=v['max_hinge_speed_rad_s'])),'released':bool(np.all(z['hand_force'][tail]<=v['max_released_force_N'])),'retracted':bool(np.all(z['tcp_clearance'][tail]>=v['min_tcp_clearance_m'])),'stable_duration':bool(t[tail][-1]-t[tail][0]>=v['stable_duration_s']-1e-8)}
    assert set(checks)==set(c['required'])
    return dict(native_success=all(checks.values()),evaluation_status='evaluated',criteria_sha256=expected,trace_sha256=hashlib.sha256(Path(trace).read_bytes()).hexdigest(),checks=checks,metrics=dict(initial_opening_rad=float(z['residual'][0]),final_max_opening_rad=float(z['residual'][tail].max()),max_closing_rotation_rad=float(progress.max()),native_contact_duration_s=duration,final_max_hinge_speed_rad_s=float(z['slide_speed'][tail].max()),final_max_hand_force_N=float(z['hand_force'][tail].max()),final_min_tcp_clearance_m=float(z['tcp_clearance'][tail].min()),stable_window_s=float(t[tail][-1]-t[tail][0]),max_penetration_m_diagnostic=float(z['penetration'].max())),goal_reference_rgb_used=False,source_goal_pose_comparison=False,thresholds_modified=False)
if __name__=='__main__':
    trace,criteria,expected,out=sys.argv[1:];Path(out).write_text(json.dumps(evaluate(trace,criteria,expected),indent=2))
