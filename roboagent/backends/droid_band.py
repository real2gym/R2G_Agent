"""D05 native elastic-band recording and independent post-episode evaluation."""
import json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np
import mujoco

class BandTask:
    sample_every=200
    def __init__(self,m,d,root,config):
        self.m,self.d=m,d;self.rows=[]
        self.criteria_path=Path(config['success_criteria_path']);self.criteria_hash=config['success_criteria_sha256']
        assert hashlib.sha256(self.criteria_path.read_bytes()).hexdigest()==self.criteria_hash
        self.criteria=json.loads(self.criteria_path.read_text());self.evaluator=Path(__file__);self.evaluator_hash=hashlib.sha256(self.evaluator.read_bytes()).hexdigest()
        z=np.load(Path(root)/'initial_state.npz');d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];mujoco.mj_forward(m,d)
        self.ids=np.array([m.body(f'band_node_{i}').id for i in range(64)]);self.band_ids=set(self.ids.tolist());self.sample()
    def contact_forces(self):
        left=right=support=pen=0.;m,d=self.m,self.d
        for i,c in enumerate(d.contact):
            bodies=[int(m.geom_bodyid[g]) for g in c.geom]
            if not any(b in self.band_ids for b in bodies):continue
            f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);load=max(0.,float(f[0]));pen=max(pen,-float(c.dist))
            for b in bodies:
                n=m.body(b).name
                if n.startswith('rq_'):
                    if 'left' in n:left+=load
                    else:right+=load
                if n in ('jar','lid'):support+=load
        return left,right,support,pen
    def sample(self):
        m,d=self.m,self.d;left,right,support,pen=self.contact_forces();nodes=d.xpos[self.ids].copy();center=d.body('lid').xpos.copy();p=nodes-center;a=np.arctan2(p[:,1],p[:,0]);winding=float(np.angle(np.exp(1j*(np.roll(a,-1)-a))).sum()/(2*np.pi))
        self.rows.append(dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),qfrc_applied=d.qfrc_applied.copy(),band_nodes=nodes,cap_center=center,winding=winding,left_force=left,right_force=right,support_force=support,penetration=pen,tcp_clearance=float(np.linalg.norm(d.site('rq_pinch').xpos-center)),warning_count=int(sum(w.number for w in d.warning))))
    def finish(self,steps,folder):
        m,d=self.m,self.d
        for i in range(round(self.criteria['thresholds']['settle_duration_s']/m.opt.timestep)):
            d.qfrc_applied[:7]=d.qfrc_bias[:7];mujoco.mj_step(m,d)
            if (i+1)%self.sample_every==0:mujoco.mj_forward(m,d);self.sample()
        folder=Path(folder);trace=folder/'native_trace.npz';np.savez_compressed(trace,**{k:np.array([r[k] for r in self.rows]) for k in self.rows[0]})
        assert hashlib.sha256(self.evaluator.read_bytes()).hexdigest()==self.evaluator_hash
        out=folder/'evaluation.json';subprocess.run([sys.executable,str(self.evaluator),str(trace),str(self.criteria_path),self.criteria_hash,str(out)],check=True,capture_output=True,text=True)
        result=json.loads(out.read_text());result.update(task_id='D05',executed_steps=steps,simulation_time_s=float(d.time),evaluator_sha256=self.evaluator_hash,object_state_reset_after_initialization=False,object_attachments=False,source_trajectory_used=False,physics_dt_s=float(m.opt.timestep),recorded_dt_s=float(m.opt.timestep*self.sample_every));return result

def evaluate(trace,criteria,expected):
    assert hashlib.sha256(Path(criteria).read_bytes()).hexdigest()==expected,'Criteria changed'
    c=json.loads(Path(criteria).read_text());v=c['thresholds'];z=np.load(trace);t=z['time'];p=z['band_nodes']-z['cap_center'][:,None,:];height=p[:,:,2];rad=np.linalg.norm(p[:,:,:2],axis=2).max(1);speed=np.linalg.norm(z['qvel'][:,16:208].reshape(-1,64,3),axis=2).max(1);loads=z['left_force']+z['right_force'];center=z['band_nodes'].mean(1);dist=np.linalg.norm(center-center[0],axis=1)
    seated=(np.abs(z['winding'])>=v['min_winding'])&(height.mean(1)>=v['min_mean_height_m'])&(height.mean(1)<=v['max_mean_height_m'])&(height.min(1)>=v['min_node_height_m'])&(height.max(1)<=v['max_node_height_m'])&(rad<=v['max_radius_m'])
    tail=t>=t[-1]-v['stable_duration_s']-1e-8
    carried=np.flatnonzero((dist>=v['min_transport_m'])&(loads>=v['min_carry_contact_N']));seat=np.flatnonzero(seated);release=np.flatnonzero(seated&(loads<=v['max_released_hand_force_N']))
    checks={'finite_states':bool(all(np.isfinite(z[k]).all() for k in ['qpos','qvel','ctrl','band_nodes'])),'zero_solver_warnings':bool(z['warning_count'].max()==0),'native_transport':bool(len(carried)),'encircle_after_transport':bool(len(carried) and len(seat) and seat[-1]>carried[0]),'native_bottle_contact':bool(z['support_force'].max()>=v['min_bottle_contact_N']),'stable_encirclement':bool(np.all(seated[tail])),'released_from_hand':bool(np.all(loads[tail]<=v['max_released_hand_force_N'])),'stable_band':bool(np.all(speed[tail]<=v['max_node_speed_m_s'])),'gripper_retracted':bool(np.all(z['tcp_clearance'][tail]>=v['min_tcp_clearance_m'])),'stable_duration':bool(t[tail][-1]-t[tail][0]>=v['stable_duration_s']-1e-8)}
    assert set(checks)==set(c['required'])
    return dict(native_success=all(checks.values()),evaluation_status='evaluated',criteria_sha256=expected,trace_sha256=hashlib.sha256(Path(trace).read_bytes()).hexdigest(),checks=checks,metrics=dict(max_transport_m=float(dist.max()),final_winding=float(z['winding'][-1]),final_mean_height_m=float(height[-1].mean()),final_max_radius_m=float(rad[-1]),final_max_node_speed_m_s=float(speed[tail].max()),final_max_hand_force_N=float(loads[tail].max()),final_min_tcp_clearance_m=float(z['tcp_clearance'][tail].min()),stable_window_s=float(t[tail][-1]-t[tail][0]),max_penetration_m_diagnostic=float(z['penetration'].max())),goal_reference_rgb_used=False,goal_position_comparison=False,thresholds_modified=False)
if __name__=='__main__':
    trace,criteria,expected,out=sys.argv[1:];Path(out).write_text(json.dumps(evaluate(trace,criteria,expected),indent=2))
