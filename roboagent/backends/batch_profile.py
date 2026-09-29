"""Outer-only task initialization and trajectory evaluator, never policy input."""
from pathlib import Path
import hashlib,json,sys,subprocess
import numpy as np
import mujoco

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class BatchTask:
 def __init__(self,m,d,root,cfg):
  self.m,self.d,self.task,self.cfg=m,d,cfg['task_id'],cfg;self.rows=[];self.sample_every=round(.005/m.opt.timestep)
  self.criteria=Path(cfg['success_criteria_path']);assert digest(self.criteria)==cfg['success_criteria_sha256'];self.rules=json.loads(self.criteria.read_text());self.source_hash=digest(__file__)
  p=Path(root);z=json.loads((p/'initial_state.json').read_text()) if (p/'initial_state.json').exists() else np.load(p/'initial_state.npz');d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel']
  e=self.task.startswith('E');names=[f'right_fr3_joint{i}' if e else f'joint{i}' for i in range(1,8)];j=[m.joint(n).id for n in names];self.qa=m.jnt_qposadr[j];self.va=m.jnt_dofadr[j];self.aa=np.arange(7);self.grav=self.va.copy()
  self.ga=7 if self.task!='E02' else 15;self.gq=14 if e else 7
  d.ctrl[self.aa]=d.qpos[self.qa];d.ctrl[self.ga]=255*np.clip(d.qpos[self.gq]/.04 if self.task=='E01' else d.qpos[self.gq]/.8,0,1)
  if e:
   lj=[m.joint(f'left_fr3_joint{i}').id for i in range(1,8)];la=[m.actuator(f'left_fr3_joint{i}').id for i in range(1,8)];d.ctrl[la]=d.qpos[m.jnt_qposadr[lj]];self.grav=np.r_[self.va,m.jnt_dofadr[lj]];d.ctrl[m.actuator('left_hand_actuator8').id]=255*np.clip(d.qpos[m.jnt_qposadr[m.joint('left_hand_finger_joint1').id]]/.04,0,1)
  self.names={'D09':['cup_A','cup_B','cup_C','cup_D'],'D10':['adapter'],'D11':['utensil','door'],'D12':['cup']+[f'candy_{i}' for i in range(12)],'E01':['target'],'E02':['target']}[self.task];self.bids=[m.body(n).id for n in self.names];self.tcp=m.site('right_tcp' if e else 'rq_pinch').id
  self.objmap={}
  for b in range(m.nbody):
   p=b
   while p:
    if p in self.bids:self.objmap[b]=self.bids.index(p);break
    p=int(m.body_parentid[p])
  self.target_geoms=[g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in self.objmap]
  mujoco.mj_forward(m,d);self.sample()
 def forces(self):
  m,d=self.m,self.d;f=np.zeros((len(self.names),4));pen=0.
  for i,c in enumerate(d.contact):
   g0,g1=map(int,c.geom);b0,b1=int(m.geom_bodyid[g0]),int(m.geom_bodyid[g1]);w=np.zeros(6);mujoco.mj_contactForce(m,d,i,w);v=max(0.,float(w[0]));pen=max(pen,-float(c.dist))
   for b,o,g in [(b0,b1,g1),(b1,b0,g0)]:
    if b not in self.objmap:continue
    ix=self.objmap[b];n=m.body(o).name or '';gn=m.geom(g).name or ''
    hand=('rq_' in n or 'right_robotiq_' in n or n.startswith('right_hand_'))
    robot=hand or n.startswith('link') or n.startswith('right_fr3_')
    if robot:
     f[ix,2]+=v
     if hand and 'left' in n:f[ix,0]+=v
     if hand and 'right' in n.replace('right_hand_','').replace('right_robotiq_',''):f[ix,1]+=v
    elif o not in self.objmap or (self.task=='D11' and self.objmap.get(o)==1):
     if self.task=='E01' and not gn.startswith('plate_'):continue
     if self.task=='E02' and o!=m.body('tray').id:continue
     if self.task=='D12' and gn!='table_top':continue
     f[ix,3]+=v
  return f,pen
 def contact_forces(self):
  f,p=self.forces();return float(f[:,0].sum()),float(f[:,1].sum()),float(f[:,3].sum()),p
 def sample(self):
  m,d=self.m,self.d;forces,pen=self.forces();pos=d.xpos[self.bids].copy();rot=d.xmat[self.bids].reshape(-1,3,3).copy();vel=[]
  for b in self.bids:
   v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,b,v,0);vel.append(v)
  metrics={};task=self.task
  if task=='D10':
   strip=d.body('power_strip');sr=strip.xmat.reshape(3,3);blades=np.array([sr.T@(d.geom(f'blade_{i}').xpos-strip.xpos) for i in range(2)]);metrics['blades']=blades;metrics['prong_down']=-float((sr.T@rot[0])[2,1])
  if task=='D11':
   vertices=[]
   for g in self.target_geoms:
    if self.objmap[int(m.geom_bodyid[g])]!=0 or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_BOX:continue
    for x in [-1,1]:
     for y in [-1,1]:
      for z in [-1,1]:vertices.append(d.geom_xpos[g]+d.geom_xmat[g].reshape(3,3)@(m.geom_size[g]*[x,y,z]))
   cab=d.body('cabinet');vs=(np.array(vertices)-cab.xpos)@cab.xmat.reshape(3,3);metrics['tool_bounds']=np.array([vs.min(0),vs.max(0)]);metrics['door_angle']=float(d.qpos[m.jnt_qposadr[m.joint('door_hinge').id]])
  if task=='D12':
   g=m.geom('table_top').id;rel=(pos[1:]-d.geom_xpos[g])@d.geom_xmat[g].reshape(3,3);metrics['table_relative']=rel;metrics['table_radius']=float(m.geom_size[g,0]);metrics['table_top_z']=float(d.geom_xpos[g,2]+m.geom_size[g,1]);metrics['candy_relative_cup']=(pos[1:]-pos[0])@rot[0]
  if task=='E01':metrics['target_relative']=pos[0]-d.geom('plate_bottom').xpos
  if task=='E02':metrics['target_relative']=d.body('tray').xmat.reshape(3,3).T@(pos[0]-d.body('tray').xpos)
  self.rows.append(dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),qfrc_applied=d.qfrc_applied.copy(),pos=pos,rot=rot,vel=np.array(vel),forces=forces,clearance=np.linalg.norm(pos-d.site_xpos[self.tcp],axis=1),penetration=pen,warnings=int(sum(w.number for w in d.warning)),**metrics))
 def finish(self,steps,folder):
  m,d=self.m,self.d
  for i in range(round(self.rules['settle_s']/m.opt.timestep)):
   d.qfrc_applied[self.grav]=d.qfrc_bias[self.grav];mujoco.mj_step(m,d)
   if (i+1)%self.sample_every==0:mujoco.mj_forward(m,d);self.sample()
  p=Path(folder);trace=p/'native_trace.npz';np.savez_compressed(trace,**{k:np.array([r[k] for r in self.rows]) for k in self.rows[0]});assert digest(__file__)==self.source_hash
  out=p/'evaluation.json';subprocess.run([sys.executable,__file__,str(trace),str(self.criteria),self.cfg['success_criteria_sha256'],str(out)],check=True,capture_output=True,text=True)
  r=json.loads(out.read_text());r.update(executed_steps=steps,simulation_time_s=float(d.time),object_state_reset_after_initialization=False,object_attachments=False,source_trajectory_used=False,evaluator_sha256=self.source_hash);return r

def evaluate(trace,criteria,expected):
 assert digest(criteria)==expected
 c=json.loads(Path(criteria).read_text());z=np.load(trace);task=c['task_id'];v=c['thresholds'];t=z['time'];tail=t>=t[-1]-c['stable_s']-1e-8;p=z['pos'];rot=z['rot'];f=z['forces'];vel=z['vel'];dt=np.r_[0,np.diff(t)];lin=np.linalg.norm(vel[:,: ,3:],axis=2);ang=np.linalg.norm(vel[:,:,:3],axis=2)
 checks={'finite_states':all(np.isfinite(z[k]).all() for k in ['qpos','qvel','ctrl','forces']),'zero_solver_warnings':z['warnings'].max()==0,'stable_duration':t[tail][-1]-t[tail][0]>=c['stable_s']-1e-8,'released':np.all(f[tail,:,2]<=v['release_N']),'retracted':np.all(z['clearance'][tail,0]>=v['retract_m'])}
 metrics={'steps_recorded':len(t),'simulation_s':float(t[-1]),'max_penetration_m_diagnostic':float(z['penetration'].max()),'max_final_speed_m_s':float(lin[tail].max())}
 contact=f[:,:,2]>v['contact_N'];lift=p[:,:,2]-p[0,:,2];checks['native_manipulation']=bool(np.any(contact&(lift>=v['lift_m'])))
 checks['stable']=bool(np.all(lin[tail]<=v['speed_m_s']) and np.all(ang[tail]<=v['angular_rad_s']))
 if task=='D09':
  def nested(inner,outer):
   rel=np.einsum('nji,nj->ni',rot[:,outer],p[:,inner]-p[:,outer]);return (np.linalg.norm(rel[:,:2],axis=1)<=v['center_spread_m'])&(rel[:,2]>=v['min_stack_rise_m'])&(rel[:,2]<=v['max_stack_rise_m'])
  da=nested(3,0);ac=da&nested(0,2);cb=ac&nested(2,1);s1=np.flatnonzero(da);s2=np.flatnonzero(ac);s3=np.flatnonzero(cb)
  checks['ordered_nesting']=bool(len(s1) and len(s2) and len(s3) and s1[0]<s2[0]<s3[0]);checks['all_nested']=bool(cb[tail].all());checks['upright']=bool(np.all(rot[tail,:,2,2]>=v['upright_dot']));checks['each_inner_transported']=bool(all(np.any(contact[:,i]&(lift[:,i]>=v['lift_m'])) for i in [3,0,2]));metrics['final_centers']=p[-1].tolist()
 elif task=='D10':
  b=z['blades'];dep=-b[:,:,2];xe=np.minimum(abs(b[:,:,0]-.00635),abs(b[:,:,0]+.00635));ye=np.min(abs(b[:,:,1,None]-np.array(v['slot_y'])),axis=2);same=np.argmin(abs(b[:,:,1,None]-np.array(v['slot_y'])),axis=2);seated=(dep.min(1)>=v['depth_min_m'])&(dep.max(1)<=v['depth_max_m'])&(xe.max(1)<=v['x_tol_m'])&(ye.max(1)<=v['y_tol_m'])&(same[:,0]==same[:,1])&(b[:,0,0]*b[:,1,0]<0)&(z['prong_down']>=v['down_dot']);checks['inserted']=bool(seated[tail].all());metrics['final_blades']=b[-1].tolist()
 elif task=='D11':
  a=z['door_angle'];bb=z['tool_bounds'];inside=(bb[:,0]>=np.array(v['cavity_min'])).all(1)&(bb[:,1]<=np.array(v['cavity_max'])).all(1);opened=np.flatnonzero(a>=v['open_rad']);stored=np.flatnonzero(inside);moved=np.flatnonzero((np.linalg.norm(p[:,0]-p[0,0],axis=1)>.03)&contact[:,0]);checks['open_retrieve_store_order']=bool(len(opened) and len(stored) and len(moved) and opened[0]<moved[0]<stored[0]);checks['stored']=bool(inside[tail].all());checks['door_closed']=bool((abs(a[tail])<=v['closed_rad']).all());checks['supported']=bool((f[tail,0,3]>v['support_N']).all());checks['native_manipulation']=bool(len(moved));metrics['final_door_angle']=float(a[-1]);metrics['tool_bounds']=bb[-1].tolist()
 elif task=='D12':
  rel=z['candy_relative_cup'];outside=(np.linalg.norm(rel[:,:,:2],axis=2)>v['cup_outer_radius_m'])|(rel[:,:,2]<-.015)|(rel[:,:,2]>.14);on=(np.linalg.norm(z['table_relative'][:,:,:2],axis=2)+v['candy_extent_m']<=z['table_radius'][:,None])&(f[:,1:,3]>v['support_N'])&outside;checks['candies_on_table']=bool((on[tail].sum(1)>=v['min_candies']).all());checks['tilted_to_pour']=bool(np.any((rot[:,0,2,2]<v['pour_dot'])&contact[:,0]));checks['cup_returned']=bool((rot[tail,0,2,2]>=v['upright_dot']).all() and (f[tail,0,3]>v['support_N']).all());metrics['final_candies_on_table']=int(on[-1].sum())
 else:
  rel=z['target_relative'];inside=(np.linalg.norm(rel[:,:2],axis=1)+v['target_radius_m']<=v['plate_inner_radius_m']) if task=='E01' else (np.abs(rel[:,:2])+v['target_radius_m']<=np.array(v['tray_inner_half_xy'])).all(1)
  checks['inside_destination']=bool(inside[tail].all());checks['supported']=bool((f[tail,0,3]>v['support_N']).all());carry=(lift[:,0]>=v['lift_m'])&(lin[:,0]>.005);bil=(f[:,0,0]>v['contact_N'])&(f[:,0,1]>v['contact_N']);fraction=float(np.sum(dt*carry*bil)/max(1e-12,np.sum(dt*carry)));checks['bilateral_transport']=bool(np.sum(dt*carry)>.1 and fraction>=v['bilateral_fraction']);metrics['bilateral_transport_fraction']=fraction;metrics['final_relative_position']=rel[-1].tolist()
 checks={k:bool(vv) for k,vv in checks.items()};assert set(checks)==set(c['required'])
 return dict(task_id=task,native_success=all(checks.values()),evaluation_status='evaluated',checks=checks,metrics=metrics,criteria_sha256=expected,trace_sha256=digest(trace),thresholds_modified=False,goal_reference_rgb_used=False,source_goal_pose_comparison=False)
if __name__=='__main__':
 trace,criteria,expected,out=sys.argv[1:];Path(out).write_text(json.dumps(evaluate(trace,criteria,expected),indent=2))
