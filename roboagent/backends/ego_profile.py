"""Outer-only bimanual initialization, native trace capture and task evaluators."""
from pathlib import Path
import json,hashlib,sys,subprocess,itertools
import numpy as np
import mujoco

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
NAMES={'E03':['target'],'E04':['target'],'E05':['target'],'E06':['base','green','orange','gray'],'E07':['cup','lid'],'E08':['cabinet','drawer','toy'],'E09':['hub','plug'],'E10':['bottle','cap'],'E11':['strip','plug'],'E12':['wheel_front','wheel_back','bolt','nut','existing_base_assembly']}
class EgoTask:
 def __init__(self,m,d,root,cfg):
  self.m,self.d,self.task,self.cfg=m,d,cfg['task_id'],cfg;self.rows=[];self.active='right';self.sample_every=round(.005/m.opt.timestep);self.criteria=Path(cfg['success_criteria_path']);assert digest(self.criteria)==cfg['success_criteria_sha256'];self.rules=json.loads(self.criteria.read_text());self.source_hash=digest(__file__)
  p=Path(root);z=json.loads((p/'initial_state.json').read_text()) if (p/'initial_state.json').exists() else np.load(p/'initial_state.npz');d.qpos[:]=z['qpos'];qv=z.get('qvel',np.zeros(m.nv)) if isinstance(z,dict) else z['qvel'];d.qvel[:]=np.zeros(m.nv) if isinstance(qv,str) and qv=='zero MjData default' else qv;self.arms={}
  for side in ['right','left']:
   js=[m.joint(f'{side}_fr3_joint{i}').id for i in range(1,8)];aa=np.array([m.actuator(f'{side}_fr3_joint{i}').id for i in range(1,8)]);qa=m.jnt_qposadr[js];va=m.jnt_dofadr[js];ga=m.actuator(side+'_hand_actuator8').id;gq=int(m.jnt_qposadr[m.joint(side+'_hand_finger_joint1').id]);self.arms[side]=dict(qa=qa,va=va,aa=aa,ga=ga,gq=gq);d.ctrl[aa]=d.qpos[qa];d.ctrl[ga]=255*np.clip(d.qpos[gq]/.04,0,1)
  a=self.arms['right'];self.qa,self.va,self.aa,self.ga,self.gq=a['qa'],a['va'],a['aa'],a['ga'],a['gq'];self.grav=np.r_[self.arms['right']['va'],self.arms['left']['va']];self.names=NAMES[self.task];self.bids=[m.body(n).id for n in self.names];self.objmap={}
  for b in range(m.nbody):
   parent=b
   while parent:
    if parent in self.bids:self.objmap[b]=self.bids.index(parent);break
    parent=int(m.body_parentid[parent])
  self.vertices={}
  for i,b in enumerate(self.bids):
   gs=[g for g in range(m.ngeom) if m.geom_bodyid[g]==b and (m.geom_contype[g] or m.geom_conaffinity[g])];self.vertices[i]=[(g,self.geom_points(g)) for g in gs]
  mujoco.mj_forward(m,d);self.sample()
 def geom_points(self,g):
  m=self.m;typ=int(m.geom_type[g]);s=m.geom_size[g]
  if typ==7:
   mid=m.geom_dataid[g];a=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);return m.mesh_vert[a:a+n].copy()
  if typ==6:return np.array(list(itertools.product([-1,1],repeat=3)))*s
  if typ==5:
   a=np.linspace(0,2*np.pi,48,endpoint=False);return np.array([[s[0]*np.cos(x),s[0]*np.sin(x),z] for x in a for z in [-s[1],s[1]]])
  if typ in (2,4):
   a=np.linspace(0,2*np.pi,24,endpoint=False);b=np.linspace(-np.pi/2,np.pi/2,13);pts=np.array([[np.cos(y)*np.cos(x),np.cos(y)*np.sin(x),np.sin(y)] for x in a for y in b]);return pts*(s[0] if typ==2 else s)
  return np.array(list(itertools.product([-1,1],repeat=3)))*np.array([s[0],s[0],s[0]+s[1]])
 def points(self,i):
  d=self.d;return np.concatenate([np.einsum('nj,ij->ni',pts,d.geom_xmat[g].reshape(3,3))+d.geom_xpos[g] for g,pts in self.vertices[i]])
 def forces(self):
  m,d=self.m,self.d;n=len(self.bids);f=np.zeros((n,2,3));table=np.zeros(n);pairs=np.zeros((n,n));box=np.zeros(n);pen=0.
  for j,c in enumerate(d.contact):
   g0,g1=map(int,c.geom);b0,b1=int(m.geom_bodyid[g0]),int(m.geom_bodyid[g1]);w=np.zeros(6);mujoco.mj_contactForce(m,d,j,w);v=max(0.,float(w[0]));pen=max(pen,-float(c.dist))
   for b,other,g in [(b0,b1,g1),(b1,b0,g0)]:
    if b not in self.objmap:continue
    i=self.objmap[b];name=m.body(other).name or '';gn=m.geom(g).name or ''
    for ai,side in enumerate(['right','left']):
     if name.startswith(side+'_hand_') or name.startswith(side+'_fr3_'):
      f[i,ai,2]+=v
      if name==side+'_hand_left_finger':f[i,ai,0]+=v
      if name==side+'_hand_right_finger':f[i,ai,1]+=v
    if gn=='table':table[i]+=v
    if gn=='box':box[i]+=v
    if other in self.objmap and self.objmap[other]!=i:pairs[i,self.objmap[other]]+=v
  return f,table,pairs,box,pen
 def contact_forces(self):
  f,t,p,b,pen=self.forces();ai=0 if self.active=='right' else 1;return float(f[:,ai,0].sum()),float(f[:,ai,1].sum()),float(t.sum()),pen
 def sample(self):
  m,d=self.m,self.d;f,t,pairs,box,pen=self.forces();pos=d.xpos[self.bids].copy();rot=d.xmat[self.bids].reshape(-1,3,3).copy();vel=[]
  for b in self.bids:
   v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,b,v,0);vel.append(v)
  tcp=np.array([d.site(side+'_tcp').xpos.copy() for side in ['right','left']]);extra={};task=self.task
  if task in ['E04','E05']:
   g=m.geom('box').id;pts=np.einsum('ni,ij->nj',self.points(0)-d.geom_xpos[g],d.geom_xmat[g].reshape(3,3));extra['box_bounds']=np.array([pts.min(0),pts.max(0)]);extra['box_size']=m.geom_size[g].copy()
  if task=='E08':extra['drawer_q']=float(d.qpos[m.jnt_qposadr[m.joint('drawer_slide').id]])
  if task in ['E09','E10','E11']:
   extra['relative']=rot[0].T@(pos[1]-pos[0]);extra['relative_rot']=rot[0].T@rot[1]
  if task=='E11':
   corners=[]
   for k in range(2):
    g=m.geom(f'plug_pin_{k}').id;v=(self.geom_points(g)@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-pos[0])@rot[0];corners.append(v)
   extra['pin_corners']=np.array(corners);extra['cable_error']=float(np.linalg.norm(d.site('cable_end_site').xpos-d.site('cable_anchor_site').xpos))
  if task=='E12':
   b=d.body('existing_block');extra['block_position']=b.xpos.copy();extra['block_rotation']=b.xmat.reshape(3,3).copy();extra['relative_bolt']=np.array([rot[2].T@(pos[i]-pos[2]) for i in [0,1,3]]);extra['nut_relative_rotation']=rot[2].T@rot[3]
  self.rows.append(dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),qfrc_applied=d.qfrc_applied.copy(),pos=pos,rot=rot,vel=np.array(vel),forces=f,table_support=t,pair_support=pairs,box_support=box,tcp=tcp,penetration=pen,warnings=int(sum(w.number for w in d.warning)),**extra))
 def finish(self,steps,folder):
  m,d=self.m,self.d
  for i in range(round(self.rules['settle_s']/m.opt.timestep)):
   d.qfrc_applied[self.grav]=d.qfrc_bias[self.grav];mujoco.mj_step(m,d)
   if (i+1)%self.sample_every==0:mujoco.mj_forward(m,d);self.sample()
  p=Path(folder);trace=p/'native_trace.npz';np.savez_compressed(trace,**{k:np.array([r[k] for r in self.rows]) for k in self.rows[0]});assert digest(__file__)==self.source_hash;out=p/'evaluation.json';subprocess.run([sys.executable,__file__,str(trace),str(self.criteria),self.cfg['success_criteria_sha256'],str(out)],check=True,capture_output=True,text=True);r=json.loads(out.read_text());r.update(executed_steps=steps,simulation_time_s=float(d.time),object_state_reset_after_initialization=False,object_attachments=False,source_trajectory_used=False,evaluator_sha256=self.source_hash);return r

def evaluate(trace,criteria,expected):
 assert digest(criteria)==expected
 c=json.loads(Path(criteria).read_text());v=c['thresholds'];task=c['task_id'];z=np.load(trace);t=z['time'];dt=np.r_[0,np.diff(t)];tail=t>=t[-1]-c['stable_s']-1e-8;p=z['pos'];rot=z['rot'];vel=z['vel'];f=z['forces'];hand=f[:,:,:,2].sum(2);lin=np.linalg.norm(vel[:,:,3:],axis=2);ang=np.linalg.norm(vel[:,:,:3],axis=2);lift=p[:,:,2]-p[0,:,2];contact=hand>=v['contact_N'];table=z['table_support'];pair=z['pair_support'];dist=np.linalg.norm(p[:,:,None,:]-z['tcp'][:,None,:,:],axis=-1)
 checks={'finite_states':all(np.isfinite(z[k]).all() for k in z.files),'zero_solver_warnings':z['warnings'].max()==0,'stable_duration':t[tail][-1]-t[tail][0]>=c['stable_s']-1e-8,'released':np.all(hand[tail]<=v['release_N']),'retracted':np.all(dist[tail]>=v['retract_m']),'stable':np.all(lin[tail]<=v['speed_m_s']) and np.all(ang[tail]<=v['angular_rad_s'])};metrics={'simulation_s':float(t[-1]),'samples':len(t),'max_penetration_m_diagnostic':float(z['penetration'].max()),'max_final_speed_m_s':float(lin[tail].max())}
 def carried(i):
  moving=(lift[:,i]>=v['lift_m'])&(lin[:,i]>.005);bil=(f[:,i,:,:2].min(2)>v['contact_N']).any(1);fraction=float(np.sum(dt*moving*bil)/max(1e-12,np.sum(dt*moving)));metrics[f'object_{i}_bilateral_ratio']=fraction;return bool(np.sum(dt*moving)>=.05 and fraction>=v['carry_ratio'])
 if task=='E03':
  hf=f[:,0,:,2];last_side=0;last_i=0;free=[];events=[];seen_sender=hf[0,0]>=v['contact_N']
  for j in range(1,len(t)):
   if hf[j].max()<v['contact_N']:free.append(j)
   receiving=1-last_side
   if seen_sender and hf[j,receiving]>=v['contact_N'] and len(free) and t[j]-t[last_i]>=v['min_transfer_s']:
    signed=(p[j,0,0]-p[last_i,0,0])*(1 if receiving==0 else -1)
    travel=np.linalg.norm(p[free[-1],0,:2]-p[free[0],0,:2]);free_s=float(dt[free].sum())
    if signed>=v['min_travel_m'] and travel>=v['min_free_travel_m'] and free_s>=v['min_free_s']:
     events.append({'time':float(t[j]),'receiving_arm':['right','left'][receiving],'travel_m':float(signed),'free_s':free_s});last_side=receiving;last_i=j;free=[]
   if hf[j,last_side]>=v['contact_N']:last_i=j;free=[];seen_sender=True
  speed=lin[:,0];omega=vel[:,0,:3];contact_velocity=vel[:,0,3:]+np.cross(omega,np.array([0,0,-v['radius_m']]));free_mask=(hf.max(1)<v['contact_N'])&(speed>.03)&(table[:,0]>.01);slip=float(np.median(np.linalg.norm(contact_velocity[free_mask,:2],axis=1)/(speed[free_mask]+1e-9))) if free_mask.any() else float('inf')
  checks.update(ten_alternating_passes=len(events)>=10,ends_right=last_side==0 and len(events)>=10 and np.all(p[tail,0,0]>=v['workspace_split_x_m']),free_rolling=slip<=v['max_slip_ratio'],on_table=np.all(table[tail,0]>.01));metrics.update(traversals=len(events),events=events,median_free_rolling_slip_ratio=slip if np.isfinite(slip) else None)
 elif task in ['E04','E05']:
  bb=z['box_bounds'];size=z['box_size'];checks.update(native_carry=carried(0),whole_object_on_box=np.all(bb[tail,0,:2]>=-size[tail,:2]) and np.all(bb[tail,1,:2]<=size[tail,:2]),box_height=np.all(abs(bb[tail,0,2]-size[tail,2])<=v['height_tol_m']),box_supported=np.all(z['box_support'][tail,0]>=v['support_N']))
 elif task=='E06':
  ins=[]
  for inner,outer in [(1,0),(2,1),(3,2)]:
   rel=np.einsum('nji,nj->ni',rot[:,outer],p[:,inner]-p[:,outer]);ins.append((np.linalg.norm(rel[:,:2],axis=1)<=v['nest_xy_m'])&(rel[:,2]>=v['nest_min_z_m'])&(rel[:,2]<=v['nest_max_z_m']))
  first=[np.flatnonzero(x) for x in ins];checks.update(ordered_nesting=all(len(x) for x in first) and first[0][0]<first[1][0]<first[2][0],nested=all(x[tail].all() for x in ins),upright=np.all(rot[tail,:,2,2]>=.95),native_carry=all(carried(i) for i in [1,2,3]),chain_support=np.all(table[tail,0]>=v['support_N']) and all(np.all(pair[tail,i,i-1]>=v['support_N']) for i in [1,2,3]))
 elif task in ['E07','E09']:
  drift=np.linalg.norm(p[:,0,:2]-p[0,0,:2],axis=1);sep=np.linalg.norm(p[:,1,:2]-p[:,0,:2],axis=1);checks.update(native_carry=carried(1),lifted=lift[:,1].max()>=v['lift_m'],separated=np.all(sep[tail]>=v['separation_m']),base_stable=drift.max()<=v['base_drift_m'],table_supported=np.all(table[tail,1]>=v['support_N']) and np.all(table[tail,0]>.01))
  if task=='E07':checks['front_placement']=np.all((p[tail,1,:2]-p[tail,0,:2])@np.array(v['front_axis_xy'])>=v['front_offset_m'])
  else:
   rel=z['relative'];delta=rel-rel[0];checks['axial_extraction']=bool(np.any((delta[:,2]>=v['extraction_m'])&contact[:,1]));metrics['maximum_axial_extraction_m']=float(delta[:,2].max())
 elif task=='E08':
  ext=z['drawer_q'];opened=np.flatnonzero(ext>=v['open_m']);lifted=np.flatnonzero((lift[:,2]>=v['lift_m'])&contact[:,2]);left=(p[:,2,:2]-p[:,0,:2])@np.array(v['left_axis_xy']);checks.update(open_before_retrieve=bool(len(opened) and len(lifted) and opened[0]<lifted[0]),native_carry=carried(2),drawer_closed=np.all(abs(ext[tail])<=v['closed_m']),toy_on_left_table=np.all(left[tail]>=v['left_offset_m']) and np.all(table[tail,2]>=v['support_N']),cabinet_stable=np.linalg.norm(p[:,0,:2]-p[0,0,:2],axis=1).max()<=v['base_drift_m']);metrics['max_drawer_extension_m']=float(ext.max())
 elif task=='E10':
  rel=z['relative'];rr=z['relative_rot'];depth=v['bottle_height_m']-rel[:,2];lateral=np.linalg.norm(rel[:,:2],axis=1);tilt=np.arccos(np.clip(rr[:,2,2],-1,1));angle=np.unwrap(np.arctan2(rr[:,1,0],rr[:,0,0]));valid=(lateral<v['engaged_lateral_m'])&(tilt<.2)&(depth>0)&(pair[:,0,1]>.005);rotation=float(np.sum(np.diff(angle)*valid[1:]*valid[:-1]));checks.update(cap_carry=carried(1),bottle_carry=carried(0),screwed=rotation<=-v['rotation_rad'],engaged_depth=np.all((depth[tail]>=v['depth_min_m'])&(depth[tail]<=v['depth_max_m'])),cap_aligned=np.all(lateral[tail]<=v['lateral_m']) and np.all(tilt[tail]<=v['axis_rad']),bottle_upright=np.all(rot[tail,0,2,2]>=np.cos(.1)),table_supported=np.all(table[tail,0]>=v['support_N']));metrics.update(thread_rotation_rad=rotation,final_depth_m=float(depth[-1]),final_lateral_m=float(lateral[-1]))
 elif task=='E11':
  corners=z['pin_corners'];depth=v['strip_height_m']-corners[:,:,::2,2].mean(2);marginx=v['slot_half_x_m']-abs(corners[:,:,:,0]-v['slot_x_m']);marginy=v['slot_half_y_m']-abs(corners[:,:,:,1]-np.array([-v['pin_spacing_m']/2,v['pin_spacing_m']/2])[None,:,None]);margin=np.minimum(marginx,marginy).min((1,2));brace=(f[:,0,:,:2].min(2)>v['contact_N']).any(1);working=contact[:,1]&(z['relative'][:,2]<.10);br=float(np.sum(dt*working*brace)/max(1e-12,np.sum(dt*working)));checks.update(native_carry=carried(1),braced=br>=v['carry_ratio'],both_pins_seated=np.all(depth[tail]>=v['depth_min_m']) and np.all(margin[tail]>=-v['corner_tol_m']),strip_stable=np.linalg.norm(p[:,0,:2]-p[0,0,:2],axis=1).max()<=v['base_drift_m'],table_supported=np.all(table[tail,0]>=v['support_N']),cable_connected=np.all(z['cable_error'][tail]<=v['cable_tol_m']));metrics.update(brace_ratio=br,final_pin_depth_m=depth[-1].tolist(),final_corner_margin_m=float(margin[-1]))
 elif task=='E12':
  rel=z['relative_bolt'];near=np.linalg.norm(rel[:,:,:2],axis=2)<=v['radial_m'];loaded=[]
  for i in [0,1]:loaded.append(near[:,i]&(rel[:,i,2]>=v['cap_height_m']-.002)&(rel[:,i,2]<=v['shaft_end_m']-.004))
  first=[np.flatnonzero(x) for x in loaded];nutok=near[:,2]&(rel[:,2,2]>.02)&(rel[:,2,2]<v['shaft_end_m']-.004);rr=z['nut_relative_rotation'];yaw=np.unwrap(np.arctan2(rr[:,1,0],rr[:,0,0]));eng=near[:,2]&(rel[:,2,2]>.02)&(pair[:,2,3]>.001);rotation=float(abs(np.sum(np.diff(yaw)*eng[1:]*eng[:-1])));br=z['block_rotation'];bp=z['block_position'];axis=-rot[:,2,:,2];blockaxis=br[:,:,2];parallel=np.arccos(np.clip(np.sum(axis*blockaxis,axis=1),-1,1));tip=p[:,2]+rot[:,2,:,2]*v['shaft_end_m'];local=np.einsum('nji,nj->ni',br,tip-bp);radial=np.linalg.norm(local[:,:2],axis=1);depth=v['block_height_m']-local[:,2];gaplocal=np.einsum('nji,nj->ni',br,p[:,3]-bp);gap=gaplocal[:,2]+np.sum(br[:,:,2]*rot[:,3,:,2],axis=1)*v['nut_height_m']-v['block_height_m'];checks.update(wheels_loaded_in_order=bool(all(len(x) for x in first) and first[0][0]<first[1][0]),wheels_retained=all(x[tail].all() for x in loaded),nut_engaged=nutok[tail].all(),nut_rotated=rotation>=v['nut_rotation_rad'],block_connection=np.all(radial[tail]<=v['radial_m']) and np.all(parallel[tail]<=v['axis_rad']) and np.all(depth[tail]>=v['block_depth_m']),nut_block_support=np.all(abs(gap[tail])<=v['gap_m']) and np.all(pair[tail,3,4]>.001),assembly_supported=np.all(table[tail,4]>.02),native_manipulation=np.any(contact[:,2]) and np.any(contact[:,3]));metrics.update(nut_rotation_rad=rotation,final_block_depth_m=float(depth[-1]),final_axis_error_m=float(radial[-1]),final_nut_gap_m=float(gap[-1]))
 checks={k:bool(vv) for k,vv in checks.items()};assert set(checks)==set(c['required']),(task,set(checks)^set(c['required']))
 return dict(task_id=task,native_success=all(checks.values()),evaluation_status='evaluated',checks=checks,metrics=metrics,criteria_sha256=expected,trace_sha256=digest(trace),thresholds_modified=False,goal_reference_rgb_used=False,source_goal_pose_comparison=False)
if __name__=='__main__':
 trace,criteria,expected,out=sys.argv[1:];Path(out).write_text(json.dumps(evaluate(trace,criteria,expected),indent=2))
