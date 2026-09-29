"""Independent, post-episode evaluator of recorded native states and contact samples."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def evaluate(trace_path,criteria_path,expected_hash):
    p=Path(criteria_path);digest=hashlib.sha256(p.read_bytes()).hexdigest()
    if digest!=expected_hash:raise ValueError('Success criteria changed after launch')
    c=json.loads(p.read_text());v=c['thresholds'];z=np.load(trace_path,allow_pickle=False)
    t=z['time'];P=z['position'];n=z['table_normal'];normal_delta=np.sum((P-P[0])*n,axis=1)
    delta=P-P[0];distance=np.linalg.norm(delta-normal_delta[:,None]*n,axis=1)
    both=(z['left_force']>=v['min_pad_contact_N'])&(z['right_force']>=v['min_pad_contact_N'])
    # Time weights exclude the initial sample; a single force spike cannot count as grasp.
    dt=np.r_[0.,np.diff(t)]
    cumulative=np.cumsum(dt*both)
    grasp=np.flatnonzero(both & (cumulative>=v['min_grasp_duration_s']))
    g=int(grasp[0]) if len(grasp) else None
    lifted=np.flatnonzero((normal_delta>=v['min_lift_m']) & (np.arange(len(t))>=(g if g is not None else len(t))))
    l=int(lifted[0]) if len(lifted) else None
    carry=np.flatnonzero(both & (normal_delta>=v['min_carry_clearance_m']) & (distance>=v['min_relocation_m']) & (np.arange(len(t))>=(l if l is not None else len(t))))
    k=int(carry[0]) if len(carry) else None
    released=(z['opening']>=v['min_open_fraction']) & ((z['left_force']+z['right_force'])<=v['max_released_pad_force_N'])
    release=np.flatnonzero(released & (np.arange(len(t))>(k if k is not None else len(t))))
    rr=int(release[0]) if len(release) else None
    tail=t>=t[-1]-v['stable_duration_s']-1e-9
    flat=np.degrees(np.arcsin(np.clip(np.abs(np.sum(z['axis']*n,axis=1)),0,1)))
    if c.get('orientation')=='upright':flat=np.degrees(np.arccos(np.clip(np.sum(z['axis']*n,axis=1),-1,1)))
    stable=released & (z['support_force']>=v['min_support_force_N']) & (z['linear_speed']<=v['max_linear_speed_m_s']) & (z['angular_speed']<=v['max_angular_speed_rad_s'])
    enough=bool(t[tail][-1]-t[tail][0]>=v['stable_duration_s']-1e-8)
    checks={'finite_states':bool(all(np.isfinite(z[a]).all() for a in ('qpos','qvel','ctrl','position','axis','left_force','right_force','support_force'))),'zero_solver_warnings':bool(z['warning_count'].max()==0),'native_grasp':g is not None,'lift_after_grasp':l is not None,'carried_displacement_after_lift':k is not None,'release_after_carry':bool(rr is not None and rr<=np.flatnonzero(tail)[0]),'stable_supported_release':bool(enough and np.all(stable[tail])),'final_relocation':bool(np.all(distance[tail]>=v['min_relocation_m'])),'flat_orientation':bool(np.all(flat[tail]<=v['max_plane_axis_angle_deg'])),'inside_table':bool(np.all(z['inside_table'][tail])),'gripper_retracted':bool(np.all(z['tcp_clearance'][tail]>=v['min_tcp_marker_clearance_m']))}
    if c.get('orientation')=='upright':checks['upright_orientation']=checks.pop('flat_orientation')
    assert set(checks)==set(c['required'])
    return {'native_success':all(checks[x] for x in c['required']),'evaluation_status':'evaluated','criteria_sha256':digest,'trace_sha256':hashlib.sha256(Path(trace_path).read_bytes()).hexdigest(),'checks':checks,'metrics':{'max_lift_m':float(normal_delta.max()),'final_min_relocation_m':float(distance[tail].min()),'final_max_axis_plane_angle_deg':float(flat[tail].max()),'final_max_speed_m_s':float(z['linear_speed'][tail].max()),'final_max_angular_speed_rad_s':float(z['angular_speed'][tail].max()),'final_min_support_N':float(z['support_force'][tail].min()),'final_max_pad_contact_N':float((z['left_force']+z['right_force'])[tail].max()),'final_min_tcp_clearance_m':float(z['tcp_clearance'][tail].min()),'stable_window_s':float(t[tail][-1]-t[tail][0]),'max_penetration_m_diagnostic':float(z['penetration'].max())},'event_times_s':{name:float(t[i]) if i is not None else None for name,i in [('grasp',g),('lift',l),('carry',k),('release',rr)]},'goal_position_comparison':False,'goal_reference_rgb_used':False,'thresholds_modified':False}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('trace');a.add_argument('criteria');a.add_argument('expected_hash');a.add_argument('output');q=a.parse_args();result=evaluate(q.trace,q.criteria,q.expected_hash);Path(q.output).write_text(json.dumps(result,indent=2));print(json.dumps({'native_success':result['native_success'],'checks':result['checks']}))
