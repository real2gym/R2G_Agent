"""Actual SAM3 / GraspNet HTTP contracts used by the experiment executors.

Servers, model code and weights must be provisioned separately. No fallback.
"""
import base64
import io
import json
import os
import urllib.request
import numpy as np
from PIL import Image


def np_b64(value):
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode()


def b64_np(value):
    blob = base64.b64decode(value, validate=True)
    if blob.startswith(b'\x93NUMPY'):
        return np.load(io.BytesIO(blob), allow_pickle=False)
    if blob.startswith(b'\x89PNG'):
        return np.asarray(Image.open(io.BytesIO(blob)))
    return np.frombuffer(blob, dtype=np.uint8)


class PerceptionClient:
    def __init__(self, sam3_url=None, graspnet_url=None, timeout=30):
        self.service_urls = {
            'sam3': sam3_url or os.environ.get('R2G_SAM3_URL'),
            'graspnet': graspnet_url or os.environ.get('R2G_GRASPNET_URL')}
        if not all(self.service_urls.values()):
            raise ValueError('Set R2G_SAM3_URL and R2G_GRASPNET_URL to real deployed services')
        if not all(url.startswith(('http://','https://')) for url in self.service_urls.values()):
            raise ValueError('Perception endpoints must be HTTP(S) URLs')
        self.timeout = timeout

    def _post(self, service, route, payload):
        request = urllib.request.Request(self.service_urls[service].rstrip('/')+route,
            data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    @staticmethod
    def _png(rgb):
        stream = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(stream, format='PNG')
        return base64.b64encode(stream.getvalue()).decode()

    def segment_sam3_text_prompt(self, rgb, text_prompt):
        raw = self._post('sam3', '/segment', {'image_base64':self._png(rgb), 'text_prompt':text_prompt})
        return [{'mask':b64_np(x['mask_base64']).reshape(x['shape']).astype(bool),
                 'score':x.get('score',0.), 'box':x.get('box')} for x in raw['results']]

    def segment_sam3_point_prompt(self, rgb, point_coords):
        raw = self._post('sam3', '/segment_point',
            {'image_base64':self._png(rgb), 'point_coords':list(point_coords)})
        masks = np.frombuffer(base64.b64decode(raw['masks_base64'], validate=True),
            dtype=np.dtype(raw['masks_dtype'])).reshape(raw['masks_shape'])
        return [{'mask':m.astype(bool),'score':float(score)} for m,score in zip(masks,raw['scores'])]

    def plan_grasp(self, depth, K, mask, segmap_id=1, **kwargs):
        payload = {'depth_base64':np_b64(depth), 'cam_K_base64':np_b64(K),
            'segmap_base64':np_b64(np.asarray(mask,dtype=np.int32)), 'segmap_id':int(segmap_id)}
        payload.update({k:v for k,v in kwargs.items() if k in
            {'local_regions','filter_grasps','skip_border_objects','z_range','forward_passes','max_retries'}})
        raw = self._post('graspnet','/plan',payload)
        return b64_np(raw['grasps_base64']), b64_np(raw['scores_base64'])
