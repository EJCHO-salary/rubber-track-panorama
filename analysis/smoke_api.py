"""Exercise a real multipart job against the local server, with link count blank."""
from contextlib import ExitStack
from pathlib import Path
import hashlib
import json
import time
import httpx

root=Path(__file__).resolve().parents[1]
measurements=json.loads((root/'analysis/measurements.json').read_text(encoding='utf-8'))
for record in measurements['images']:
    assert hashlib.sha256((root/'ref_img'/record['file']).read_bytes()).hexdigest()==record['sha256']
print('Original case1 hashes unchanged.',flush=True)
with ExitStack() as stack, httpx.Client(base_url='http://127.0.0.1:8765',timeout=60) as client:
    paths=sorted((root/'ref_img/case2').glob('*.jpg'))
    files=[('images',(p.name,stack.enter_context(p.open('rb')),'image/jpeg')) for p in paths]
    response=client.post('/api/jobs',data={'width_mm':'230','pitch_mm':'48','total_links':''},files=files)
    response.raise_for_status()
    job_id=response.json()['id'];print('Job:',job_id,flush=True)
    deadline=time.monotonic()+300
    previous=None
    while time.monotonic()<deadline:
        state=client.get('/api/jobs/'+job_id).json()
        if state['message']!=previous:
            print(state['status'],state['message'],flush=True);previous=state['message']
        if state['status']=='needs_review':raise RuntimeError(state['message'])
        if state['status']=='complete':break
        time.sleep(3)
    else:raise TimeoutError('Job did not finish within five minutes.')
    result=state['result']
    assert result['settings']['total_links'] is None
    assert result['verified_loop_links']==72
    assert result['output_size_wh']==[17280,1150]
    for name in ['panorama.png','review.jpg','quality_report.json']:
        artifact=client.get(f'/api/jobs/{job_id}/files/{name}')
        artifact.raise_for_status();assert len(artifact.content)>1000
    (root/'analysis/api_smoke_result.json').write_text(json.dumps({'job_id':job_id,'status':'complete','output_size_wh':result['output_size_wh'],'verified_loop_links':72},indent=2),encoding='utf-8')
    print('Actual multipart upload, asynchronous processing, and artifact download passed.',flush=True)
