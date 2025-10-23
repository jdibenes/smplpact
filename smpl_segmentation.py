
import json
import numpy as np
import smplpact

filename_uv = './data/smpl_uv.obj'
filename_v = './data/smpl_vert_segmentation.json'
filename_f = './data/smpl_face_segmentation.json'

v_a, _, f_a, _, _, _, _ = smplpact.texture_load_uv(filename_uv)
mesh_a = smplpact.mesh_create(v_a, f_a)

with open(filename_v, 'rt') as file_v:
    vert_segmentation = json.load(file_v)

face_segmentation = {}

faces = mesh_a.faces.view(np.ndarray)
vertices = mesh_a.vertices.view(np.ndarray)

for k, v in vert_segmentation.items():
    face_indices = list(smplpact.mesh_faces_of_vertices(mesh_a, v))
    face_centroids = np.vstack([np.mean(vertices[faces[face_index], :], axis=0).reshape((1, -1)) for face_index in face_indices])
    limb_centroid = np.mean(face_centroids, axis=0).reshape((1, -1))
    face_offsets = face_centroids - limb_centroid
    u, s, vh = np.linalg.svd(face_offsets)
    dv = vh[0:1, :]
    dp = (face_offsets @ dv.T).reshape((-1,))
    di = np.argsort(dp)
    face_segmentation[k] = [int(face_indices[i]) for i in di]
    print(f'Processed {k}: {len(face_segmentation[k])}')

with open(filename_f, 'wt') as file_f:
    json.dump(face_segmentation, file_f, indent=4)










     #{'partition' : partition}
#print(len(s_v.keys()))


'''

partition = {
  'head' : False,
  'neck' : True,

  'spine' : False,
  'spine1' : True,
  'spine2' : False,
  'hips' : True,

  'leftShoulder' : True,
  'leftArm' : False,
  'leftForeArm' : True,
  'leftHand' : False,
  'leftHandIndex1' : True,

  'rightShoulder' : True,
  'rightArm' : False,
  'rightForeArm' : True,
  'rightHand' : False,
  'rightHandIndex1' : True,

  'leftUpLeg' : False,
  'leftLeg' : True,
  'leftFoot' : False,
  'leftToeBase' : True,

  'rightUpLeg' : False,
  'rightLeg' : True,
  'rightFoot' : False,
  'rightToeBase' : True,
}




'''