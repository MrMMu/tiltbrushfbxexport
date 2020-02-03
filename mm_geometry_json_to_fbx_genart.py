#!/usr/bin/env python

# Copyright 2016 Google Inc. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Historical sample code that converts Tilt Brush '.json' exports to .fbx.
# This script is superseded by Tilt Brush native .fbx exports.
# 
# There are command-line options to fine-tune the fbx creation.
# The defaults are:
# 
# - Weld vertices
# - Join strokes using the same brush into a single mesh
# - Don't create backface geometry for single-sided brushes"""

import argparse
from itertools import groupby
import os
import platform
import sys
import json
import numpy

FNAME='bside.json'
MERGE_BRUSH=False
RELOCATE_BRUSHES=True #move all brushes to [0,0,0] for generative art
EXPORT_RELOCATION=1   #1: single file 2:seperate files (one per brush)
EXPORT_BRUSH_AREA=True  #calculate brush size and export


try:
  sys.path.append(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'Python'))
  from tiltbrush.export import iter_meshes, TiltBrushMesh, SINGLE_SIDED_FLAT_BRUSH
except ImportError:
  print >>sys.stderr, "Please put the 'Python' directory in your PYTHONPATH"
  sys.exit(1)

#arch = 'x64' if '64' in platform.architecture()[0] else 'x86'
arch='ub' #MM amend to fit mac
dir = '/Applications/Autodesk/FBX Python SDK'
versions = sorted(os.listdir(dir), reverse=True)
found = False
for version in versions:
  path = '{0}/{1}/lib/Python27_{2}'.format(dir, version, arch)
  if os.path.exists(path):
    sys.path.append(path)
    try:
      from fbx import *
      found = True
    except ImportError:
      print >>sys.stderr, "Failed trying to import fbx from {0}".format(path)
      sys.exit(1)
    break
if not found:
  print >>sys.stderr, "Please install the Python FBX SDK: http://www.autodesk.com/products/fbx/"

# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------

def as_fvec4(tup, scale=1):
  if len(tup) == 3:
    return FbxVector4(tup[0]*scale, tup[1]*scale, tup[2]*scale)
  else:
    return FbxVector4(tup[0]*scale, tup[1]*scale, tup[2]*scale, tup[3]*scale)

def as_fvec2(tup):
  return FbxVector2(tup[0], tup[1])

def as_fcolor(abgr_int, memo={}):
  try:
    return memo[abgr_int]
  except KeyError:
    a = (abgr_int >> 24) & 0xff
    b = (abgr_int >> 16) & 0xff
    g = (abgr_int >>  8) & 0xff
    r = (abgr_int      ) & 0xff
    scale = 1.0 / 255.0
    memo[abgr_int] = val = FbxColor(r * scale, g * scale, b * scale, a * scale)
    return val

#--------------
# MM CALCULATE POLYGON SURFACE AREA
# https://stackoverflow.com/questions/12642256/python-find-area-of-polygon-from-xyz-coordinates/12643315
#--------------
#determinant of matrix a
def det(a):
  return a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] - a[0][2]*a[1][1]*a[2][0] - a[0][1]*a[1][0]*a[2][2] - a[0][0]*a[1][2]*a[2][1]

#unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
  x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
  y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
  z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
  magnitude = (x**2 + y**2 + z**2)**.5
  return (x/magnitude, y/magnitude, z/magnitude)

#dot product of vectors a and b
def dot(a, b):
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

#cross product of vectors a and b
def cross(a, b):
  x = a[1] * b[2] - a[2] * b[1]
  y = a[2] * b[0] - a[0] * b[2]
  z = a[0] * b[1] - a[1] * b[0]
  return (x, y, z)

#area of polygon poly
def area(poly):
  if len(poly) < 3: # not a plane - no area
    return 0

  total = [0, 0, 0]
  for i in range(len(poly)):
    vi1 = poly[i]
    if i is len(poly)-1:
      vi2 = poly[0]
    else:
      vi2 = poly[i+1]
    prod = cross(vi1, vi2)
    total[0] += prod[0]
    total[1] += prod[1]
    total[2] += prod[2]
  result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
  return abs(result/2)

def unit_normal(a, b, c):
  x =numpy.linalg.det([[1,a[1],a[2]],
                       [1,b[1],b[2]],
         [1,c[1],c[2]]])
  y = numpy.linalg.det([[a[0],1,a[2]],
                       [b[0],1,b[2]],
         [c[0],1,c[2]]])
  z = numpy.linalg.det([[a[0],a[1],1],
                       [b[0],b[1],1],
         [c[0],c[1],1]])
  magnitude = (x**2 + y**2 + z**2)**.5
  return (x/magnitude, y/magnitude, z/magnitude)

#area of polygon poly
def poly_area(poly):
  if len(poly) < 3: # not a plane - no area
    return 0
  total = [0, 0, 0]
  N = len(poly)
  for i in range(N):
    vi1 = poly[i]
    vi2 = poly[(i+1) % N]
    prod = numpy.cross(vi1, vi2)
    total[0] += prod[0]
    total[1] += prod[1]
    total[2] += prod[2]
  result = numpy.dot(total, unit_normal(poly[0], poly[1], poly[2]))
  return abs(result/2)

#-------------------------------
#-------------------------------

# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------

def write_fbx_meshes(meshes, outf_name):
  """Emit a TiltBrushMesh as a .fbx file"""
  import FbxCommon
  global n
  (sdk, scene) = FbxCommon.InitializeSdkObjects()

  docInfo = FbxDocumentInfo.Create(sdk, 'DocInfo')
  docInfo.Original_ApplicationVendor.Set('Google')
  docInfo.Original_ApplicationName.Set('Tilt Brush')
  docInfo.LastSaved_ApplicationVendor.Set('Google')
  docInfo.LastSaved_ApplicationName.Set('Tilt Brush')
  scene.SetDocumentInfo(docInfo)
  
  contentid=os.path.splitext(outf_name)[0]

  if EXPORT_RELOCATION==1 or RELOCATE_BRUSHES is False:
    for mesh in meshes:
      add_mesh_to_scene(sdk, scene, mesh,contentid)
    FbxCommon.SaveScene(sdk, scene, "ALLINONE")
    
  if EXPORT_RELOCATION==2:
    #BRUSH IN INDIVIDUAL FILE
    mesh=meshes
    add_mesh_to_scene(sdk, scene, mesh,contentid)
    FbxCommon.SaveScene(sdk, scene, mesh.brush_name+"_"+str(mesh.c[0])+"_"+contentid+"_"+str(n))
  
  

def create_fbx_layer(fbx_mesh, data, converter_fn, layer_class,
                     allow_index=False, allow_allsame=False):
  """Returns an instance of layer_class populated with the passed data,
  or None if the passed data is empty/nonexistent.
  
  fbx_mesh      FbxMesh
  data          list of Python data
  converter_fn  Function converting data -> FBX data
  layer_class   FbxLayerElementXxx class
  allow_index   Allow the use of eIndexToDirect mode. Useful if the data
                has many repeated values. Unity3D doesn't seem to like it
                when this is used for vertex colors, though.
  allow_allsame Allow the use of eAllSame mode. Useful if the data might
                be entirely identical.  This allows passing an empty data list,
                in which case FBX will use a default value."""
  # No elements, or all missing data.
  if not allow_allsame and (len(data) == 0 or data[0] == None):
    return None

  layer_elt = layer_class.Create(fbx_mesh, "")
  direct = layer_elt.GetDirectArray()
  index = layer_elt.GetIndexArray()
  
  if allow_allsame or allow_index:
    unique_data = sorted(set(data))

  # Something about this eIndexToDirect code isn't working for vertex colors and UVs.
  # Do it the long-winded way for now, I guess.
  allow_index = False
  if allow_allsame and len(unique_data) <= 1:
    layer_elt.SetMappingMode(FbxLayerElement.eAllSame)
    layer_elt.SetReferenceMode(FbxLayerElement.eDirect)
    if len(unique_data) == 1:
      direct.Add(converter_fn(unique_data[0]))
  elif allow_index and len(unique_data) <= len(data) * .7:
    layer_elt.SetMappingMode(FbxLayerElement.eByControlPoint)
    layer_elt.SetReferenceMode(FbxLayerElement.eIndexToDirect)
    for datum in unique_data:
      direct.Add(converter_fn(datum))
    for i in range(len(data)-len(unique_data)-5):
      direct.Add(converter_fn(unique_data[0]))
    data_to_index = dict((d, i) for (i, d) in enumerate(unique_data))
    for i,datum in enumerate(data):
      #index.Add(data_to_index[datum])
      index.Add(data_to_index[datum])
  else:
    layer_elt.SetMappingMode(FbxLayerElement.eByControlPoint)
    layer_elt.SetReferenceMode(FbxLayerElement.eDirect)
    for datum in data:
      direct.Add(converter_fn(datum))

  return layer_elt


def mm_save_mesh_metadata(name,mesh):
  global export_name
  global metadata
  d={}
  m={}
  d["meshname"] = str(name)
  m["brush_name"]=mesh.brush_name #Roughly analagous to a material
  m["brush_guid"]=str(mesh.brush_guid)
  m["v"]=mesh.v          #list of positions (3-tuples)
  m["n"]=mesh.n          #list of normals (3-tuples, or None if missing)
  m["uv0"]=mesh.uv0        #list of uv0 (2-, 3-, 4-tuples, or None if missing)
  m["uv1"]=mesh.uv1        #see uv0
  m["c"]=mesh.c          #list of colors, as a uint32. abgr little-endian, rgba big-endian
  m["t"]=mesh.t          #list of tangents (4-tuples, or None if missing)
  m["tri"]=mesh.tri        #list of triangles (3-tuples of ints)  
  d["meshmeta"] = m  
  metadata['fbxmeta'].append(d)
  #print d
  #print json.dumps(d)
  

def add_mesh_to_scene(sdk, scene, mesh, contentid):
  """Emit a TiltBrushMesh as a .fbx file"""
  global n
  name = contentid+"_"+str(n)
  n+=1
  # Todo: pass scene instead?
  fbx_mesh = FbxMesh.Create(sdk, name)
  fbx_mesh.CreateLayer()
  layer0 = fbx_mesh.GetLayer(0)

  # Verts

  fbx_mesh.InitControlPoints(len(mesh.v))
  if RELOCATE_BRUSHES is True:
    print mesh.v
    #MM TRANSLATE BRUSHES
    filler=(0,0,0)
    newmeshv=[]
    for i, v in enumerate(mesh.v):
      if i==0:
        reference=v
        newmeshv.append(filler)
      else:
        newmeshv.append(tuple(numpy.subtract(v,reference)))
    print newmeshv
    mesh.v=newmeshv
  
  for i, v in enumerate(mesh.v):
    fbx_mesh.SetControlPointAt(as_fvec4(v, scale=100), i)

  layer_elt = create_fbx_layer(
      fbx_mesh, mesh.n, as_fvec4, FbxLayerElementNormal)
  if layer_elt is not None:
    layer0.SetNormals(layer_elt)

  layer_elt = create_fbx_layer(
      fbx_mesh, mesh.c, as_fcolor, FbxLayerElementVertexColor,
      allow_index = True,
      allow_allsame = True)
  if layer_elt is not None:
    layer0.SetVertexColors(layer_elt)

  # Tilt Brush may have 3- or 4-element UV channels, and may have multiple
  # UV channels. This only handles the standard case of 2-component UVs
  layer_elt = create_fbx_layer(
    fbx_mesh, mesh.uv0, as_fvec2, FbxLayerElementUV,
    allow_index = True)
  if layer_elt is not None:
    layer0.SetUVs(layer_elt, FbxLayerElement.eTextureDiffuse)
    pass

  layer_elt = create_fbx_layer(
    fbx_mesh, mesh.t, as_fvec4, FbxLayerElementTangent,
    allow_index = True)
  if layer_elt is not None:
    layer0.SetTangents(layer_elt)

  # Unity's FBX import requires Binormals to be present in order to import the
  # tangents but doesn't actually use them, so we just output some dummy data.
  layer_elt = create_fbx_layer(
    fbx_mesh, ((0, 0, 0, 0),), as_fvec4, FbxLayerElementBinormal,
    allow_allsame = True)
  if layer_elt is not None:
    layer0.SetBinormals(layer_elt)

  layer_elt = create_fbx_layer(
    fbx_mesh, (), lambda x: x, FbxLayerElementMaterial, allow_allsame = True)
  if layer_elt is not None:
    layer0.SetMaterials(layer_elt)

  # Polygons

  for triplet in mesh.tri:
    fbx_mesh.BeginPolygon(-1, -1, False)
    fbx_mesh.AddPolygon(triplet[0])
    fbx_mesh.AddPolygon(triplet[1])
    fbx_mesh.AddPolygon(triplet[2])
    fbx_mesh.EndPolygon()

  material = FbxSurfaceLambert.Create(sdk, mesh.brush_name)
  name=mesh.brush_name+"_"+str(mesh.c[0])+"_"+name
  
  if EXPORT_BRUSH_AREA is True:
    ps=[]
    for t in mesh.v:
      ps.append(list(t))
    #ps2=[]
    #for t in mesh.t:
     # ps2.append(list(t[0:3]))      
   # print len(mesh.tri)
    #print len(mesh.v)
    #print ps
    print  name+","+str(poly_area(ps))
    #print poly_area(ps2)
    #poly = [[0, 3, 1], [0, 2, 3], [2, 5, 3], [2, 4, 5], [4, 7, 5], [4, 6, 7], [6, 9, 7], [6, 8, 9], [8, 11, 9], [8, 10, 11], [10, 13, 11], [10, 12, 13], [12, 15, 13], [12, 14, 15]]
    #print poly_area(poly)  
    global polyareadata
    polyareadata.append(name+","+str(poly_area(ps)))
  
  print name
  mm_save_mesh_metadata(name,mesh)
  #print mesh.brush_name #Roughly analagous to a material
  #print mesh.brush_guid
  #print mesh.v          #list of positions (3-tuples)
  #print mesh.n          #list of normals (3-tuples, or None if missing)
  #print mesh.uv0        #list of uv0 (2-, 3-, 4-tuples, or None if missing)
  #print mesh.uv1        #see uv0
  #print mesh.c          #list of colors, as a uint32. abgr little-endian, rgba big-endian
  #print mesh.t          #list of tangents (4-tuples, or None if missing)
  #print mesh.tri        #list of triangles (3-tuples of ints)
  
  # Node tree

  root = scene.GetRootNode()
  node = FbxNode.Create(sdk, name)
  node.SetNodeAttribute(fbx_mesh)
  node.AddMaterial(material)
  node.SetShadingMode(FbxNode.eTextureShading)  # Hmm
  root.AddChild(node)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
  
  global n
  global export_name
  global metadata
  global polyareadata
  polyareadata=[]
  metadata={}
  metadata['fbxmeta']=[]
  n=1000 #name counter
  
  import argparse
  parser = argparse.ArgumentParser(description="""Converts Tilt Brush '.json' exports to .fbx.""")
  parser.add_argument('filename', help="Exported .json files to convert to fbx",action='store_true')
  grp = parser.add_argument_group(description="Merging and optimization")
  grp.add_argument('--merge-stroke', action='store_true',
                   help="Merge all strokes into a single mesh")

  grp.add_argument('--merge-brush', action='store_true',
                   help="(default) Merge strokes that use the same brush into a single mesh")
  grp.add_argument('--no-merge-brush', action='store_false', dest='merge_brush',
                   help="Turn off --merge-brush")

  grp.add_argument('--weld-verts', action='store_true',
                   help="(default) Weld vertices")
  grp.add_argument('--no-weld-verts', action='store_false', dest='weld_verts',
                   help="Turn off --weld-verts")

  parser.add_argument('--add-backface', action='store_true',
                   help="Add backfaces to strokes that don't have them")

  parser.add_argument('-o', dest='output_filename', metavar='FILE',
                      help="Name of output file; defaults to <filename>.fbx")
  parser.set_defaults(merge_brush=True, weld_verts=True)
  args = parser.parse_args()
  args.filename=FNAME
  args.merge_brush=MERGE_BRUSH
  

  if args.output_filename is None:
    args.output_filename = os.path.splitext(args.filename)[0] + '.fbx'
  
  export_name=os.path.splitext(args.filename)[0]+"_fbx_metadata.json"

  meshes = list(iter_meshes(args.filename))
  for mesh in meshes:
    mesh.remove_degenerate()
    if args.add_backface and mesh.brush_guid in SINGLE_SIDED_FLAT_BRUSH:
      mesh.add_backface()

  if args.merge_stroke:
    meshes = [ TiltBrushMesh.from_meshes(meshes, name='strokes') ]
  elif args.merge_brush:
    def by_guid(m): return (m.brush_guid, m.brush_name)
    meshes = [ TiltBrushMesh.from_meshes(list(group), name='All %s' % (key[1], ))
               for (key, group) in groupby(sorted(meshes, key=by_guid), key=by_guid) ]

  if args.weld_verts:
    for mesh in meshes:
      # We don't write out tangents, so it's safe to ignore them when welding
      mesh.collapse_verts(ignore=('t',))
      mesh.remove_degenerate()


  if EXPORT_RELOCATION==1 or RELOCATE_BRUSHES is False:
    write_fbx_meshes(meshes, args.output_filename)
  
  if EXPORT_RELOCATION==2:
    #BRUSH IN INDIVIDUAL FILE
    for mesh in meshes:
      write_fbx_meshes(mesh, args.output_filename)
    
  metadata['fbxname']=args.output_filename
  print "Wrote", args.output_filename
  
  if EXPORT_BRUSH_AREA is True:
    areaexport_name=os.path.splitext(args.filename)[0]+"_brushsize.txt"
    print areaexport_name
    with open(areaexport_name, 'w') as f:
        json.dump(polyareadata, f)      
  
  with open(export_name, 'w') as f:
      json.dump(metadata, f)     
  

        
  print "Wrote", export_name


if __name__ == '__main__':
  main()

#MM Metadata structure:
#{'fbxname':FBXNAME,
 #'fbxmeta': 
     #[{'meshname':MESHNAME,
       #'meshmeta':
          #{'brush_name':BRUSHNAME,
           #'brush_guid':BRUSH_GUID,
           #'v':V,
           #'n':N,
           #'uv0':UV0,
           #'uv1':UV1,
           #'c':C,
           #'t':T,
           #'tri':TRI
           #}
       #},{},...]     
#}