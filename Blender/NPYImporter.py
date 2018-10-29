import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty
from mathutils import Vector

import os
import numpy as np



class BrowseNPY(bpy.types.Operator, ImportHelper):
  bl_idname = 'deep_blender.browse_npy'
  bl_label = "Import NPY..."
  bl_description = "Import an NPY as texture/image."

  filename_ext = ".npy"

  filter_glob = StringProperty(
      default="*.npy",
      options={'HIDDEN'},
      maxlen=255)

  def execute(self, context):
    npy_image = np.load(self.filepath)
    if npy_image.shape[2] == 3:
      ones = np.ones((npy_image.shape[0], npy_image.shape[1], 1))
      npy_image = np.concatenate((npy_image, ones), axis=2)
    npy_image = np.flip(npy_image, axis=0)

    image_name = os.path.basename(self.filepath)
    image_name = os.path.splitext(image_name)[0]
    
    image = bpy.data.images.new(image_name, width=npy_image.shape[1], height=npy_image.shape[0], float_buffer=True)
    image.pixels = npy_image.ravel()

    return{'FINISHED'}


class NPYImporterPanel(bpy.types.Panel):
  bl_label = "NPY Importer"
  bl_idname = "CATEGORY_PT_NPY_IMPORTER"
  bl_space_type = 'VIEW_3D'
  bl_region_type = 'TOOLS'
  bl_category = "DeepBlender"

  def draw(self, context):
    layout = self.layout
        
    column = layout.column()
    column.operator('deep_blender.browse_npy', text='Import NPY...')


classes = [NPYImporterPanel, BrowseNPY]

def register():
  for i in classes:
    bpy.utils.register_class(i)

  
def unregister():
  for i in classes:
    bpy.utils.unregister_class(i)


if __name__ == "__main__":
  register()