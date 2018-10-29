
# The RenderPasses and UI code are kept in this file to make it executable in Blender without the need to install it as addon.

import bpy
import os
import sys
import random

from mathutils import Vector

class RenderPasses:
  COMBINED = 'Combined'
  ALPHA = 'Alpha'
  DEPTH = 'Depth'
  MIST = 'Mist'
  NORMAL = 'Normal'
  SCREEN_SPACE_NORMAL = 'Screen Space Normal'
  MOTION_VECTOR = 'Motion Vector'
  OBJECT_ID = 'Object ID'
  MATERIAL_ID = 'Material ID'
  UV = 'UV'
  SHADOW = 'Shadow'
  AMBIENT_OCCLUSION = 'Ambient Occlusion'
  EMISSION = 'Emission'
  ENVIRONMENT = 'Environment'
  DIFFUSE_DIRECT = 'Diffuse Direct'
  DIFFUSE_INDIRECT = 'Diffuse Indirect'
  DIFFUSE_COLOR = 'Diffuse Color'
  GLOSSY_DIRECT = 'Glossy Direct'
  GLOSSY_INDIRECT = 'Glossy Indirect'
  GLOSSY_COLOR = 'Glossy Color'
  TRANSMISSION_DIRECT = 'Transmission Direct'
  TRANSMISSION_INDIRECT = 'Transmission Indirect'
  TRANSMISSION_COLOR = 'Transmission Color'
  SUBSURFACE_DIRECT = 'Subsurface Direct'
  SUBSURFACE_INDIRECT = 'Subsurface Indirect'
  SUBSURFACE_COLOR = 'Subsurface Color'
  VOLUME_DIRECT = 'Volume Direct'
  VOLUME_INDIRECT = 'Volume Indirect'


class DeepDenoiserRender:

  @staticmethod
  def prepare_image_settings():
    render = bpy.context.scene.render
    image_settings = render.image_settings
    
    image_settings.file_format = 'OPEN_EXR'
    image_settings.color_mode = 'RGBA'
    image_settings.color_depth = '32'
    image_settings.exr_codec = 'ZIP'

    render.use_border = False
    render.use_crop_to_border = False
    render.use_file_extension = True
    render.use_stamp = False

    # Save buffers have to be disabled due to a bug.
    # If it is enabled, the volumetric passes are not saved.
    render.use_save_buffers = False
  
  @staticmethod
  def prepare_cycles():
    bpy.context.scene.render.engine = 'CYCLES'
    scene = bpy.context.scene
    cycles = scene.cycles
    
    # No branched path tracing for now.
    cycles.progressive = 'PATH'
    
    cycles_render_layer = scene.render.layers.active.cycles
    cycles_render_layer.use_denoising = False
  
  @staticmethod
  def prepare_passes():
    render_layer = bpy.context.scene.render.layers[0]
    cycles_render_layer = render_layer.cycles

    render_layer.use_pass_diffuse_direct = True
    render_layer.use_pass_diffuse_indirect = True
    render_layer.use_pass_diffuse_color = True

    render_layer.use_pass_glossy_direct = True
    render_layer.use_pass_glossy_indirect = True
    render_layer.use_pass_glossy_color = True

    render_layer.use_pass_transmission_direct = True
    render_layer.use_pass_transmission_indirect = True
    render_layer.use_pass_transmission_color = True

    render_layer.use_pass_subsurface_direct = True
    render_layer.use_pass_subsurface_indirect = True
    render_layer.use_pass_subsurface_color = True

    cycles_render_layer.use_pass_volume_direct = True
    cycles_render_layer.use_pass_volume_indirect = True

    render_layer.use_pass_combined = True
    
    render_layer.use_pass_z = True
    render_layer.use_pass_mist = True
    render_layer.use_pass_normal = True
    render_layer.use_pass_vector = True

    render_layer.use_pass_object_index = True
    render_layer.use_pass_material_index = True
    render_layer.use_pass_uv = True
    
    render_layer.use_pass_emit = True
    render_layer.use_pass_environment = True

    render_layer.use_pass_shadow = True
    render_layer.use_pass_ambient_occlusion = True

  @staticmethod  
  def prepare_compositor(target_folder):
    bpy.context.scene.render.use_compositing = True
    
    bpy.context.scene.use_nodes = True
    node_tree = bpy.context.scene.node_tree
    
    for node in node_tree.nodes:
      node_tree.nodes.remove(node)
    
    input_node = node_tree.nodes.new('CompositorNodeRLayers')
    output_node = node_tree.nodes.new('CompositorNodeOutputFile')
    output_node.layer_slots.clear()
    output_node.location = (300, 0)
    
    samples_per_pixel = bpy.context.scene.cycles.samples
    relative_frame_number = bpy.context.scene.frame_current
    seed = bpy.context.scene.cycles.seed
    
    path = os.path.join(
        target_folder, DeepDenoiserRender.blend_filename() + '_' +
        str(samples_per_pixel) + '_' + str(relative_frame_number) + '_' + str(seed))
    path = bpy.path.abspath(path)
    path = os.path.realpath(path)
    output_node.base_path = path

    
    links = node_tree.links
    
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'Image', output_node, RenderPasses.COMBINED)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'Alpha', output_node, RenderPasses.ALPHA)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'Depth', output_node, RenderPasses.DEPTH)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'Normal', output_node, RenderPasses.NORMAL)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'UV', output_node, RenderPasses.UV)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'Vector', output_node, RenderPasses.MOTION_VECTOR)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'Shadow', output_node, RenderPasses.SHADOW)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'AO', output_node, RenderPasses.AMBIENT_OCCLUSION)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'IndexOB', output_node, RenderPasses.OBJECT_ID)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'IndexMA', output_node, RenderPasses.MATERIAL_ID)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'Mist', output_node, RenderPasses.MIST)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'Emit', output_node, RenderPasses.EMISSION)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'Env', output_node, RenderPasses.ENVIRONMENT)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'DiffDir', output_node, RenderPasses.DIFFUSE_DIRECT)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'DiffInd', output_node, RenderPasses.DIFFUSE_INDIRECT)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'DiffCol', output_node, RenderPasses.DIFFUSE_COLOR)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'GlossDir', output_node, RenderPasses.GLOSSY_DIRECT)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'GlossInd', output_node, RenderPasses.GLOSSY_INDIRECT)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'GlossCol', output_node, RenderPasses.GLOSSY_COLOR)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'TransDir', output_node, RenderPasses.TRANSMISSION_DIRECT)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'TransInd', output_node, RenderPasses.TRANSMISSION_INDIRECT)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'TransCol', output_node, RenderPasses.TRANSMISSION_COLOR)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'SubsurfaceDir', output_node, RenderPasses.SUBSURFACE_DIRECT)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'SubsurfaceInd', output_node, RenderPasses.SUBSURFACE_INDIRECT)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'SubsurfaceCol', output_node, RenderPasses.SUBSURFACE_COLOR)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'VolumeDir', output_node, RenderPasses.VOLUME_DIRECT)
    DeepDenoiserRender.connect_pass_to_new_file_output(
        links, input_node, 'VolumeInd', output_node, RenderPasses.VOLUME_INDIRECT)
    
    viewer_node = node_tree.nodes.new('CompositorNodeViewer')
    viewer_node.location = (300, 150)
    links.new(input_node.outputs[RenderPasses.NORMAL], viewer_node.inputs[0])

  @staticmethod
  def connect_pass_to_new_file_output(links, input_node, pass_name, output_node, short_output_name):
    output_name = DeepDenoiserRender.extended_name(short_output_name)
    output_slot = output_node.layer_slots.new(output_name)
    links.new(input_node.outputs[pass_name], output_slot)

  @staticmethod
  def blend_filename():
    blend_filename = bpy.path.basename(bpy.context.blend_data.filepath)
    result = os.path.splitext(blend_filename)[0]
    result = result.replace('_', ' ')
    return result

  @staticmethod
  def extended_name(name):
    samples_per_pixel = bpy.context.scene.cycles.samples
    relative_frame_number = bpy.context.scene.frame_current
    seed = bpy.context.scene.cycles.seed
    
    result = (
        DeepDenoiserRender.blend_filename() + '_' +
        str(samples_per_pixel) + '_' + str(relative_frame_number) + '_' + str(seed) + '_' +
        name + '_')
    return result
  
  @staticmethod
  def is_render_layer_valid():
    result = False
    if (
        len(bpy.context.scene.render.layers) == 1 and
        bpy.context.scene.render.layers[0].use):
      result = True
    return result
  
  @staticmethod
  def render(target_folder):      
    DeepDenoiserRender.prepare_passes()
    DeepDenoiserRender.prepare_image_settings()
    DeepDenoiserRender.prepare_cycles()
    DeepDenoiserRender.prepare_compositor(target_folder)
    bpy.ops.render.render()


# UI

class DeepDenoiserRenderJobPropertyGroup(bpy.types.PropertyGroup):
  use_render_job = bpy.props.BoolProperty(name='use_render_job')
  

class DeepDenoiserRenderPropertyGroup(bpy.types.PropertyGroup):
  target_folder = bpy.props.StringProperty(
      name='target_folder', description='Base directory for the rendered results', default='//OpenEXR/', maxlen=1024, subtype='DIR_PATH')
  render_jobs_initialized = bpy.props.BoolProperty(
      name='render_jobs_initialized', description='Were the render jobs initialized. Only false when the script was never initialized',
      default=False)


class RENDER_JOB_prepare(bpy.types.Operator):
  bl_idname = 'deep_blender_render.prepare'
  bl_label = "Prepare Settings"
  bl_description = "Prepare all the Blender settings to experiment with the same settings that are used when rendering"

  def execute(self, context):
    target_folder = context.scene.deep_denoiser_generator_property_group.target_folder
    
    DeepDenoiserRender.prepare_passes()
    DeepDenoiserRender.prepare_image_settings()
    DeepDenoiserRender.prepare_cycles()
    DeepDenoiserRender.prepare_compositor(target_folder)
    
    return{'FINISHED'}


class RENDER_JOB_render(bpy.types.Operator):
  bl_idname = 'deep_blender_render.render'
  bl_label = "Render main frame noiseless"
  bl_description = "Render main frame noiseless"

  def execute(self, context):
    target_folder = context.scene.deep_denoiser_generator_property_group.target_folder
    
    samples_per_pixel = bpy.context.scene.cycles.samples
    DeepDenoiserRender.render(target_folder)
    return{'FINISHED'}



class DeepDenoiserRenderPanel(bpy.types.Panel):
  bl_label = "DeepDenoiser Render"
  bl_idname = "CATEGORY_PT_DeepDenoiserRender"
  bl_space_type = 'VIEW_3D'
  bl_region_type = 'TOOLS'
  bl_category = "DeepBlender"

  def draw(self, context):
    scene = context.scene
    layout = self.layout
    
    render = scene.render
    
    column = layout.column()
    column.operator('deep_blender_render.prepare')
    
    column = layout.column()
    column.prop(scene.deep_denoiser_generator_property_group, 'target_folder', text='Folder')
            
    is_render_layer_valid = DeepDenoiserRender.is_render_layer_valid()
    if not is_render_layer_valid:
      box = layout.box()
      inner_column = box.column()
      inner_column.label(text="Render Layer Error", icon='ERROR')
      inner_column.label(text="The scene is only allowed to have one render layer and that one has to be active!")
    
    
    column = layout.column()
    if not is_render_layer_valid:
      column.enabled = False
    column.label(text="Render:")

    column.operator('deep_blender_render.render', text='Render', icon='RENDER_STILL')
    

classes = [
    DeepDenoiserRenderJobPropertyGroup, DeepDenoiserRenderPropertyGroup, DeepDenoiserRenderPanel,
    RENDER_JOB_prepare, RENDER_JOB_render]

def register():
  for i in classes:
    bpy.utils.register_class(i)

  # Store properties in the scene
  bpy.types.Scene.deep_denoiser_generator_property_group = bpy.props.PointerProperty(type=DeepDenoiserRenderPropertyGroup)
  bpy.types.Scene.render_jobs = bpy.props.CollectionProperty(type=DeepDenoiserRenderJobPropertyGroup)
  
def unregister():
  for i in classes:
    bpy.utils.unregister_class(i)


if __name__ == "__main__":
  register()