# TODO: Option for progressive rendering with merging (DeepBlender)


import bpy
import os
import sys
import random

from mathutils import Vector

class RenderPasses:
  COMPOSED = 'Composed'
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


class DeepDenoiserDataGenerator:

  def prepare_image_settings(width, height):
    render = bpy.context.scene.render
    image_settings = render.image_settings
    
    image_settings.file_format = 'OPEN_EXR'
    image_settings.color_mode = 'RGBA'
    image_settings.color_depth = '32'
    image_settings.exr_codec = 'ZIP'
    
    render.resolution_x = width
    render.resolution_y = height
    render.resolution_percentage = 100

    render.use_border = False
    render.use_crop_to_border = False
    render.use_file_extension = True
    render.use_stamp = False


  def prepare_cycles(samples_per_pixel, seed):
  
    bpy.context.scene.render.engine = 'CYCLES'
  
    cycles = bpy.context.scene.cycles

    cycles.progressive = 'PATH'
    
    cycles.use_square_samples = False
    cycles.samples = samples_per_pixel
    
    cycles.seed = seed
    cycles.use_animated_seed = False
    
    cycles.use_transparent_shadows = True
    cycles.transparent_min_bounces = 8
    cycles.transparent_max_bounces = 128
    
    cycles.min_bounces = 3
    cycles.max_bounces = 128
    
    cycles.diffuse_bounces = 128
    cycles.glossy_bounces = 128
    cycles.transmission_bounces = 128
    cycles.volume_bounces = 128
    
    cycles.caustics_reflective = True
    cycles.caustics_refractive = True
    cycles.blur_glossy = 0.0
    
    cycles.sample_clamp_direct = 0.0
    cycles.sample_clamp_indirect = 0.0
    cycles.light_sampling_threshold = 0.0

    cycles_render_layer = bpy.context.scene.render.layers.active.cycles
    cycles_render_layer.use_denoising = False
    
    bpy.context.scene.render.use_motion_blur = False

  
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

    render_layer.pass_alpha_threshold = 0.5
    
    
  def prepare_compositor(target_folder, samples_per_pixel):
    bpy.context.scene.render.use_compositing = True
    
    bpy.context.scene.use_nodes = True
    node_tree = bpy.context.scene.node_tree
    
    for node in node_tree.nodes:
      node_tree.nodes.remove(node)
    
    input_node = node_tree.nodes.new('CompositorNodeRLayers')
    output_node = node_tree.nodes.new('CompositorNodeOutputFile')
    output_node.layer_slots.clear()
    output_node.location = (300, 0)
    output_node.base_path = target_folder + '/' + DeepDenoiserDataGenerator.blend_filename() + '/' + DeepDenoiserDataGenerator.blend_filename() + '_' + str(samples_per_pixel) + '_' + str(bpy.context.scene.cycles.seed)
    
    links = node_tree.links
    
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'Image', output_node, RenderPasses.COMPOSED, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'Alpha', output_node, RenderPasses.ALPHA, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'Depth', output_node, RenderPasses.DEPTH, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'Normal', output_node, RenderPasses.NORMAL, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'UV', output_node, RenderPasses.UV, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'Vector', output_node, RenderPasses.MOTION_VECTOR, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'Shadow', output_node, RenderPasses.SHADOW, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'AO', output_node, RenderPasses.AMBIENT_OCCLUSION, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'IndexOB', output_node, RenderPasses.OBJECT_ID, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'IndexMA', output_node, RenderPasses.MATERIAL_ID, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'Mist', output_node, RenderPasses.MIST, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'Emit', output_node, RenderPasses.EMISSION, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'Env', output_node, RenderPasses.ENVIRONMENT, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'DiffDir', output_node, RenderPasses.DIFFUSE_DIRECT, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'DiffInd', output_node, RenderPasses.DIFFUSE_INDIRECT, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'DiffCol', output_node, RenderPasses.DIFFUSE_COLOR, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'GlossDir', output_node, RenderPasses.GLOSSY_DIRECT, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'GlossInd', output_node, RenderPasses.GLOSSY_INDIRECT, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'GlossCol', output_node, RenderPasses.GLOSSY_COLOR, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'TransDir', output_node, RenderPasses.TRANSMISSION_DIRECT, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'TransInd', output_node, RenderPasses.TRANSMISSION_INDIRECT, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'TransCol', output_node, RenderPasses.TRANSMISSION_COLOR, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'SubsurfaceDir', output_node, RenderPasses.SUBSURFACE_DIRECT, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'SubsurfaceInd', output_node, RenderPasses.SUBSURFACE_INDIRECT, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'SubsurfaceCol', output_node, RenderPasses.SUBSURFACE_COLOR, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'VolumeDir', output_node, RenderPasses.VOLUME_DIRECT, samples_per_pixel)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(links, input_node, 'VolumeInd', output_node, RenderPasses.VOLUME_INDIRECT, samples_per_pixel)
    
    viewer_node = node_tree.nodes.new('CompositorNodeViewer')
    viewer_node.location = (300, 150)
    links.new(input_node.outputs[RenderPasses.NORMAL], viewer_node.inputs[0])

  def prepare_world():
    # Settings for multi importance sampling
    cycles = bpy.context.scene.world.cycles
    cycles.sample_map_resolution = 1024
    cycles.max_bounces = 1024
    
  def connect_pass_to_new_file_output(links, input_node, pass_name, output_node, short_output_name, samples_per_pixel):
    output_name = DeepDenoiserDataGenerator.extended_name(short_output_name, samples_per_pixel)
    output_slot = output_node.layer_slots.new(output_name)
    links.new(input_node.outputs[pass_name], output_node.inputs[output_name])
    
    
  def blend_filename():
    blend_filename = bpy.path.basename(bpy.context.blend_data.filepath)
    result = os.path.splitext(blend_filename)[0]
    result = result.replace('_', ' ')
    return result

    
  def extended_name(name, samples_per_pixel):
    result = DeepDenoiserDataGenerator.blend_filename() + '_' + str(samples_per_pixel)+ '_'  + str(bpy.context.scene.cycles.seed) + '_' + name + '_'
    return result
    
    
  def calculate_screen_space_normals(target_folder, samples_per_pixel):

    image = bpy.data.images['Viewer Node']
    pixels = image.pixels
    
    # Copy the pixels for faster access.
    pixels = list(image.pixels)

    camera = bpy.context.scene.camera
    camera_rotation = camera.rotation_euler.to_quaternion()
    camera_rotation.invert()

    for i in range(0, len(pixels), 4):
      normal = Vector((pixels[i+0], pixels[i+1], pixels[i+2]))
      computed_screen_space_normal = camera_rotation * normal
      pixels[i+0] = computed_screen_space_normal[0]
      pixels[i+1] = computed_screen_space_normal[1]
      pixels[i+2] = computed_screen_space_normal[2]

    width = image.size[0]
    height = image.size[1]
    
    if RenderPasses.SCREEN_SPACE_NORMAL in bpy.data.images:
      bpy.data.images.remove(bpy.data.images[RenderPasses.SCREEN_SPACE_NORMAL])
    
    screen_space_normal_image = bpy.data.images.new(name=RenderPasses.SCREEN_SPACE_NORMAL, width=width, height=height, float_buffer=True)
    
    current_frame = bpy.context.scene.frame_current
    current_frame = format(current_frame, '04d')
    screen_space_normal_image.pixels[:] = pixels
    
    file_path = target_folder + '/' + DeepDenoiserDataGenerator.blend_filename() + '/' + DeepDenoiserDataGenerator.blend_filename() + '_' + str(samples_per_pixel) + '_' + str(bpy.context.scene.cycles.seed) + '/' + DeepDenoiserDataGenerator.blend_filename() + '_' + str(samples_per_pixel) + '_' + str(bpy.context.scene.cycles.seed) + '_' + RenderPasses.SCREEN_SPACE_NORMAL + '_' + current_frame + '.exr'
    file_path = bpy.path.abspath(file_path)
    screen_space_normal_image.save_render(file_path)
    
    
  def render(target_folder, width, height, samples_per_pixel, seed):
    DeepDenoiserDataGenerator.prepare_passes()
    DeepDenoiserDataGenerator.prepare_image_settings(width, height)
    DeepDenoiserDataGenerator.prepare_cycles(samples_per_pixel, seed)
    DeepDenoiserDataGenerator.prepare_compositor(target_folder, samples_per_pixel)
    DeepDenoiserDataGenerator.prepare_world()
    bpy.ops.render.render()
    DeepDenoiserDataGenerator.calculate_screen_space_normals(target_folder, samples_per_pixel)


seed_min = int(0)
seed_max = int(1e5)

class DeepDenoiserRenderJobPropertyGroup(bpy.types.PropertyGroup):
  use_render_job = bpy.props.BoolProperty(name='use_render_job')
  samples_per_pixel = bpy.props.IntProperty(name='samples_per_pixel', default=1, min=1)
  number_of_renders = bpy.props.IntProperty(name='number_of_renders', default=1, min=1)
  

class DeepDenoiserDataGeneratorPropertyGroup(bpy.types.PropertyGroup):
  target_folder = bpy.props.StringProperty(name='target_folder', description='Base directory for the rendered results.', default='//OpenEXR/', maxlen=1024, subtype='DIR_PATH')
  seed = bpy.props.IntProperty(name='seed', description='Seed used as basis for the rendering, such that there is a strong variation with deterministic results.', default=0, min=seed_min, max=seed_max)
  
  
class RandomizeSeedOperator(bpy.types.Operator):
  bl_idname = "deep_blender.randomize_seed_operator"
  bl_label = "Randomize Seed Operator"

  def execute(self, context):
    random.seed(context.scene.deep_denoiser_generator_property_group.seed)
    context.scene.deep_denoiser_generator_property_group.seed = random.randint(seed_min, seed_max)
    return {'FINISHED'}


class RENDER_JOB_OT_add(bpy.types.Operator):
  bl_idname = 'deep_blender.render_jobs_add'
  bl_label = "Add render job"
  bl_description = "Add render job"

  def execute(self, context):
    render_jobs = context.scene.render_jobs
    render_jobs.add()
    return{'FINISHED'}
 
 
class RENDER_JOB_OT_remove(bpy.types.Operator):
  bl_idname = 'deep_blender.render_jobs_remove'
  bl_label = "Remove render job"
  bl_description = "Remove render job"

  def execute(self, context):
    render_jobs = context.scene.render_jobs
    selected_render_job_index = context.scene.selected_render_job_index
     
    if selected_render_job_index >= 0:
       render_jobs.remove(selected_render_job_index)
       context.scene.selected_render_job_index -= 1
    return{'FINISHED'}
    
class RENDER_JOB_OT_move_up(bpy.types.Operator):
  bl_idname = 'deep_blender.render_jobs_move_up'
  bl_label = "Remove render job"
  bl_description = "Remove render job"

  def execute(self, context):
    render_jobs = context.scene.render_jobs
    selected_render_job_index = context.scene.selected_render_job_index
    
    if selected_render_job_index > 0:
       render_jobs.move(selected_render_job_index, selected_render_job_index - 1)
       context.scene.selected_render_job_index -= 1
    return{'FINISHED'}
    
class RENDER_JOB_OT_move_down(bpy.types.Operator):
  bl_idname = 'deep_blender.render_jobs_move_down'
  bl_label = "Remove render job"
  bl_description = "Remove render job"

  def execute(self, context):
    render_jobs = context.scene.render_jobs
    selected_render_job_index = context.scene.selected_render_job_index
    
    if selected_render_job_index != -1 and selected_render_job_index < len(render_jobs) - 1:
       render_jobs.move(selected_render_job_index, selected_render_job_index + 1)
       context.scene.selected_render_job_index += 1
    return{'FINISHED'}

class RENDER_JOB_render(bpy.types.Operator):
  bl_idname = 'deep_blender.render'
  bl_label = "Render"
  bl_description = "Render"

  def execute(self, context):
    render = bpy.context.scene.render
    target_folder = context.scene.deep_denoiser_generator_property_group.target_folder
    render_jobs = context.scene.render_jobs
    for render_job in render_jobs:
      seed = context.scene.deep_denoiser_generator_property_group.seed
      if render_job.use_render_job:
        for i in range(render_job.number_of_renders):
          if i == 0:
            current_seed = seed
          elif i == 1:
            current_seed = current_seed + render_job.samples_per_pixel
          else:
            random.seed(current_seed)
            current_seed = random.randint(seed_min, seed_max)
          
          DeepDenoiserDataGenerator.render(target_folder, render.resolution_x, render.resolution_y, render_job.samples_per_pixel, current_seed)
    return{'FINISHED'}


class DeepDenoiserItemUI(bpy.types.UIList):
  def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
    render_job = item
    layout.prop(render_job, 'use_render_job', text='')
    layout.prop(render_job, 'samples_per_pixel', text='Samples')
    layout.prop(render_job, 'number_of_renders', text='Renders')

class DeepDenoiserDataGeneratorPanel(bpy.types.Panel):
  bl_label = "DeepDenoiser Data Generator"
  bl_idname = "CATEGORY_PT_DeepDenoiserDataGenerator"
  bl_space_type = 'VIEW_3D'
  bl_region_type = 'TOOLS'
  bl_category = "DeepBlender"

  def draw(self, context):
    scene = context.scene
    layout = self.layout

    row = layout.row()
    row.prop(scene.deep_denoiser_generator_property_group, 'seed', text='Seed')
    row.operator('deep_blender.randomize_seed_operator', icon='FILE_REFRESH', text='')
    
    render = scene.render
    column = layout.column()
    column.label(text="Resolution:")
    column.prop(render, "resolution_x", text="X")
    column.prop(render, "resolution_y", text="Y")
    
    column.prop(scene.world.cycles, "sample_as_light", text="Multiple Importance Sampling")
    
    layout.prop(scene.deep_denoiser_generator_property_group, 'target_folder', text='Folder')
    
    row = layout.row()
    row.template_list('DeepDenoiserItemUI', 'compact', scene, 'render_jobs', scene, 'selected_render_job_index')
    col = row.column(align=True)
    col.operator('deep_blender.render_jobs_add', icon='ZOOMIN', text='')
    col.operator('deep_blender.render_jobs_remove', icon='ZOOMOUT', text='')
    
    if len(scene.render_jobs) > 1 and scene.selected_render_job_index >= 0:
      col.separator()
      col.operator('deep_blender.render_jobs_move_up', icon='TRIA_UP', text='')
      col.operator('deep_blender.render_jobs_move_down', icon='TRIA_DOWN', text='')

    row = layout.row()
    row.operator('deep_blender.render', text='Render', icon='RENDER_STILL')


classes = (DeepDenoiserRenderJobPropertyGroup, DeepDenoiserDataGeneratorPropertyGroup, DeepDenoiserDataGeneratorPanel, DeepDenoiserItemUI, RandomizeSeedOperator, RENDER_JOB_OT_add, RENDER_JOB_OT_remove, RENDER_JOB_OT_move_up, RENDER_JOB_OT_move_down, RENDER_JOB_render)

def register():
  for i in classes:
    bpy.utils.register_class(i)

  # Store properties in the scene
  bpy.types.Scene.deep_denoiser_generator_property_group = bpy.props.PointerProperty(type=DeepDenoiserDataGeneratorPropertyGroup)
  bpy.types.Scene.render_jobs = bpy.props.CollectionProperty(type=DeepDenoiserRenderJobPropertyGroup)
  bpy.types.Scene.selected_render_job_index = bpy.props.IntProperty(name='selected_render_job_index', description= "Selected render job index", default= -1, min= -1)
  
def unregister():
  for i in classes:
    bpy.utils.unregister_class(i)


if __name__ == "__main__":
  register()