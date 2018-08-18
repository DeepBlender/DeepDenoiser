
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


class DeepDenoiserDataGenerator:

  def prepare_image_settings():
    render = bpy.context.scene.render
    image_settings = render.image_settings
    
    image_settings.file_format = 'OPEN_EXR'
    image_settings.color_mode = 'RGBA'
    image_settings.color_depth = '32'
    image_settings.exr_codec = 'ZIP'

    render.resolution_percentage = 100

    render.use_border = False
    render.use_crop_to_border = False
    render.use_file_extension = True
    render.use_stamp = False

  def prepare_relative_frame_number(relative_frame_number):
    bpy.context.scene.frame_current = bpy.context.scene.deep_denoiser_generator_property_group.main_frame + relative_frame_number
    assert relative_frame_number == DeepDenoiserDataGenerator.current_relative_frame_number()
  
  def prepare_cycles(samples_per_pixel, seed):
    bpy.context.scene.render.engine = 'CYCLES'
    scene = bpy.context.scene
    cycles = scene.cycles
    
    cycles.use_square_samples = False
    cycles.samples = samples_per_pixel
    
    cycles.seed = seed
    cycles.use_animated_seed = False
    
    cycles_render_layer = scene.render.layers.active.cycles
    cycles_render_layer.use_denoising = False
    
    # Those should be learned by the DeepDenoiser.
    cycles.blur_glossy = 0.0
    cycles.sample_clamp_direct = 0.0
    cycles.sample_clamp_indirect = 0.0
    cycles.light_sampling_threshold = 0.0
  
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
    
  def prepare_compositor(target_folder):
    samples_per_pixel = DeepDenoiserDataGenerator.current_samples_per_pixel()
    relative_frame_number = DeepDenoiserDataGenerator.current_relative_frame_number()
    seed = DeepDenoiserDataGenerator.current_seed()
  
    bpy.context.scene.render.use_compositing = True
    
    bpy.context.scene.use_nodes = True
    node_tree = bpy.context.scene.node_tree
    
    for node in node_tree.nodes:
      node_tree.nodes.remove(node)
    
    input_node = node_tree.nodes.new('CompositorNodeRLayers')
    output_node = node_tree.nodes.new('CompositorNodeOutputFile')
    output_node.layer_slots.clear()
    output_node.location = (300, 0)
    output_node.base_path = (
        target_folder + '/' + DeepDenoiserDataGenerator.blend_filename() + '/' +
        DeepDenoiserDataGenerator.blend_filename() + '_' +
        str(samples_per_pixel) + '_' + str(relative_frame_number) + '_' + str(seed))

    
    links = node_tree.links
    
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'Image', output_node, RenderPasses.COMBINED)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'Alpha', output_node, RenderPasses.ALPHA)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'Depth', output_node, RenderPasses.DEPTH)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'Normal', output_node, RenderPasses.NORMAL)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'UV', output_node, RenderPasses.UV)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'Vector', output_node, RenderPasses.MOTION_VECTOR)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'Shadow', output_node, RenderPasses.SHADOW)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'AO', output_node, RenderPasses.AMBIENT_OCCLUSION)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'IndexOB', output_node, RenderPasses.OBJECT_ID)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'IndexMA', output_node, RenderPasses.MATERIAL_ID)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'Mist', output_node, RenderPasses.MIST)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'Emit', output_node, RenderPasses.EMISSION)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'Env', output_node, RenderPasses.ENVIRONMENT)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'DiffDir', output_node, RenderPasses.DIFFUSE_DIRECT)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'DiffInd', output_node, RenderPasses.DIFFUSE_INDIRECT)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'DiffCol', output_node, RenderPasses.DIFFUSE_COLOR)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'GlossDir', output_node, RenderPasses.GLOSSY_DIRECT)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'GlossInd', output_node, RenderPasses.GLOSSY_INDIRECT)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'GlossCol', output_node, RenderPasses.GLOSSY_COLOR)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'TransDir', output_node, RenderPasses.TRANSMISSION_DIRECT)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'TransInd', output_node, RenderPasses.TRANSMISSION_INDIRECT)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'TransCol', output_node, RenderPasses.TRANSMISSION_COLOR)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'SubsurfaceDir', output_node, RenderPasses.SUBSURFACE_DIRECT)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'SubsurfaceInd', output_node, RenderPasses.SUBSURFACE_INDIRECT)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'SubsurfaceCol', output_node, RenderPasses.SUBSURFACE_COLOR)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'VolumeDir', output_node, RenderPasses.VOLUME_DIRECT)
    DeepDenoiserDataGenerator.connect_pass_to_new_file_output(
        links, input_node, 'VolumeInd', output_node, RenderPasses.VOLUME_INDIRECT)
    
    viewer_node = node_tree.nodes.new('CompositorNodeViewer')
    viewer_node.location = (300, 150)
    links.new(input_node.outputs[RenderPasses.NORMAL], viewer_node.inputs[0])

  def connect_pass_to_new_file_output(links, input_node, pass_name, output_node, short_output_name):
    output_name = DeepDenoiserDataGenerator.extended_name(short_output_name)
    output_slot = output_node.layer_slots.new(output_name)
    links.new(input_node.outputs[pass_name], output_node.inputs[output_name])
      
  def blend_filename():
    blend_filename = bpy.path.basename(bpy.context.blend_data.filepath)
    result = os.path.splitext(blend_filename)[0]
    result = result.replace('_', ' ')
    return result

  def extended_name(name):
    samples_per_pixel = DeepDenoiserDataGenerator.current_samples_per_pixel()
    relative_frame_number = DeepDenoiserDataGenerator.current_relative_frame_number()
    seed = DeepDenoiserDataGenerator.current_seed()
    
    result = (
        DeepDenoiserDataGenerator.blend_filename() + '_' +
        str(samples_per_pixel) + '_' + str(relative_frame_number) + '_' + str(seed) + '_' +
        name + '_')
    return result
  
  def reset_render_jobs():
    render_jobs = bpy.context.scene.render_jobs
    
    render_jobs.clear()
    samples_per_pixel_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for samples_per_pixel in samples_per_pixel_list:
      render_job = render_jobs.add()
      render_job.use_render_job = True
      render_job.samples_per_pixel = samples_per_pixel
      render_job.number_of_renders = 2
    
    bpy.context.scene.selected_render_job_index = 0
  
  def current_relative_frame_number():
    scene = bpy.context.scene
    result = scene.frame_current - scene.deep_denoiser_generator_property_group.main_frame
    return result
  
  def current_samples_per_pixel():
    result = bpy.context.scene.cycles.samples
    return result
  
  def current_seed():
    result = bpy.context.scene.cycles.seed
    return result
  
  def seed(samples_per_pixel, samples_per_pixel_render_index, relative_frame_number):
    initial_seed = bpy.context.scene.deep_denoiser_generator_property_group.seed
    seed = random.seed(initial_seed + samples_per_pixel)
    for _ in range(samples_per_pixel_render_index + 1):
      seed = random.randint(seed_min, seed_max)
    seed = seed + relative_frame_number
    return seed
  
  def is_resolution_valid():
    result = True
    resolution_x = bpy.context.scene.render.resolution_x
    resolution_y = bpy.context.scene.render.resolution_y
    if resolution_x % 128 != 0 or resolution_y % 128 != 0:
      result = False
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
    
    screen_space_normal_image = bpy.data.images.new(
        name=RenderPasses.SCREEN_SPACE_NORMAL, width=width, height=height, float_buffer=True)
    
    current_frame = bpy.context.scene.frame_current
    current_frame = format(current_frame, '04d')
    screen_space_normal_image.pixels[:] = pixels
    
    samples_per_pixel = DeepDenoiserDataGenerator.current_samples_per_pixel()
    relative_frame_number = DeepDenoiserDataGenerator.current_relative_frame_number()
    seed = DeepDenoiserDataGenerator.current_seed()
    
    file_path = (
        target_folder + '/' + DeepDenoiserDataGenerator.blend_filename() + '/' +
        DeepDenoiserDataGenerator.blend_filename() + '_' + str(samples_per_pixel) + '_' + str(relative_frame_number) + '_' + str(seed) + '/' +
        DeepDenoiserDataGenerator.blend_filename() + '_' + str(samples_per_pixel) + '_' + str(relative_frame_number) + '_' + str(seed) + '_' +
        RenderPasses.SCREEN_SPACE_NORMAL + '_' + current_frame + '.exr')
    file_path = bpy.path.abspath(file_path)
    screen_space_normal_image.save_render(file_path)
  
  def render(target_folder, samples_per_pixel, samples_per_pixel_render_index, relative_frame_numbers):
    for relative_frame_number in relative_frame_numbers:
      seed = DeepDenoiserDataGenerator.seed(samples_per_pixel, samples_per_pixel_render_index, relative_frame_number)
      
      DeepDenoiserDataGenerator.prepare_passes()
      DeepDenoiserDataGenerator.prepare_image_settings()
      DeepDenoiserDataGenerator.prepare_relative_frame_number(relative_frame_number)
      DeepDenoiserDataGenerator.prepare_cycles(samples_per_pixel, seed)
      DeepDenoiserDataGenerator.prepare_compositor(target_folder)
      bpy.ops.render.render()
      DeepDenoiserDataGenerator.calculate_screen_space_normals(target_folder, samples_per_pixel)



# UI

main_frame_min = int(4)

seed_min = int(0)
seed_max = int(1e5)

class DeepDenoiserRenderJobPropertyGroup(bpy.types.PropertyGroup):
  use_render_job = bpy.props.BoolProperty(name='use_render_job')
  samples_per_pixel = bpy.props.IntProperty(name='samples_per_pixel', default=1, min=1)
  number_of_renders = bpy.props.IntProperty(name='number_of_renders', default=1, min=1)
  

class DeepDenoiserDataGeneratorPropertyGroup(bpy.types.PropertyGroup):
  target_folder = bpy.props.StringProperty(
      name='target_folder', description='Base directory for the rendered results', default='//OpenEXR/', maxlen=1024, subtype='DIR_PATH')
  main_frame = bpy.props.IntProperty(
      name='main_frame', description='Main frame to be rendered',
      default=main_frame_min, min=main_frame_min)
  main_frame_samples_per_pixel = bpy.props.IntProperty(
      name='main_frame_samples_per_pixel', description='Samples per pixel to render the main frame without noise',
      default=4096, min=1)
  seed = bpy.props.IntProperty(
      name='seed', description='Seed used as basis for the rendering, such that there is a strong variation with deterministic results',
      default=0, min=seed_min, max=seed_max)
  render_jobs_initialized = bpy.props.BoolProperty(
      name='render_jobs_initialized', description='Were the render jobs initialized. Only false when the script was never initialized',
      default=False)

class JumpToMainFrameOperator(bpy.types.Operator):
  bl_idname = "deep_blender.jump_to_main_frame_operator"
  bl_label = "Main frame"

  def execute(self, context):
    context.scene.frame_current = context.scene.deep_denoiser_generator_property_group.main_frame
    return {'FINISHED'}

class JumpToMainFrameMinusFourOperator(bpy.types.Operator):
  bl_idname = "deep_blender.jump_to_main_frame_minus_four_operator"
  bl_label = "-4"

  def execute(self, context):
    context.scene.frame_current = context.scene.deep_denoiser_generator_property_group.main_frame - 4
    return {'FINISHED'}
    
class JumpToMainFrameMinusThreeOperator(bpy.types.Operator):
  bl_idname = "deep_blender.jump_to_main_frame_minus_three_operator"
  bl_label = "-3"

  def execute(self, context):
    context.scene.frame_current = context.scene.deep_denoiser_generator_property_group.main_frame - 3
    return {'FINISHED'}

class JumpToMainFrameMinusTwoOperator(bpy.types.Operator):
  bl_idname = "deep_blender.jump_to_main_frame_minus_two_operator"
  bl_label = "-2"

  def execute(self, context):
    context.scene.frame_current = context.scene.deep_denoiser_generator_property_group.main_frame - 2
    return {'FINISHED'}

class JumpToMainFrameMinusOneOperator(bpy.types.Operator):
  bl_idname = "deep_blender.jump_to_main_frame_minus_one_operator"
  bl_label = "-1"

  def execute(self, context):
    context.scene.frame_current = context.scene.deep_denoiser_generator_property_group.main_frame - 1
    return {'FINISHED'}

class JumpToMainFramePlusOneOperator(bpy.types.Operator):
  bl_idname = "deep_blender.jump_to_main_frame_plus_one_operator"
  bl_label = "+1"

  def execute(self, context):
    context.scene.frame_current = context.scene.deep_denoiser_generator_property_group.main_frame + 1
    return {'FINISHED'}

class JumpToMainFramePlusTwoOperator(bpy.types.Operator):
  bl_idname = "deep_blender.jump_to_main_frame_plus_two_operator"
  bl_label = "+2"

  def execute(self, context):
    context.scene.frame_current = context.scene.deep_denoiser_generator_property_group.main_frame + 2
    return {'FINISHED'}

class JumpToMainFramePlusThreeOperator(bpy.types.Operator):
  bl_idname = "deep_blender.jump_to_main_frame_plus_three_operator"
  bl_label = "+3"

  def execute(self, context):
    context.scene.frame_current = context.scene.deep_denoiser_generator_property_group.main_frame + 3
    return {'FINISHED'}

class JumpToMainFramePlusFourOperator(bpy.types.Operator):
  # Add this dummy for symmetry reasons, even though it is not needed.
  # Very likely avoids plenty of reports, as it looks pretty off, without it :)
  bl_idname = "deep_blender.jump_to_main_frame_plus_four_operator"
  bl_label = "+4"

  def execute(self, context):
    context.scene.frame_current = context.scene.deep_denoiser_generator_property_group.main_frame + 4
    return {'FINISHED'}
   
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
  
class RENDER_JOB_OT_reset(bpy.types.Operator):
  bl_idname = 'deep_blender.render_jobs_reset'
  bl_label = "Reset the render jobs to the default settings"
  bl_description = "Reset the render jobs to the default settings"

  def execute(self, context):
    DeepDenoiserDataGenerator.reset_render_jobs()
    
    return{'FINISHED'}

class RENDER_JOB_prepare(bpy.types.Operator):
  bl_idname = 'deep_blender.prepare'
  bl_label = "Prepare Settings"
  bl_description = "Prepare all the Blender settings to experiment with the same settings that are used when rendering"

  def execute(self, context):
    render = bpy.context.scene.render
    target_folder = context.scene.deep_denoiser_generator_property_group.target_folder
    seed = context.scene.deep_denoiser_generator_property_group.seed
    
    # Just a dummy.
    samples_per_pixel = 1
    relative_frame_number = 0
    
    DeepDenoiserDataGenerator.prepare_passes()
    DeepDenoiserDataGenerator.prepare_image_settings()
    DeepDenoiserDataGenerator.prepare_relative_frame_number(relative_frame_number)
    DeepDenoiserDataGenerator.prepare_cycles(samples_per_pixel, seed)
    DeepDenoiserDataGenerator.prepare_compositor(target_folder)
    
    return{'FINISHED'}

class RENDER_JOB_render_main_frame_noisy(bpy.types.Operator):
  bl_idname = 'deep_blender.render_main_frame_noisy'
  bl_label = "Render main frame noisy"
  bl_description = "Render main frame noisy"

  def execute(self, context):
    render = bpy.context.scene.render
    target_folder = context.scene.deep_denoiser_generator_property_group.target_folder
    render_jobs = context.scene.render_jobs
    
    for render_job in render_jobs:
      if render_job.use_render_job:
        for i in range(render_job.number_of_renders):
          DeepDenoiserDataGenerator.render(
              target_folder, render_job.samples_per_pixel, i, [0])
    return{'FINISHED'}

class RENDER_JOB_render_main_frame_noiseless(bpy.types.Operator):
  bl_idname = 'deep_blender.render_main_frame_noiseless'
  bl_label = "Render main frame noiseless"
  bl_description = "Render main frame noiseless"

  def execute(self, context):
    render = bpy.context.scene.render
    target_folder = context.scene.deep_denoiser_generator_property_group.target_folder
    
    samples_per_pixel = context.scene.deep_denoiser_generator_property_group.main_frame_samples_per_pixel
    DeepDenoiserDataGenerator.render(target_folder, samples_per_pixel, 0, [0])
    return{'FINISHED'}

class RENDER_JOB_render_main_frame(bpy.types.Operator):
  bl_idname = 'deep_blender.render_main_frame'
  bl_label = "Render main frame noiseless and noisy"
  bl_description = "Render main frame noiseless and noisy"

  def execute(self, context):
    render = bpy.context.scene.render
    target_folder = context.scene.deep_denoiser_generator_property_group.target_folder
    render_jobs = context.scene.render_jobs
    
    # Noiseless
    samples_per_pixel = context.scene.deep_denoiser_generator_property_group.main_frame_samples_per_pixel
    DeepDenoiserDataGenerator.render(target_folder, samples_per_pixel, 0, [0])
    
    # Noisy
    for render_job in render_jobs:
      if render_job.use_render_job:
        for i in range(render_job.number_of_renders):
          DeepDenoiserDataGenerator.render(
              target_folder, render_job.samples_per_pixel, i, [0])
    return{'FINISHED'}

class RENDER_JOB_render_pre_post_noisy(bpy.types.Operator):
  bl_idname = 'deep_blender.render_pre_post_noisy'
  bl_label = "Render noisy frames before and after the main frame"
  bl_description = "Render noisy frames before and after the main frame"

  def execute(self, context):
    render = bpy.context.scene.render
    target_folder = context.scene.deep_denoiser_generator_property_group.target_folder
    render_jobs = context.scene.render_jobs
    
    for render_job in render_jobs:
      if render_job.use_render_job:
        for i in range(render_job.number_of_renders):
          pre_post_frame_numbers = [-3, -2, -1, 1, 2, 3]
          DeepDenoiserDataGenerator.render(
              target_folder, render_job.samples_per_pixel, i, pre_post_frame_numbers)
    return{'FINISHED'}

class RENDER_JOB_render_all(bpy.types.Operator):
  bl_idname = 'deep_blender.render_all'
  bl_label = "Render everything"
  bl_description = "Render everything"

  def execute(self, context):
    render = bpy.context.scene.render
    target_folder = context.scene.deep_denoiser_generator_property_group.target_folder
    render_jobs = context.scene.render_jobs
    
    # Noiseless
    samples_per_pixel = context.scene.deep_denoiser_generator_property_group.main_frame_samples_per_pixel
    DeepDenoiserDataGenerator.render(target_folder, samples_per_pixel, 0, [0])
    
    # Noisy
    for render_job in render_jobs:
      if render_job.use_render_job:
        for i in range(render_job.number_of_renders):
          relative_frame_numbers = [-3, -2, -1, 0, 1, 2, 3]
          DeepDenoiserDataGenerator.render(
              target_folder, render_job.samples_per_pixel, i, relative_frame_numbers)
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
    
    render = scene.render
    
    column = layout.column()
    column.operator('deep_blender.prepare')
    
    column = layout.column()
    column.prop(scene.deep_denoiser_generator_property_group, 'target_folder', text='Folder')
    
    
    column = layout.column(align=True)
    column.label(text="Resolution:")
    column.prop(render, "resolution_x", text="X")
    column.prop(render, "resolution_y", text="Y")
    
    is_resolution_valid = DeepDenoiserDataGenerator.is_resolution_valid()
    if not is_resolution_valid:
      box = layout.box()
      inner_column = box.column()
      inner_column.label(text="Resolution Error", icon='ERROR')
      inner_column.label(text="Both the x and y resolutions have to be multiples of 128!")
    
    column = layout.column()
    column.label(text="Frames:")
    column.prop(scene.deep_denoiser_generator_property_group, 'main_frame', text='Main frame')
    
    column = layout.column()
    column.label(text="Check Frames:")
    row = column.row(align=True)
    row.operator('deep_blender.jump_to_main_frame_minus_four_operator')
    row.operator('deep_blender.jump_to_main_frame_minus_three_operator')
    row.operator('deep_blender.jump_to_main_frame_minus_two_operator')
    row.operator('deep_blender.jump_to_main_frame_minus_one_operator')
    row.operator('deep_blender.jump_to_main_frame_operator')
    row.operator('deep_blender.jump_to_main_frame_plus_one_operator')
    row.operator('deep_blender.jump_to_main_frame_plus_two_operator')
    row.operator('deep_blender.jump_to_main_frame_plus_three_operator')
    row.operator('deep_blender.jump_to_main_frame_plus_four_operator')
    
    column = layout.column()
    column.prop(scene.deep_denoiser_generator_property_group, 'main_frame_samples_per_pixel', text='Samples for noiseless main frame')
    
    
    column.label(text="Noisy samples per pixel:")
    
    row = column.row()
    row.template_list('DeepDenoiserItemUI', 'compact', scene, 'render_jobs', scene, 'selected_render_job_index')
    column = row.column(align=True)
    column.operator('deep_blender.render_jobs_add', icon='ZOOMIN', text='')
    column.operator('deep_blender.render_jobs_remove', icon='ZOOMOUT', text='')
    
    if len(scene.render_jobs) > 1 and scene.selected_render_job_index >= 0:
      column.separator()
      column.operator('deep_blender.render_jobs_move_up', icon='TRIA_UP', text='')
      column.operator('deep_blender.render_jobs_move_down', icon='TRIA_DOWN', text='')

    column.separator()
    column.operator('deep_blender.render_jobs_reset', icon='FILE_REFRESH', text='')
    
    column = layout.column()
    if not is_resolution_valid:
      column.enabled = False
    column.label(text="Render:")
    if not is_resolution_valid:
      box = column.box()
      inner_column = box.column()
      inner_column.label(text="Resolution Error", icon='ERROR')
      inner_column.label(text="Rendering is only enabled, if all errors are resolved.")
    column.operator('deep_blender.render_main_frame_noiseless', text='Main frame noiseless', icon='RENDER_STILL')
    column.operator('deep_blender.render_main_frame_noisy', text='Main frame noisy', icon='RENDER_STILL')
    column.operator('deep_blender.render_main_frame', text='Main frame', icon='RENDER_STILL')
    
    # Not yet needed.
    column = layout.column()
    column.enabled = False
    column.label(text="Render for temporal denoising: (Not yet needed)")
    column.operator('deep_blender.render_pre_post_noisy', text='Pre- and post-frames noisy', icon='RENDER_STILL')
    column.operator('deep_blender.render_all', text='All', icon='RENDER_STILL')
    
    # Should not be needed.
    column = layout.column()
    column.enabled = False
    column.label(text="Advanced: (Only for testing)")
    row = column.row()
    row.prop(scene.deep_denoiser_generator_property_group, 'seed', text='Seed')
    row.operator('deep_blender.randomize_seed_operator', icon='FILE_REFRESH', text='')
    


classes = (
    DeepDenoiserRenderJobPropertyGroup, DeepDenoiserDataGeneratorPropertyGroup, DeepDenoiserDataGeneratorPanel,
    DeepDenoiserItemUI, RandomizeSeedOperator, JumpToMainFrameOperator,
    JumpToMainFrameMinusFourOperator, JumpToMainFrameMinusThreeOperator, JumpToMainFrameMinusTwoOperator, JumpToMainFrameMinusOneOperator,
    JumpToMainFramePlusOneOperator, JumpToMainFramePlusTwoOperator, JumpToMainFramePlusThreeOperator, JumpToMainFramePlusFourOperator,
    RENDER_JOB_OT_add, RENDER_JOB_OT_remove, RENDER_JOB_OT_move_up, RENDER_JOB_OT_move_down, RENDER_JOB_OT_reset, RENDER_JOB_prepare,
    RENDER_JOB_render_main_frame_noisy, RENDER_JOB_render_main_frame_noiseless, RENDER_JOB_render_main_frame,
    RENDER_JOB_render_pre_post_noisy, RENDER_JOB_render_all)

def register():
  for i in classes:
    bpy.utils.register_class(i)

  # Store properties in the scene
  bpy.types.Scene.deep_denoiser_generator_property_group = bpy.props.PointerProperty(type=DeepDenoiserDataGeneratorPropertyGroup)
  bpy.types.Scene.render_jobs = bpy.props.CollectionProperty(type=DeepDenoiserRenderJobPropertyGroup)
  bpy.types.Scene.selected_render_job_index = bpy.props.IntProperty(
      name='selected_render_job_index', description= "Selected render job index", default= -1, min= -1)

  # Initialize the render jobs the very first time the script gets executed.
  if not bpy.context.scene.deep_denoiser_generator_property_group.render_jobs_initialized:
    bpy.context.scene.deep_denoiser_generator_property_group.render_jobs_initialized = True
    DeepDenoiserDataGenerator.reset_render_jobs()
  
def unregister():
  for i in classes:
    bpy.utils.unregister_class(i)


if __name__ == "__main__":
  register()