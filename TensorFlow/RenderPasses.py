from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class RenderPasses:
  COMBINED = 'Combined'
  COMBINED_DIFFUSE = 'Diffuse'
  COMBINED_GLOSSY = 'Glossy'
  COMBINED_SUBSURFACE = 'Subsurface'
  COMBINED_TRANSMISSION = 'Transmission'
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
  
  @staticmethod
  def number_of_channels(render_pass_name):
    result = 3
    if render_pass_name == RenderPasses.ALPHA or render_pass_name == RenderPasses.DEPTH:
      result = 1
    return result

  @staticmethod
  def is_combined_feature_render_pass(render_pass_name):
    result = (
        render_pass_name == 'Diffuse' or render_pass_name == 'Glossy' or
        render_pass_name == 'Subsurface' or render_pass_name == 'Transmission')
    return result
  
  @staticmethod
  def is_volume_render_pass(render_pass_name):
    return 'Volume' in render_pass_name

  @staticmethod
  def is_direct_or_indirect_render_pass(render_pass_name):
    return render_pass_name.endswith(' Direct') or render_pass_name.endswith(' Indirect')
  
  @staticmethod
  def is_color_render_pass(render_pass_name):
    return render_pass_name.endswith(' Color')
  
  @staticmethod
  def is_rgb_color_render_pass(render_pass_name):
    result = True
        
    if (
        render_pass_name == RenderPasses.ALPHA or
        render_pass_name == RenderPasses.DEPTH or
        render_pass_name == RenderPasses.MIST or
        render_pass_name == RenderPasses.NORMAL or
        render_pass_name == RenderPasses.SCREEN_SPACE_NORMAL or
        render_pass_name == RenderPasses.MOTION_VECTOR or
        render_pass_name == RenderPasses.OBJECT_ID or
        render_pass_name == RenderPasses.MATERIAL_ID or
        render_pass_name == RenderPasses.UV):
      result = False
    return result
  
  @staticmethod
  def direct_or_indirect_to_color_render_pass(render_pass_name):
    result = None
    if render_pass_name.endswith(' Direct'):
      result = render_pass_name.replace(' Direct', ' Color')
    elif render_pass_name.endswith(' Indirect'):
      result = render_pass_name.replace(' Inirect', ' Color')

    # TODO: Special cases, rather hacky:
    if result.startswith(RenderPasses.ALPHA):
      result = RenderPasses.ALPHA
    elif result.startswith(RenderPasses.EMISSION):
      result = RenderPasses.EMISSION
    elif result.startswith(RenderPasses.ENVIRONMENT):
      result = RenderPasses.ENVIRONMENT

    elif result.startswith(RenderPasses.AMBIENT_OCCLUSION):
      result = RenderPasses.AMBIENT_OCCLUSION
    elif result.startswith(RenderPasses.SHADOW):
      result = RenderPasses.SHADOW

    return result
  
  @staticmethod
  def combined_to_color_render_pass(render_pass_name):
    result = render_pass_name + ' Color'

    # TODO: Special cases, rather hacky:
    if result.startswith(RenderPasses.ALPHA):
      result = RenderPasses.ALPHA
    elif result.startswith(RenderPasses.EMISSION):
      result = RenderPasses.EMISSION
    elif result.startswith(RenderPasses.ENVIRONMENT):
      result = RenderPasses.ENVIRONMENT

    elif result.startswith(RenderPasses.AMBIENT_OCCLUSION):
      result = RenderPasses.AMBIENT_OCCLUSION
    elif result.startswith(RenderPasses.SHADOW):
      result = RenderPasses.SHADOW

    return result
  
  @staticmethod
  def combined_to_direct_render_pass(render_pass_name):
    return render_pass_name + ' Direct'
  
  @staticmethod
  def combined_to_indirect_render_pass(render_pass_name):
    return render_pass_name + ' Indirect'

class RenderPassesUsage:
  def __init__(
      self,
      use_alpha=False, use_depth=False, use_mist=False, use_normal=False, use_screen_space_normal=False,
      use_motion_vector=False, use_object_id=False, use_material_id=False, use_uv=False,
      use_shadow=False, use_ambient_occlusion=False, use_emission=False, use_environment=False,
      use_diffuse_color=False, use_diffuse_direct=False, use_diffuse_indirect=False,
      use_glossy_color=False, use_glossy_direct=False, use_glossy_indirect=False,
      use_transmission_color=False, use_transmission_direct=False, use_transmission_indirect=False,
      use_subsurface_color=False, use_subsurface_direct=False, use_subsurface_indirect=False,
      use_volume_direct=False, use_volume_indirect=False):
    self.use_alpha = use_alpha
    self.use_depth = use_depth
    self.use_mist = use_mist
    self.use_normal = use_normal
    self.use_screen_space_normal = use_screen_space_normal
    self.use_motion_vector = use_motion_vector
    self.use_object_id = use_object_id
    self.use_material_id = use_material_id
    self.use_uv = use_uv
    self.use_shadow = use_shadow
    self.use_ambient_occlusion = use_ambient_occlusion
    self.use_emission = use_emission
    self.use_environment = use_environment
    self.use_diffuse_color = use_diffuse_color
    self.use_diffuse_direct = use_diffuse_direct
    self.use_diffuse_indirect = use_diffuse_indirect
    self.use_glossy_color = use_glossy_color
    self.use_glossy_direct = use_glossy_direct
    self.use_glossy_indirect = use_glossy_indirect
    self.use_transmission_color = use_transmission_color
    self.use_transmission_direct = use_transmission_direct
    self.use_transmission_indirect = use_transmission_indirect
    self.use_subsurface_color = use_subsurface_color
    self.use_subsurface_direct = use_subsurface_direct
    self.use_subsurface_indirect = use_subsurface_indirect
    self.use_volume_direct = use_volume_direct
    self.use_volume_indirect = use_volume_indirect
  
  def render_passes(self):
    result = []
    if self.use_alpha:
      result.append(RenderPasses.ALPHA)
    if self.use_depth:
      result.append(RenderPasses.DEPTH)
    if self.use_mist:
      result.append(RenderPasses.MIST)
    if self.use_normal:
      result.append(RenderPasses.NORMAL)
    if self.use_screen_space_normal:
      result.append(RenderPasses.SCREEN_SPACE_NORMAL)
    if self.use_motion_vector:
      result.append(RenderPasses.MOTION_VECTOR)
    if self.use_object_id:
      result.append(RenderPasses.OBJECT_ID)
    if self.use_material_id:
      result.append(RenderPasses.MATERIAL_ID)
    if self.use_uv:
      result.append(RenderPasses.UV)
    if self.use_shadow:
      result.append(RenderPasses.SHADOW)
    if self.use_ambient_occlusion:
      result.append(RenderPasses.AMBIENT_OCCLUSION)
    if self.use_emission:
      result.append(RenderPasses.EMISSION)
    if self.use_environment:
      result.append(RenderPasses.ENVIRONMENT)
    if self.use_diffuse_color:
      result.append(RenderPasses.DIFFUSE_COLOR)
    if self.use_diffuse_direct:
      result.append(RenderPasses.DIFFUSE_DIRECT)
    if self.use_diffuse_indirect:
      result.append(RenderPasses.DIFFUSE_INDIRECT)
    if self.use_glossy_color:
      result.append(RenderPasses.GLOSSY_COLOR)
    if self.use_glossy_direct:
      result.append(RenderPasses.GLOSSY_DIRECT)
    if self.use_glossy_indirect:
      result.append(RenderPasses.GLOSSY_INDIRECT)
    if self.use_transmission_color:
      result.append(RenderPasses.TRANSMISSION_COLOR)
    if self.use_transmission_direct:
      result.append(RenderPasses.TRANSMISSION_DIRECT)
    if self.use_transmission_indirect:
      result.append(RenderPasses.TRANSMISSION_INDIRECT)
    if self.use_subsurface_color:
      result.append(RenderPasses.SUBSURFACE_COLOR)
    if self.use_subsurface_direct:
      result.append(RenderPasses.SUBSURFACE_DIRECT)
    if self.use_subsurface_indirect:
      result.append(RenderPasses.SUBSURFACE_INDIRECT)
    if self.use_volume_direct:
      result.append(RenderPasses.VOLUME_DIRECT)
    if self.use_volume_indirect:
      result.append(RenderPasses.VOLUME_INDIRECT)
    return result
