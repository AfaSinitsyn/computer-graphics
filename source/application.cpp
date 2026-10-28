// Lab 2, variant 3. Application code for vladeemerr/vulkan-starter-app.
#include "application.hpp"
#include "lab_scene.hpp"

#include <imgui.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace application {
namespace {

struct Buffer {
    VkBuffer handle=VK_NULL_HANDLE;
    VmaAllocation allocation=nullptr;
    void* mapped=nullptr;
    VkDeviceSize size=0;
};

struct Settings {
    lab::ObjectState object;
    lab::Camera camera;
    lab::DirectionalLight directional;
    std::vector<lab::PointLight> points;
    std::vector<lab::SpotLight> spots;
    float ambient_color[3]{1,1,1};
    float ambient_intensity=0.12f;
    int projection=0; // 0: perspective; 1: orthographic.
    bool playing=false;
    double animation_time=0;
    float animation_speed=1.0f;
};

Settings settings;
double previous_time=-1.0;
Buffer vertex_buffer, index_buffer;
Buffer scene_buffer, model_buffer, point_buffer, spot_buffer;
VkDescriptorSet descriptor_sets[2]{}; // 0: scene, 1: model
VkDescriptorSetLayout descriptor_layouts[2]{};
VkDeviceSize max_storage_range=0;
VkDescriptorPool descriptor_pool=VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout=VK_NULL_HANDLE;
VkPipeline pipeline=VK_NULL_HANDLE;
VkShaderModule vertex_shader=VK_NULL_HANDLE;
VkShaderModule fragment_shader=VK_NULL_HANDLE;

void checkVk(VkResult result, const char* operation) {
    if (result!=VK_SUCCESS)
        throw std::runtime_error(std::string(operation)+": VkResult="+std::to_string(result));
}

void writeBuffer(const Buffer& buffer, const void* data, VkDeviceSize size) {
    if (size>buffer.size || buffer.mapped==nullptr)
        throw std::runtime_error("Invalid write to a mapped Vulkan buffer");
    std::memcpy(buffer.mapped,data,static_cast<size_t>(size));
    checkVk(vmaFlushAllocation(graphics::internal::context.allocator,
                              buffer.allocation,0,size),"vmaFlushAllocation");
}

void createBuffer(Buffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage,
                  const void* initial_data=nullptr) {
    auto& context=graphics::internal::context;
    VkBufferCreateInfo info{};
    info.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size=size;
    info.usage=usage;
    info.sharingMode=VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocation{};
    allocation.usage=VMA_MEMORY_USAGE_AUTO;
    allocation.flags=VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo allocation_info{};
    checkVk(vmaCreateBuffer(context.allocator,&info,&allocation,
                           &buffer.handle,&buffer.allocation,&allocation_info),"vmaCreateBuffer");
    buffer.mapped=allocation_info.pMappedData;
    buffer.size=size;
    if (initial_data!=nullptr) writeBuffer(buffer,initial_data,size);
}

void destroyBuffer(Buffer& buffer) {
    if (buffer.handle!=VK_NULL_HANDLE || buffer.allocation!=nullptr)
        vmaDestroyBuffer(graphics::internal::context.allocator,buffer.handle,buffer.allocation);
    buffer=Buffer{};
}

VkShaderModule loadShader(const char* path) {
    std::ifstream file(path,std::ios::ate | std::ios::binary);
    if (!file)
        throw std::runtime_error(std::string("Cannot open ")+path+
            ". Build shaders and start the app from the project root.");
    const auto length=file.tellg();
    if (length<=0 || static_cast<size_t>(length)%sizeof(uint32_t)!=0)
        throw std::runtime_error(std::string("Invalid SPIR-V file: ")+path);
    std::vector<uint32_t> code(static_cast<size_t>(length)/sizeof(uint32_t));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(code.data()),static_cast<std::streamsize>(length)))
        throw std::runtime_error(std::string("Cannot read ")+path);
    VkShaderModuleCreateInfo info{};
    info.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize=code.size()*sizeof(uint32_t);
    info.pCode=code.data();
    VkShaderModule result=VK_NULL_HANDLE;
    checkVk(vkCreateShaderModule(graphics::internal::context.device,&info,nullptr,&result),
            "vkCreateShaderModule");
    return result;
}

void createGeometry() {
    const lab::Mesh mesh=lab::makeParallelepiped({2.0f,1.0f,1.5f});
    createBuffer(vertex_buffer,sizeof(mesh.vertices),VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 mesh.vertices.data());
    createBuffer(index_buffer,sizeof(mesh.indices),VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 mesh.indices.data());
}

void bindBuffer(VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                const Buffer& buffer) {
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer=buffer.handle;
    buffer_info.range=buffer.size;
    VkWriteDescriptorSet write{};
    write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet=set;
    write.dstBinding=binding;
    write.descriptorCount=1;
    write.descriptorType=type;
    write.pBufferInfo=&buffer_info;
    vkUpdateDescriptorSets(graphics::internal::context.device,1,&write,0,nullptr);
}

void createDescriptors() {
    auto& context=graphics::internal::context;
    VkDescriptorSetLayoutBinding scene_bindings[3]{};
    for (uint32_t i=0; i<3; ++i) {
        scene_bindings[i].binding=i;
        scene_bindings[i].descriptorCount=1;
        scene_bindings[i].descriptorType=i==0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                             : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        scene_bindings[i].stageFlags=i==0 ? VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
                                         : VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount=3;
    layout_info.pBindings=scene_bindings;
    checkVk(vkCreateDescriptorSetLayout(context.device,&layout_info,nullptr,&descriptor_layouts[0]),
            "vkCreateDescriptorSetLayout(scene)");

    VkDescriptorSetLayoutBinding model_binding{};
    model_binding.binding=0;
    model_binding.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    model_binding.descriptorCount=1;
    model_binding.stageFlags=VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    layout_info.bindingCount=1;
    layout_info.pBindings=&model_binding;
    checkVk(vkCreateDescriptorSetLayout(context.device,&layout_info,nullptr,&descriptor_layouts[1]),
            "vkCreateDescriptorSetLayout(model)");

    VkDescriptorPoolSize pool_sizes[2]{};
    pool_sizes[0]={VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,2};
    pool_sizes[1]={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2};
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets=2;
    pool_info.poolSizeCount=2;
    pool_info.pPoolSizes=pool_sizes;
    checkVk(vkCreateDescriptorPool(context.device,&pool_info,nullptr,&descriptor_pool),
            "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo allocate_info{};
    allocate_info.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate_info.descriptorPool=descriptor_pool;
    allocate_info.descriptorSetCount=2;
    allocate_info.pSetLayouts=descriptor_layouts;
    checkVk(vkAllocateDescriptorSets(context.device,&allocate_info,descriptor_sets),
            "vkAllocateDescriptorSets");

    createBuffer(scene_buffer,sizeof(lab::SceneUniforms),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    createBuffer(model_buffer,sizeof(lab::ModelUniforms),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    // Vulkan buffers cannot have zero size, even when the light array is empty.
    createBuffer(point_buffer,sizeof(lab::PointLightGPU),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    createBuffer(spot_buffer,sizeof(lab::SpotLightGPU),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bindBuffer(descriptor_sets[0],0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,scene_buffer);
    bindBuffer(descriptor_sets[0],1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,point_buffer);
    bindBuffer(descriptor_sets[0],2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,spot_buffer);
    bindBuffer(descriptor_sets[1],0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,model_buffer);
}

void updateStorageBuffer(Buffer& buffer, uint32_t binding, const void* data, VkDeviceSize size) {
    if (size==0) return;
    if (size>max_storage_range)
        throw std::runtime_error("Light array exceeds the GPU storage buffer limit");
    if (size>buffer.size) {
        // main.cpp already waited for the previous frame's fence before update().
        destroyBuffer(buffer);
        createBuffer(buffer,size,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        bindBuffer(descriptor_sets[0],binding,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,buffer);
    }
    writeBuffer(buffer,data,size);
}

void createPipeline() {
    auto& context=graphics::internal::context;
    vertex_shader=loadShader("shaders/parallelepiped.vert.spv");
    fragment_shader=loadShader("shaders/parallelepiped.frag.spv");
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module=vertex_shader;
    stages[0].pName="main";
    stages[1].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module=fragment_shader;
    stages[1].pName="main";

    VkVertexInputBindingDescription vertex_binding{};
    vertex_binding.binding=0;
    vertex_binding.stride=sizeof(lab::Vertex);
    vertex_binding.inputRate=VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attributes[3]{};
    attributes[0].location=0;
    attributes[0].binding=0;
    attributes[0].format=VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset=offsetof(lab::Vertex,position);
    attributes[1].location=1;
    attributes[1].binding=0;
    attributes[1].format=VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset=offsetof(lab::Vertex,color);

    attributes[2].location=2;
    attributes[2].binding=0;
    attributes[2].format=VK_FORMAT_R32G32B32_SFLOAT;
    attributes[2].offset=offsetof(lab::Vertex,normal);

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount=1;
    vertex_input.pVertexBindingDescriptions=&vertex_binding;
    vertex_input.vertexAttributeDescriptionCount=3;
    vertex_input.pVertexAttributeDescriptions=attributes;
    VkPipelineInputAssemblyStateCreateInfo assembly{};
    assembly.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount=1;
    viewport_state.scissorCount=1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode=VK_POLYGON_MODE_FILL;
    raster.cullMode=VK_CULL_MODE_BACK_BIT;
    raster.frontFace=VK_FRONT_FACE_CLOCKWISE;
    raster.lineWidth=1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable=VK_TRUE;
    depth.depthWriteEnable=VK_TRUE;
    depth.depthCompareOp=VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState color_attachment{};
    color_attachment.colorWriteMask=VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount=1;
    blend.pAttachments=&color_attachment;
    const VkDynamicState dynamic_states[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount=2;
    dynamic.pDynamicStates=dynamic_states;

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount=2;
    layout_info.pSetLayouts=descriptor_layouts;
    checkVk(vkCreatePipelineLayout(context.device,&layout_info,nullptr,&pipeline_layout),
            "vkCreatePipelineLayout");
    VkGraphicsPipelineCreateInfo info{};
    info.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount=2;
    info.pStages=stages;
    info.pVertexInputState=&vertex_input;
    info.pInputAssemblyState=&assembly;
    info.pViewportState=&viewport_state;
    info.pRasterizationState=&raster;
    info.pMultisampleState=&multisample;
    info.pDepthStencilState=&depth;
    info.pColorBlendState=&blend;
    info.pDynamicState=&dynamic;
    info.layout=pipeline_layout;
    info.renderPass=context.render_pass;
    info.subpass=0;
    info.basePipelineIndex=-1;
    checkVk(vkCreateGraphicsPipelines(context.device,VK_NULL_HANDLE,1,&info,nullptr,&pipeline),
            "vkCreateGraphicsPipelines");
    vkDestroyShaderModule(context.device,vertex_shader,nullptr);
    vkDestroyShaderModule(context.device,fragment_shader,nullptr);
    vertex_shader=fragment_shader=VK_NULL_HANDLE;
}

uint32_t panelWidth() {
    return std::min(410u,graphics::internal::context.swapchain_extent.width/3u);
}

VkRect2D sceneRectangle() {
    const auto extent=graphics::internal::context.swapchain_extent;
    const uint32_t left=std::min(panelWidth()+24u,extent.width-1u);
    return {{static_cast<int32_t>(left),0},{extent.width-left,extent.height}};
}

void showPointLights() {
    if (!ImGui::CollapsingHeader("Point lights",ImGuiTreeNodeFlags_DefaultOpen)) return;
    ImGui::PushID("point-lights");
    ImGui::BeginDisabled((settings.points.size()+1)*sizeof(lab::PointLightGPU)>max_storage_range);
    if (ImGui::Button("Add point light")) settings.points.emplace_back();
    ImGui::EndDisabled();
    for (size_t i=0; i<settings.points.size();) {
        ImGui::PushID(static_cast<int>(i));
        auto& light=settings.points[i];
        ImGui::Separator();
        ImGui::Text("Point %u",static_cast<unsigned>(i+1));
        if (ImGui::Button("Remove point")) {
            settings.points.erase(settings.points.begin()+i);
            ImGui::PopID();
            continue;
        }
        ImGui::DragFloat3("Position",light.position,0.05f);
        ImGui::ColorEdit3("Color",light.color);
        ImGui::SliderFloat("Intensity",&light.intensity,0.0f,50.0f);
        ImGui::PopID();
        ++i;
    }
    ImGui::PopID();
}

void showSpotLights() {
    if (!ImGui::CollapsingHeader("Spot lights",ImGuiTreeNodeFlags_DefaultOpen)) return;
    ImGui::PushID("spot-lights");
    ImGui::BeginDisabled((settings.spots.size()+1)*sizeof(lab::SpotLightGPU)>max_storage_range);
    if (ImGui::Button("Add spot light")) settings.spots.emplace_back();
    ImGui::EndDisabled();
    for (size_t i=0; i<settings.spots.size();) {
        ImGui::PushID(static_cast<int>(i));
        auto& light=settings.spots[i];
        ImGui::Separator();
        ImGui::Text("Spot %u",static_cast<unsigned>(i+1));
        if (ImGui::Button("Remove spot")) {
            settings.spots.erase(settings.spots.begin()+i);
            ImGui::PopID();
            continue;
        }
        ImGui::DragFloat3("Position",light.position,0.05f);
        ImGui::DragFloat3("Direction",light.direction,0.02f);
        ImGui::ColorEdit3("Color",light.color);
        ImGui::SliderFloat("Intensity",&light.intensity,0.0f,50.0f);
        ImGui::SliderFloat("Inner angle",&light.inner_angle,0.1f,89.0f,"%.1f deg",ImGuiSliderFlags_AlwaysClamp);
        light.outer_angle=std::max(light.outer_angle,light.inner_angle);
        ImGui::SliderFloat("Outer angle",&light.outer_angle,light.inner_angle,89.0f,"%.1f deg",ImGuiSliderFlags_AlwaysClamp);
        ImGui::PopID();
        ++i;
    }
    ImGui::PopID();
}

void showControls() {
    const auto extent=graphics::internal::context.swapchain_extent;
    ImGui::SetNextWindowPos(ImVec2(12.0f,12.0f),ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(static_cast<float>(panelWidth()),
        std::max(200.0f,static_cast<float>(extent.height)-24.0f)),ImGuiCond_Always);
    const ImGuiWindowFlags flags=ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoCollapse;
    if (ImGui::Begin("Computer Graphics | Lab 2",nullptr,flags)) {
        ImGui::TextUnformatted("Variant 3: Parallelepiped | Blinn-Phong");
        ImGui::TextWrapped("Scroll to see all controls. Directions follow the light rays.");
        if (ImGui::CollapsingHeader("Camera",ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushID("camera");
            ImGui::Combo("Projection",&settings.projection,"Perspective\0Orthographic\0");
            ImGui::DragFloat3("Position",settings.camera.position,0.03f);
            ImGui::DragFloat3("Rotation",settings.camera.rotation,0.5f,-180.0f,180.0f,"%.1f");
            ImGui::SliderFloat("FOV",&settings.camera.fov,15.0f,100.0f,"%.1f deg",ImGuiSliderFlags_AlwaysClamp);
            if (settings.projection==1)
                ImGui::SliderFloat("View height",&settings.camera.ortho_height,1.0f,15.0f,"%.2f",ImGuiSliderFlags_AlwaysClamp);
            if (ImGui::Button("Reset camera")) settings.camera=lab::Camera{};
            ImGui::PopID();
        }
        auto& object=settings.object;
        if (ImGui::CollapsingHeader("Object")) {
            ImGui::PushID("object");
            ImGui::DragFloat3("Position",object.position,0.02f,-6.0f,6.0f,"%.2f");
            ImGui::DragFloat3("Rotation",object.rotation,0.5f,-180.0f,180.0f,"%.1f");
            ImGui::DragFloat3("Scale",object.scale,0.01f,0.1f,3.0f,"%.2f",ImGuiSliderFlags_AlwaysClamp);
            if (ImGui::Button(settings.playing ? "Pause" : "Play")) settings.playing=!settings.playing;
            ImGui::SliderFloat("Speed",&settings.animation_speed,0.0f,3.0f,"%.2fx");
            ImGui::SliderFloat("Path radius",&object.path_radius,0.0f,2.5f);
            ImGui::SliderFloat("Path height",&object.path_height,0.0f,1.5f);
            ImGui::PopID();
        }
        if (ImGui::CollapsingHeader("Material",ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Albedo",object.tint);
            ImGui::ColorEdit3("Specular color",object.specular);
            ImGui::SliderFloat("Shininess",&object.shininess,1.0f,256.0f,"%.1f",ImGuiSliderFlags_AlwaysClamp);
            ImGui::Checkbox("Use vertex colors",&object.vertex_colors);
        }
        if (ImGui::CollapsingHeader("Ambient",ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Ambient color",settings.ambient_color);
            ImGui::SliderFloat("Ambient intensity",&settings.ambient_intensity,0.0f,1.0f);
        }
        if (ImGui::CollapsingHeader("Directional light",ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushID("directional");
            ImGui::DragFloat3("Direction",settings.directional.direction,0.02f);
            ImGui::ColorEdit3("Color",settings.directional.color);
            ImGui::SliderFloat("Intensity",&settings.directional.intensity,0.0f,3.0f);
            ImGui::PopID();
        }
        showPointLights();
        showSpotLights();
    }
    ImGui::End();
}

void copy3(const float* source, float* target) {
    std::copy_n(source,3,target);
}

void updateUniforms() {
    const VkRect2D scene=sceneRectangle();
    const float aspect=static_cast<float>(scene.extent.width)/static_cast<float>(scene.extent.height);
    const lab::Mat4 projection=settings.projection==0
        ? lab::perspective(lab::radians(settings.camera.fov),aspect,0.1f,100.0f)
        : lab::orthographic(settings.camera.ortho_height,aspect,0.1f,100.0f);

    lab::SceneUniforms scene_data{};
    scene_data.view_projection=projection*lab::viewMatrix(settings.camera);
    copy3(settings.camera.position,scene_data.camera_position);
    copy3(settings.ambient_color,scene_data.ambient);
    scene_data.ambient[3]=settings.ambient_intensity;
    const auto& directional=settings.directional;
    const lab::Vec3 dir=lab::normalize({directional.direction[0],directional.direction[1],directional.direction[2]});
    scene_data.light_direction[0]=dir.x;
    scene_data.light_direction[1]=dir.y;
    scene_data.light_direction[2]=dir.z;
    copy3(directional.color,scene_data.light_color);
    scene_data.light_color[3]=directional.intensity;
    scene_data.light_counts[0]=static_cast<uint32_t>(settings.points.size());
    scene_data.light_counts[1]=static_cast<uint32_t>(settings.spots.size());
    writeBuffer(scene_buffer,&scene_data,sizeof(scene_data));

    lab::ModelUniforms model_data{};
    model_data.model=lab::modelMatrix(settings.object,settings.animation_time);
    copy3(settings.object.tint,model_data.albedo);
    copy3(settings.object.specular,model_data.specular);
    model_data.parameters[0]=settings.object.shininess;
    model_data.parameters[1]=settings.object.vertex_colors ? 1.0f : 0.0f;
    writeBuffer(model_buffer,&model_data,sizeof(model_data));

    std::vector<lab::PointLightGPU> points(settings.points.size());
    for (size_t i=0; i<points.size(); ++i) {
        copy3(settings.points[i].position,points[i].position);
        copy3(settings.points[i].color,points[i].color);
        points[i].color[3]=settings.points[i].intensity;
    }
    updateStorageBuffer(point_buffer,1,points.data(),points.size()*sizeof(lab::PointLightGPU));
    std::vector<lab::SpotLightGPU> spots(settings.spots.size());
    for (size_t i=0; i<spots.size(); ++i) {
        const auto& light=settings.spots[i];
        copy3(light.position,spots[i].position);
        const lab::Vec3 direction=lab::normalize({light.direction[0],light.direction[1],light.direction[2]});
        spots[i].direction[0]=direction.x;
        spots[i].direction[1]=direction.y;
        spots[i].direction[2]=direction.z;
        copy3(light.color,spots[i].color);
        spots[i].color[3]=light.intensity;
        spots[i].cutoffs[0]=std::cos(lab::radians(light.inner_angle));
        spots[i].cutoffs[1]=std::cos(lab::radians(light.outer_angle));
    }
    updateStorageBuffer(spot_buffer,2,spots.data(),spots.size()*sizeof(lab::SpotLightGPU));
}

} // namespace

bool initialize() {
    settings=Settings{};
    previous_time=-1.0;
    try {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device,&properties);
        max_storage_range=properties.limits.maxStorageBufferRange;
        createGeometry();
        createDescriptors();
        createPipeline();
        ImGui::StyleColorsDark();
        updateUniforms();
        return true;
    } catch (const std::exception& error) {
        std::cerr << "Lab initialization failed: " << error.what() << '\n';
        shutdown();
        return false;
    }
}

void shutdown() {
    auto& context=graphics::internal::context;
    vkQueueWaitIdle(context.graphics_queue);
    vkDestroyPipeline(context.device,pipeline,nullptr);
    pipeline=VK_NULL_HANDLE;
    vkDestroyPipelineLayout(context.device,pipeline_layout,nullptr);
    pipeline_layout=VK_NULL_HANDLE;
    vkDestroyShaderModule(context.device,vertex_shader,nullptr);
    vkDestroyShaderModule(context.device,fragment_shader,nullptr);
    vertex_shader=fragment_shader=VK_NULL_HANDLE;
    vkDestroyDescriptorPool(context.device,descriptor_pool,nullptr);
    descriptor_pool=VK_NULL_HANDLE;
    for (int i=0; i<2; ++i) {
        descriptor_sets[i]=VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(context.device,descriptor_layouts[i],nullptr);
        descriptor_layouts[i]=VK_NULL_HANDLE;
    }
    destroyBuffer(spot_buffer);
    destroyBuffer(point_buffer);
    destroyBuffer(model_buffer);
    destroyBuffer(scene_buffer);
    destroyBuffer(index_buffer);
    destroyBuffer(vertex_buffer);
}

void update(double time) {
    const double delta=previous_time<0 ? 0.0 : std::clamp(time-previous_time,0.0,0.1);
    previous_time=time;
    showControls();
    if (settings.playing) settings.animation_time+=delta*settings.animation_speed;
    updateUniforms();
}

void render(const graphics::internal::FrameData& fd) {
    auto& context=graphics::internal::context;
    checkVk(vkResetCommandBuffer(fd.command_buffer,0),"vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin{};
    begin.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    checkVk(vkBeginCommandBuffer(fd.command_buffer,&begin),"vkBeginCommandBuffer");

    VkClearValue clears[2]{};
    clears[0].color={{0.055f,0.067f,0.090f,1.0f}};
    clears[1].depthStencil={1.0f,0};
    VkRenderPassBeginInfo pass{};
    pass.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    pass.renderPass=context.render_pass;
    pass.framebuffer=fd.framebuffer;
    pass.renderArea.extent=context.swapchain_extent;
    pass.clearValueCount=2;
    pass.pClearValues=clears;
    vkCmdBeginRenderPass(fd.command_buffer,&pass,VK_SUBPASS_CONTENTS_INLINE);

    const VkRect2D scene=sceneRectangle();
    VkViewport viewport{};
    viewport.x=static_cast<float>(scene.offset.x);
    viewport.width=static_cast<float>(scene.extent.width);
    viewport.height=static_cast<float>(scene.extent.height);
    viewport.maxDepth=1.0f;
    vkCmdSetViewport(fd.command_buffer,0,1,&viewport);
    vkCmdSetScissor(fd.command_buffer,0,1,&scene);
    vkCmdBindPipeline(fd.command_buffer,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);
    const VkDeviceSize offset=0;
    vkCmdBindVertexBuffers(fd.command_buffer,0,1,&vertex_buffer.handle,&offset);
    vkCmdBindIndexBuffer(fd.command_buffer,index_buffer.handle,0,VK_INDEX_TYPE_UINT16);

    vkCmdBindDescriptorSets(fd.command_buffer,VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout,0,2,descriptor_sets,0,nullptr);
    vkCmdDrawIndexed(fd.command_buffer,36,1,0,0,0);

    vkCmdEndRenderPass(fd.command_buffer);
    checkVk(vkEndCommandBuffer(fd.command_buffer),"vkEndCommandBuffer");
}

} // namespace application
