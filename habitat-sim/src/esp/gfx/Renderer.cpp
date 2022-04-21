// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "Renderer.h"

#include <Corrade/Containers/StridedArrayView.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/BufferImage.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Platform/GLContext.h>
#include <Magnum/ResourceManager.h>

#include "esp/core/Check.h"
#include "esp/gfx/DepthUnprojection.h"
#include "esp/gfx/RenderTarget.h"
#include "esp/gfx/TextureVisualizerShader.h"
#include "esp/gfx/magnum.h"
#include "esp/sensor/VisualSensor.h"
#include "esp/sim/Simulator.h"

#ifdef ESP_BUILD_WITH_BACKGROUND_RENDERER
#include "BackgroundRenderer.h"
#endif

// There is a depth buffer overridden even when the depth test and depth buffer
// writing is diabled. It was observed only on Mac OSX, not on linux. Suspect it
// is a bug in the GL driver on Mac.
#ifdef CORRADE_TARGET_APPLE
#define ENABLE_VISUALIZATION_WORKAROUND_ON_MAC
#endif

namespace Mn = Magnum;

namespace esp {
namespace gfx {

void Renderer::setupMagnumFeatures() {
  Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::DepthTest);
  Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::FaceCulling);
}

struct Renderer::Impl {
  explicit Impl(WindowlessContext* context, Flags flags)
      : context_{context},
        depthShader_{nullptr},
        flags_{flags},
        mesh_{Cr::Containers::NullOpt} {
    setupMagnumFeatures();

#ifdef ESP_BUILD_WITH_BACKGROUND_RENDERER
    if (flags & Flag::BackgroundRenderer) {
      CORRADE_INTERNAL_ASSERT(context_ != nullptr);
      backgroundRenderer_ = std::make_unique<BackgroundRenderer>(context_);
    }
#endif
  }

  ~Impl() {
    acquireGlContext();
    LOG(INFO) << "Deconstructing Renderer";
  }

  void draw(RenderCamera& camera,
            scene::SceneGraph& sceneGraph,
            RenderCamera::Flags flags) {
    acquireGlContext();
    for (auto& it : sceneGraph.getDrawableGroups()) {
      // TODO: remove || true and NOLINT below
      // NOLINTNEXTLINE (readability-simplify-boolean-expr)
      if (it.second.prepareForDraw(camera) || true) {
        camera.draw(it.second, flags);
      }
    }
  }

  void draw(sensor::VisualSensor& visualSensor, sim::Simulator& sim) {
    acquireGlContext();
    if (visualSensor.specification()->sensorType ==
        sensor::SensorType::Semantic) {
      ESP_CHECK(sim.semanticSceneExists(),
                "Renderer::Impl::draw(): SemanticSensor observation requested "
                "but no SemanticScene is loaded");
    }
    visualSensor.drawObservation(sim);
  }

  void visualize(sensor::VisualSensor& visualSensor,
                 float colorMapOffset,
                 float colorMapScale) {
    acquireGlContext();
    sensor::SensorType& type = visualSensor.specification()->sensorType;
    if (type == sensor::SensorType::Depth ||
        type == sensor::SensorType::Semantic) {
      Mn::GL::Renderer::disable(Mn::GL::Renderer::Feature::DepthTest);
      gfx::RenderTarget& tgt = visualSensor.renderTarget();
      if (!mesh_) {
        // prepare a big triangle mesh to cover the screen
        mesh_ = Mn::GL::Mesh{};
        mesh_->setCount(3);
      }
      esp::gfx::Renderer::Impl::RendererShaderType rendererShaderType =
          esp::gfx::Renderer::Impl::RendererShaderType::DepthTextureVisualizer;

      if (type == sensor::SensorType::Semantic) {
        rendererShaderType =
            gfx::Renderer::Impl::RendererShaderType::ObjectIdTextureVisualizer;
      }

      Magnum::Resource<Mn::GL::AbstractShaderProgram, TextureVisualizerShader>
          shader = getShader<TextureVisualizerShader>(rendererShaderType);

      if (type == sensor::SensorType::Depth) {
#ifdef ENABLE_VISUALIZATION_WORKAROUND_ON_MAC
        // create a BufferImage instance, if not already
        if (!depthBufferImage_) {
          depthBufferImage_.emplace(Mn::GL::PixelFormat::DepthComponent,
                                    Mn::GL::PixelType::Float);
        }
        tgt.getDepthTexture().image(0, *depthBufferImage_,
                                    Mn::GL::BufferUsage::StaticRead);

        // This takes the above output image (which is depth) and
        // "reinterprets" it as R32F. In other words, the image below serves
        // as an "image view".
        Mn::GL::BufferImage2D clonedDepthImage{
            depthBufferImage_->storage(), Mn::PixelFormat::R32F,
            depthBufferImage_->size(),
            Mn::GL::Buffer::wrap(depthBufferImage_->buffer().id(),
                                 Mn::GL::ObjectFlag::Created),
            depthBufferImage_->dataSize()};

        // setup a texture
        if (!visualizedTex_ ||
            visualizedTex_->imageSize(0) != tgt.framebufferSize()) {
          visualizedTex_ = Mn::GL::Texture2D{};
          (*visualizedTex_)
              .setMinificationFilter(Mn::GL::SamplerFilter::Nearest)
              .setMagnificationFilter(Mn::GL::SamplerFilter::Nearest)
              .setWrapping(Mn::GL::SamplerWrapping::ClampToEdge)
              .setStorage(1, Mn::GL::TextureFormat::R32F,
                          tgt.framebufferSize());
        }
        (*visualizedTex_).setSubImage(0, {}, clonedDepthImage);
        shader->bindDepthTexture(*visualizedTex_);
#else
        shader->bindDepthTexture(tgt.getDepthTexture());
#endif
        shader->setDepthUnprojection(*visualSensor.depthUnprojection());
      } else if (type == sensor::SensorType::Semantic) {
        shader->bindObjectIdTexture(tgt.getObjectIdTexture());
      }
      shader->setColorMapTransformation(colorMapOffset, colorMapScale);
      tgt.renderReEnter();
      shader->draw(*mesh_);
      tgt.renderExit();

      // TODO object id
      Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::DepthTest);
    }
  }

#ifdef ESP_BUILD_WITH_BACKGROUND_RENDERER
  void checkHasBackgroundRenderer() {
    ESP_CHECK(backgroundRenderer_,
              "Renderer was not created with a background render "
              "thread, cannot do async drawing");
  }

  void enqueueAsyncDrawJob(sensor::VisualSensor& visualSensor,
                           scene::SceneGraph& sceneGraph,
                           const Mn::MutableImageView2D& view,
                           RenderCamera::Flags flags) {
    checkHasBackgroundRenderer();

    backgroundRenderer_->submitRenderJob(visualSensor, sceneGraph, view, flags);
  }

  void startDrawJobs() {
    checkHasBackgroundRenderer();
    if (contextIsOwned_) {
      context_->release();
      contextIsOwned_ = false;
    }

    backgroundRenderer_->startRenderJobs();
  }

  void waitDrawJobs() {
    checkHasBackgroundRenderer();
    backgroundRenderer_->waitThreadJobs();
    if (!(flags_ & Renderer::Flag::LeaveContextWithBackgroundRenderer))
      acquireGlContext();
  }

  void waitSceneGraph() {
    if (backgroundRenderer_)
      backgroundRenderer_->waitSceneGraph();
  }

  void acquireGlContext() {
    if (!contextIsOwned_) {
      VLOG(1) << "Renderer:: Main thread acquired GL Context";
      backgroundRenderer_->releaseContext();
      context_->makeCurrent();
      contextIsOwned_ = true;
    }
  }

  bool wasBackgroundRendererInitialized() const {
    return backgroundRenderer_ && backgroundRenderer_->wasInitialized();
  }

#else
  void acquireGlContext() {}
  void waitSceneGraph() {}
  bool wasBackgroundRendererInitialized() const { return false; }
#endif

  void bindRenderTarget(sensor::VisualSensor& sensor, Flags bindingFlags) {
    acquireGlContext();
    auto depthUnprojection = sensor.depthUnprojection();
    CORRADE_ASSERT(depthUnprojection,
                   "Renderer::Impl::bindRenderTarget(): Sensor does not have a "
                   "depthUnprojection matrix", );

    if (!depthShader_) {
      depthShader_ = std::make_unique<DepthShader>(
          DepthShader::Flag::UnprojectExistingDepth);
    }

    RenderTarget::Flags renderTargetFlags = {};
    switch (sensor.specification()->sensorType) {
      case sensor::SensorType::Color:
        CORRADE_ASSERT(
            !(flags_ & Flag::NoTextures),
            "Renderer::Impl::bindRenderTarget(): Tried to setup a color "
            "render buffer while the simulator was initialized with "
            "requiresTextures = false", );
        renderTargetFlags |= RenderTarget::Flag::RgbaAttachment;
        break;

      case sensor::SensorType::Depth:
        if (bindingFlags & Flag::VisualizeTexture) {
          renderTargetFlags |= RenderTarget::Flag::RgbaAttachment;
        }
        renderTargetFlags |= RenderTarget::Flag::DepthTextureAttachment;
        break;

      case sensor::SensorType::Semantic:
        if (bindingFlags & Flag::VisualizeTexture) {
          renderTargetFlags |= RenderTarget::Flag::RgbaAttachment;
        }
        renderTargetFlags |= RenderTarget::Flag::ObjectIdAttachment;
        break;

      default:
        // I need this default, since sensor type list is long, and without
        // default clang-tidy will complain
        CORRADE_INTERNAL_ASSERT_UNREACHABLE();
        break;
    }

    sensor.bindRenderTarget(RenderTarget::create_unique(
        sensor.framebufferSize(), *depthUnprojection, depthShader_.get(),
        renderTargetFlags, &sensor));
  }

 private:
  WindowlessContext* context_;
  bool contextIsOwned_ = true;
  // TODO: shall we use shader resource manager from now?
  std::unique_ptr<DepthShader> depthShader_;
  const Flags flags_;
#ifdef ESP_BUILD_WITH_BACKGROUND_RENDERER
  std::unique_ptr<BackgroundRenderer> backgroundRenderer_ = nullptr;
#endif
  Cr::Containers::Optional<Mn::GL::Mesh> mesh_;
  Mn::ResourceManager<Mn::GL::AbstractShaderProgram> shaderManager_;
#ifdef ENABLE_VISUALIZATION_WORKAROUND_ON_MAC
  Cr::Containers::Optional<Mn::GL::Texture2D> visualizedTex_;
  Cr::Containers::Optional<Mn::GL::BufferImage2D> depthBufferImage_;
#endif

  enum class RendererShaderType : uint8_t {
    DepthShader = 0,
    DepthTextureVisualizer = 1,
    ObjectIdTextureVisualizer = 2,
  };
  template <typename T>
  Mn::Resource<Mn::GL::AbstractShaderProgram, T> getShader(
      RendererShaderType type) {
    Mn::ResourceKey key;
    switch (type) {
      case RendererShaderType::DepthShader:
        key = Mn::ResourceKey{"depthShader"};
        break;

      case RendererShaderType::DepthTextureVisualizer:
        key = Mn::ResourceKey{"depthVisualizer"};
        break;

      case RendererShaderType::ObjectIdTextureVisualizer:
        key = Mn::ResourceKey{"objectIdVisualizer"};
        break;

      default:
        CORRADE_INTERNAL_ASSERT_UNREACHABLE();
        break;
    }
    Mn::Resource<Mn::GL::AbstractShaderProgram, T> shader =
        shaderManager_.get<Mn::GL::AbstractShaderProgram, T>(key);

    if (!shader) {
      if (type == RendererShaderType::DepthShader) {
        shaderManager_.set<Mn::GL::AbstractShaderProgram>(
            shader.key(),
            new DepthShader{DepthShader::Flag::UnprojectExistingDepth},
            Mn::ResourceDataState::Final, Mn::ResourcePolicy::ReferenceCounted);
      } else if (type == RendererShaderType::DepthTextureVisualizer) {
        shaderManager_.set<Mn::GL::AbstractShaderProgram>(
            shader.key(),
            new TextureVisualizerShader{
                {TextureVisualizerShader::Flag::DepthTexture}},
            Mn::ResourceDataState::Final, Mn::ResourcePolicy::ReferenceCounted);
      } else if (type == RendererShaderType::ObjectIdTextureVisualizer) {
        shaderManager_.set<Mn::GL::AbstractShaderProgram>(
            shader.key(),
            new TextureVisualizerShader{
                {TextureVisualizerShader::Flag::ObjectIdTexture}},
            Mn::ResourceDataState::Final, Mn::ResourcePolicy::ReferenceCounted);
      }
    }
    CORRADE_INTERNAL_ASSERT(shader);

    return shader;
  }
};

Renderer::Renderer(Flags flags) : Renderer{nullptr, flags} {}

Renderer::Renderer(WindowlessContext* context, Flags flags)
    : pimpl_(spimpl::make_unique_impl<Impl>(context, flags)) {}

void Renderer::draw(RenderCamera& camera,
                    scene::SceneGraph& sceneGraph,
                    RenderCamera::Flags flags) {
  pimpl_->draw(camera, sceneGraph, flags);
}

void Renderer::draw(sensor::VisualSensor& visualSensor, sim::Simulator& sim) {
  pimpl_->draw(visualSensor, sim);
}

void Renderer::bindRenderTarget(sensor::VisualSensor& sensor,
                                Flags bindingFlags) {
  pimpl_->bindRenderTarget(sensor, bindingFlags);
}

#ifdef ESP_BUILD_WITH_BACKGROUND_RENDERER
void Renderer::enqueueAsyncDrawJob(sensor::VisualSensor& visualSensor,
                                   scene::SceneGraph& sceneGraph,
                                   const Mn::MutableImageView2D& view,
                                   RenderCamera::Flags flags) {
  pimpl_->enqueueAsyncDrawJob(visualSensor, sceneGraph, view, flags);
}

void Renderer::waitDrawJobs() {
  pimpl_->waitDrawJobs();
}

void Renderer::startDrawJobs() {
  pimpl_->startDrawJobs();
}
#endif  // ESP_BUILD_WITH_BACKGROUND_RENDERER

void Renderer::acquireGlContext() {
  pimpl_->acquireGlContext();
}

void Renderer::waitSceneGraph() {
  pimpl_->waitSceneGraph();
}

bool Renderer::wasBackgroundRendererInitialized() const {
  return pimpl_->wasBackgroundRendererInitialized();
}

void Renderer::visualize(sensor::VisualSensor& sensor,
                         float colorMapOffset,
                         float colorMapScale) {
  pimpl_->visualize(sensor, colorMapOffset, colorMapScale);
}

}  // namespace gfx
}  // namespace esp
