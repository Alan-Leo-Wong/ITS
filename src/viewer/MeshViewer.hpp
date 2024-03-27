//
// Created by Lei on 2/9/2024.
//
#ifndef ITS_MESHVIEWER_HPP
#define ITS_MESHVIEWER_HPP

#include "Config.hpp"
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

NAMESPACE_BEGIN(ITS)
namespace viewer {
    using namespace igl::opengl::glfw;

    class MeshViewer : public Viewer {
    private:
        igl::opengl::glfw::imgui::ImGuiPlugin plugin;
        igl::opengl::glfw::imgui::ImGuiMenu menu;

    public:
        MeshViewer() {
            // attach a menu plugin
            this->plugins.emplace_back(&plugin);
            plugin.widgets.emplace_back(&menu);
        }
    };

}
NAMESPACE_END(ITS)

#endif // ITS_MESHVIEWER_HPP
