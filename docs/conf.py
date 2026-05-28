"""Sphinx configuration for RobotBlockset."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
DOCS_ROOT = Path(__file__).resolve().parent
API_INDEX = DOCS_ROOT / "api" / "index.rst"
API_GENERATED_DIR = DOCS_ROOT / "api" / "generated"
sys.path.insert(0, str(SRC_ROOT))

project = "robotblockset"
author = "Leon Zlajpah"
release = "1.2.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
autosummary_imported_members = False
autosummary_filename_map = {
    "robotblockset.rbf": "robotblockset.rbf",
}
autoclass_content = "both"
autodoc_class_signature = "mixed"
autosummary_context = {}
napoleon_include_init_with_doc = True
napoleon_custom_sections = [("Attributes", "params_style")]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
    "undoc-members": False,
}

autodoc_mock_imports = [
    "IPython",
    "actionlib",
    "actionlib_msgs",
    "action_msgs",
    "action_msgs.msg",
    "builtin_interfaces",
    "builtin_interfaces.msg",
    "compliant_controllers_msgs",
    "compliant_controllers_msgs.msg",
    "control_msgs",
    "control_msgs.action",
    "coppeliasim_zmqremoteapi_client",
    "controller_manager_msgs",
    "controller_manager_msgs.msg",
    "controller_manager_msgs.srv",
    "cv_bridge",
    "cv2",
    "easyhid",
    "franka_gripper",
    "franka_gripper.msg",
    "franka_msgs",
    "geometry_msgs",
    "genesis",
    "glfw",
    "lxml",
    "lxml.etree",
    "mediapy",
    "move_base_msgs",
    "mujoco",
    "mujoco.viewer",
    "open3d",
    "open3d.core",
    "ompl",
    "ompl.base",
    "ompl.geometric",
    "panda_py",
    "pal_navigation_msgs",
    "pygments",
    "pydantic",
    "pypylon",
    "pyrobotiqgripper",
    "pyrealsense2",
    "rclpy",
    "rclpy.duration",
    "rclpy.node",
    "rclpy.executors",
    "rclpy.qos",
    "rclpy.action",
    "rclpy.task",
    "rclpy.time",
    "robot_module_msgs",
    "robot_module_msgs.msg",
    "roscpp",
    "roscpp.srv",
    "rospy",
    "rtde_control",
    "rtde_receive",
    "rtde_io",
    "sensor_msgs",
    "sensor_msgs.msg",
    "sensor_msgs_py",
    "sensor_msgs_py.point_cloud2",
    "std_msgs",
    "std_msgs.msg",
    "std_srvs",
    "tf2_ros",
    "torch",
    "trajectory_msgs",
    "trajectory_msgs.msg",
    "typing_extensions",
    "ur_rtde",
    "yourdfpy",
    "dashboard_client",
    "pyzed",
    "pyzed.sl",
]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 6,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]


def _strip_doc_metadata(app, what, name, obj, options, lines):
    """Remove copyright and author lines from rendered docstrings."""
    filtered = []
    skip_blank = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Copyright"):
            skip_blank = True
            continue
        if stripped.startswith("Authors:"):
            skip_blank = True
            continue
        if skip_blank and not stripped:
            skip_blank = False
            continue
        skip_blank = False
        filtered.append(line)

    lines[:] = filtered


def _is_subpackage(qualified_name: str) -> bool:
    """Return whether a fully qualified module name refers to a package."""
    try:
        module = importlib.import_module(qualified_name)
    except Exception:
        return False
    return hasattr(module, "__path__")


autosummary_context["is_subpackage"] = _is_subpackage


def _iter_autosummary_sources() -> Iterable[str]:
    """Yield autosummary source files for recursive stub generation."""
    yield str(API_INDEX)
    if API_GENERATED_DIR.exists():
        for rst_path in sorted(API_GENERATED_DIR.glob("*.rst")):
            yield str(rst_path)


def _generate_api_stubs(app) -> None:
    """Generate autosummary stubs before Sphinx starts reading sources."""
    from sphinx.ext.autosummary.generate import generate_autosummary_docs

    if API_GENERATED_DIR.exists():
        for rst_path in API_GENERATED_DIR.glob("*.rst"):
            rst_path.unlink()
    else:
        API_GENERATED_DIR.mkdir(parents=True)

    seen_sources: set[str] = set()
    pending_sources = [str(API_INDEX)]

    while pending_sources:
        current_sources = [source for source in pending_sources if source not in seen_sources]
        if not current_sources:
            break

        generate_autosummary_docs(
            current_sources,
            output_dir=str(API_GENERATED_DIR),
            base_path=str(DOCS_ROOT),
            imported_members=autosummary_imported_members,
            app=app,
            overwrite=True,
            encoding="utf-8",
        )
        seen_sources.update(current_sources)
        pending_sources = list(_iter_autosummary_sources())


def setup(app):
    """Register Sphinx event handlers."""
    app.connect("builder-inited", _generate_api_stubs)
    app.connect("autodoc-process-docstring", _strip_doc_metadata)
