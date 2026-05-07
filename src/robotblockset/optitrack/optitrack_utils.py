"""OptiTrack Utilities Module.

This module provides utility functions to work with OptiTrack Motive rigid-body XML files.
It includes functions to read and write marker locations for rigid bodies from/to XML files,
preserving comments and formatting as far as supported by the standard-library
XML parser. The module uses ``xml.etree.ElementTree`` to parse and modify the
XML files while providing functions to handle marker locations and related
properties.

Copyright (c) 2024- Jozef Stefan Institute

Authors: Leon Zlajpah.
"""

import numpy as np
import xml.etree.ElementTree as ET

from typing import Tuple

from robotblockset.rbs_typing import ArrayLike


def get_rigid_body_markers(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read marker locations from a Motive rigid-body XML file.

    Parameters
    ----------
    filename : str
        Path to the Motive rigid-body XML file.

    Returns
    -------
    tuple
        A tuple containing three NumPy arrays:
        - M (n, 3): Marker positions (n markers, 3 coordinates each)
        - Mv (n, 3): Marker location values
        - Mdv (n, 3): Default marker location values
    """

    # Parse the XML file while preserving comments where supported.
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(filename, parser=parser)
    root = tree.getroot()

    # Find all markers
    markers = root.find("./NodeAssets/rigid_body/markers")
    marker_list = markers.findall("marker")

    n = len(marker_list)
    M = np.full((n, 3), np.nan)
    Mv = np.full((n, 3), np.nan)
    Mdv = np.full((n, 3), np.nan)

    for marker in marker_list:
        position = np.array(list(map(float, marker.find("position").text.split(","))))
        idx = int(marker.find("label_id").text) - 1  # Convert 1-based MATLAB index to 0-based Python index
        M[idx, :] = position

    # Extract properties
    properties = root.find("./NodeAssets/rigid_body/properties")
    j = 0
    for prop in properties.findall("property"):
        name = prop.find("name").text
        if "MarkerLocation" in name:
            Mv[j, :] = np.array(list(map(float, prop.find("value").text.split(","))))
            Mdv[j, :] = np.array(list(map(float, prop.find("defaultValue").text.split(","))))
            j += 1

    return M, Mv, Mdv


def set_rigid_body_markers(filename: str, M: ArrayLike) -> None:
    """
    Write new marker locations to a Motive rigid-body XML file.

    Parameters
    ----------
    filename : str
        Path to the Motive rigid-body XML file.
    M : ArrayLike
        to the XML file (n markers, 3 coordinates each).

    Returns
    -------
    None
    """

    # Parse the XML file while preserving comments where supported.
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(filename, parser=parser)
    root = tree.getroot()

    # Find all markers
    markers = root.find("./NodeAssets/rigid_body/markers")
    marker_list = markers.findall("marker")

    n = len(marker_list)
    assert M.shape[0] == n, "Incorrect number of markers"

    for i, marker in enumerate(marker_list):
        marker.find("label_id").text = str(i + 1)  # Adjust MATLAB's 1-based indexing
        marker.find("position").text = f"{M[i, 0]:.8f},{M[i, 1]:.8f},{M[i, 2]:.8f}"

    # Update properties
    properties = root.find("./NodeAssets/rigid_body/properties")
    j = 0
    for prop in properties.findall("property"):
        name = prop.find("name").text
        if "MarkerLocation" in name:
            prop.find("value").text = f"{M[j, 0]:.8f},{M[j, 1]:.8f},{M[j, 2]:.8f}"
            prop.find("defaultValue").text = f"{M[j, 0]:.8f},{M[j, 1]:.8f},{M[j, 2]:.8f}"
            j += 1

    # Re-indent the tree before saving to keep the output readable.
    ET.indent(tree, space="  ")
    tree.write(filename, encoding="utf-8", xml_declaration=True)
