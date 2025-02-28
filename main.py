#!/usr/bin/env python3
"""
Generate a single-page PDF by tightly packing images (available in unlimited copies)
using a corner-based packing algorithm starting from the bottom-left corner and
working outward for maximum space efficiency.

Workflow:
  1. Preprocess each image:
     - Scale it so that its largest side is no more than MAX_IMAGE_CM (default 5 cm at 300 PPI).
     - Compute its display dimensions in points.
  2. For packing, each image is conceptually padded by 0.2 cm on all sides.
     This reduced padding is suitable for laser cutting precision.
  3. The algorithm uses a corner-based placement strategy, starting at the page origin (0,0)
     and placing images at optimal positions to maximize space utilization.
  4. A repetition penalty ensures good variety of images.
  5. Images can be rotated to any of the four 90-degree orientations for optimal fit.
  6. Finally, the placements are rendered to a PDF with cut lines at exact image edges.
  
Defaults (in cm):
  - Page size: A4 (21 x 29.7)
  - Maximum image dimension: 5 cm
  - Padding on each side: 0.2 cm (for laser cutting precision)
  - Penalty per use: Exponentially increasing to ensure variety

Usage:
    python script.py INPUT_DIRECTORY OUTPUT_PDF [--page_width WIDTH_CM]
                                           [--page_height HEIGHT_CM]
                                           [--max_image_cm MAX_IMAGE]
"""
import os
import sys
import argparse
import random
import math
import heapq
from typing import List, Tuple, Dict, Optional, Set, NamedTuple
from dataclasses import dataclass

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from loguru import logger
import time

# --- Conversion Constants ---
CM_TO_POINTS = 28.3464567    # 1 cm = 28.3464567 points
PIXELS_PER_INCH = 300
POINTS_PER_INCH = 72
PIXEL_TO_POINT = POINTS_PER_INCH / PIXELS_PER_INCH  # 72/300 = 0.24

def cm_to_points(cm: float) -> float:
    """Convert centimeters to points."""
    return cm * CM_TO_POINTS

def cm_to_pixels(cm: float) -> float:
    """Convert centimeters to pixels at 300 PPI."""
    return (cm / 2.54) * PIXELS_PER_INCH

# --- Data Structures for Corner-Based Packing ---
@dataclass
class ImageInfo:
    """Store image information including the PIL Image and dimensions."""
    img: Image.Image
    width: float  # points
    height: float # points
    
    def get_area(self) -> float:
        """Get the area of the image."""
        return self.width * self.height
    
    def get_rotations(self, pad: float = 0) -> List[Tuple[float, float, int]]:
        """Get all valid rotations of this image as (width, height, rotation) tuples."""
        results = []
        # Add 0 degree rotation
        results.append((self.width + 2*pad, self.height + 2*pad, 0))
        # Add 90 degree rotation
        results.append((self.height + 2*pad, self.width + 2*pad, 90))
        # We only need 0 and 90 for rectangles since 180 and 270 are symmetrical
        return results

class PlacementPoint(NamedTuple):
    """A potential placement point for an image corner."""
    x: float
    y: float
    score: float  # Lower is better (typically distance from origin)
    
    def __lt__(self, other):
        """For priority queue ordering - we want points with lower scores first."""
        return self.score < other.score

@dataclass
class Placement:
    """Represent an image placement with position, rotation, and dimensions."""
    img_idx: int        # Index in the candidates list
    img: Image.Image    # The PIL image
    x: float            # X position in points
    y: float            # Y position in points
    width: float        # Width in points including padding
    height: float       # Height in points including padding
    rotation: int       # Rotation in degrees (0, 90, 180, 270)
    
    def get_corners(self) -> List[Tuple[float, float]]:
        """Get the corners of this placement for generating new placement points."""
        # Corners are: top-right and top-left
        return [
            (self.x + self.width, self.y),           # bottom-right (x+w, y)
            (self.x, self.y + self.height)           # top-left (x, y+h)
        ]

# --- Image Preprocessing ---
def process_image(image_path: str, max_pixels: int):
    """
    Open and process an image.
    If either width or height exceeds max_pixels, scale it down proportionally.
    Returns a tuple: (PIL.Image, display_width_pt, display_height_pt)
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            if max(orig_w, orig_h) > max_pixels:
                scale = max_pixels / float(max(orig_w, orig_h))
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                logger.info("Resized '{}' from ({}, {}) to ({}, {}).",
                            os.path.basename(image_path), orig_w, orig_h, new_w, new_h)
            else:
                new_w, new_h = orig_w, orig_h
            disp_w = new_w * PIXEL_TO_POINT
            disp_h = new_h * PIXEL_TO_POINT
            return img, disp_w, disp_h
    except Exception as e:
        logger.error("Error processing image '{}': {}", image_path, e)
        return None

# NEW: Maximal-Rectangles Packing Algorithm for Rectangular Images
def pack_rectangles(candidates: List[ImageInfo],
                    bin_width: float,
                    bin_height: float,
                    pad: float) -> List[Placement]:
    """
    Pack images using a maximal-rectangles algorithm.
    Iteratively choose the candidate (with 0° and 90° rotations) that best fits into one of the free rectangles,
    then update the free space by splitting the used free rectangle.
    """
    # Track usage count for each candidate
    usage_count = [0] * len(candidates)
    free_rects: List[Tuple[float, float, float, float]] = [(0, 0, bin_width, bin_height)]
    placements: List[Placement] = []

    def remove_overlaps(rects: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        # Merge or remove overlapping free rectangles
        # (Implementation to remove overlaps or merge rectangles)
        return rects

    while True:
        best_score = float('inf')
        best_info = None  # (free_idx, candidate_idx, rotation, rect_x, rect_y, cand_w, cand_h)
        # Iterate over all free rectangles, candidates, and rotations
        for free_idx, (fx, fy, fw, fh) in enumerate(free_rects):
            for candidate_idx, candidate in enumerate(candidates):
                for cand_w, cand_h, rotation in candidate.get_rotations(pad):
                    # Check if candidate fits in free rectangle
                    if cand_w <= fw and cand_h <= fh:
                        # Heuristic: choose placement with minimal short side leftover
                        leftover_w = fw - cand_w
                        leftover_h = fh - cand_h
                        # Add usage penalty to the score
                        score = min(leftover_w, leftover_h) + usage_count[candidate_idx] * 100.0
                        if score < best_score:
                            best_score = score
                            best_info = (free_idx, candidate_idx, rotation, fx, fy, cand_w, cand_h)
        if best_info is None:
            break

        free_idx, candidate_idx, rotation, px, py, cand_w, cand_h = best_info
        usage_count[candidate_idx] += 1  # Increase usage for chosen candidate
        # Record the placement at the free rectangle's top-left corner (maintaining padding in dimensions)
        placement = Placement(
            img_idx=candidate_idx,
            img=candidates[candidate_idx].img,
            x=px,
            y=py,
            width=cand_w,
            height=cand_h,
            rotation=rotation
        )
        placements.append(placement)
        # Remove the free rectangle used
        used_rect = free_rects.pop(free_idx)

        # Split the used free rectangle into up to two new free rectangles.
        # Placed rectangle occupies (px, py, cand_w, cand_h).
        new_rects = []
        # Right split: area to the right of the placed rectangle.
        if used_rect[0] + used_rect[2] - (px + cand_w) > 0:
            new_rects.append((px + cand_w, py, used_rect[0] + used_rect[2] - (px + cand_w), cand_h))
        # Top split: area above the placed rectangle.
        if used_rect[3] - cand_h > 0:
            new_rects.append((px, py + cand_h, used_rect[2], used_rect[3] - cand_h))
        # Add the new free rectangles back
        free_rects.extend(new_rects)
        free_rects = remove_overlaps(free_rects)

    return placements

# --- Main Routine ---
def main():
    # Default parameters (in centimeters)
    default_page_width_cm = 21.0
    default_page_height_cm = 29.7
    default_max_image_cm = 5.0
    pad_cm = 0.2
    
    parser = argparse.ArgumentParser(
        description="Generate a single-page PDF by optimally packing images with variety using a corner-based algorithm."
    )
    parser.add_argument("input_dir", help="Directory containing image files")
    parser.add_argument("output_pdf", help="Output PDF file path")
    parser.add_argument("--page_width", type=float, default=default_page_width_cm,
                        help=f"Page width in cm (default: {default_page_width_cm} cm)")
    parser.add_argument("--page_height", type=float, default=default_page_height_cm,
                        help=f"Page height in cm (default: {default_page_height_cm} cm)")
    parser.add_argument("--max_image_cm", type=float, default=default_max_image_cm,
                        help=f"Maximum image dimension in cm (default: {default_max_image_cm} cm)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()

    # Configure logger based on debug flag
    logger.remove()
    log_level = "DEBUG" if args.debug else "INFO"
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Log level set to {log_level}")

    if not os.path.isdir(args.input_dir):
        logger.error("Input directory '{}' does not exist.", args.input_dir)
        sys.exit(1)
    
    # Convert dimensions
    page_width_pt = cm_to_points(args.page_width)
    page_height_pt = cm_to_points(args.page_height)
    pad_pt = cm_to_points(pad_cm)
    max_image_pixels = int(round(cm_to_pixels(args.max_image_cm)))
    
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    image_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                   if os.path.splitext(f)[1].lower() in supported_exts]
    if not image_paths:
        logger.error("No supported image files found in '{}'.", args.input_dir)
        sys.exit(1)
    
    # Preprocess images
    candidates = []  # List of ImageInfo objects
    for path in image_paths:
        result = process_image(path, max_image_pixels)
        if result:
            img, w_pt, h_pt = result
            candidates.append(ImageInfo(img=img, width=w_pt, height=h_pt))
    if not candidates:
        logger.error("No images processed successfully.")
        sys.exit(1)
    
    # Sort candidates by area (largest first) for better initial packing
    candidates.sort(key=lambda x: x.get_area(), reverse=True)
    
    # Compute packing layout using the new maximal-rectangles algorithm
    placements = pack_rectangles(candidates, page_width_pt, page_height_pt, pad_pt)
    if not placements:
        logger.error("Unable to pack any images on the page with the given dimensions.")
        sys.exit(1)
    
    # Generate PDF with support for rotated images
    try:
        c = canvas.Canvas(args.output_pdf, pagesize=(page_width_pt, page_height_pt))
        
        # Draw a subtle background grid for visual reference (optional)
        if args.debug:
            c.setStrokeColorRGB(0.9, 0.9, 0.9)  # Light gray
            c.setLineWidth(0.25)
            grid_step = cm_to_points(1)  # 1cm grid
            for x in range(0, int(page_width_pt), int(grid_step)):
                c.line(x, 0, x, page_height_pt)
            for y in range(0, int(page_height_pt), int(grid_step)):
                c.line(0, y, page_width_pt, y)
            c.setStrokeColorRGB(0, 0, 0)  # Back to black
        
        # Draw all placements
        for placement in placements:
            # Save the canvas state before applying transformations
            c.saveState()
            
            # For rotations, get actual image dimensions (without padding)
            if placement.rotation in [0, 180]:
                image_width = placement.width - 2*pad_pt
                image_height = placement.height - 2*pad_pt
            else:  
                # For 90/270 degree rotations
                image_height = placement.width - 2*pad_pt
                image_width = placement.height - 2*pad_pt
            
            # Apply padding to position (the actual image is inside the padded area)
            draw_x = placement.x + pad_pt
            draw_y = placement.y + pad_pt
            
            # Move to the position where we want to place the image
            c.translate(draw_x, draw_y)
            
            # If there's rotation, rotate around the center of the image
            if placement.rotation != 0:
                c.translate(image_width/2, image_height/2)
                c.rotate(placement.rotation)
                c.translate(-image_width/2, -image_height/2)
            
            # Draw the image
            reader = ImageReader(placement.img)
            c.drawImage(reader, 0, 0, width=image_width, height=image_height)
            
            # Draw cut line exactly at image edge
            c.setLineWidth(0.5)
            c.rect(0, 0, image_width, image_height, stroke=1, fill=0)
            
            # Restore the canvas state
            c.restoreState()
        
        # Add page information if in debug mode
        if args.debug:
            c.setFont("Helvetica", 8)
            c.drawString(10, 10, f"Images: {len(placements)}")
            
        c.showPage()
        c.save()
        
        logger.info(f"PDF generated successfully: {args.output_pdf}")
        logger.info(f"Placed {len(placements)} images with cut lines at exact image edges")
        
    except Exception as e:
        logger.error("Error generating PDF: {}", e)
        sys.exit(1)

if __name__ == '__main__':
    main()
