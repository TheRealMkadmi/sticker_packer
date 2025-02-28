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
def check_placement_collision(new_x, new_y, new_w, new_h, placements, debug=False) -> bool:
    """Check if a new placement would overlap with any existing placements."""
    epsilon = 1e-3  # small tolerance for floating point comparisons
    for i, p in enumerate(placements):
        if not (
            new_x + new_w <= p.x + epsilon or p.x + p.width <= new_x + epsilon or
            new_y + new_h <= p.y + epsilon or p.y + p.height <= new_y + epsilon
        ):
            if debug:
                logger.debug(f"Collision detected: New({new_x:.2f},{new_y:.2f},{new_w:.2f},{new_h:.2f}) "
                            f"with existing placement {i}: ({p.x:.2f},{p.y:.2f},{p.width:.2f},{p.height:.2f})")
            return True
    return False

def skyline_pack(candidates: List[ImageInfo],
                 bin_width: float,
                 bin_height: float,
                 pad: float,
                 debug: bool = False) -> List[Placement]:
    """
    Pack images using a skyline-based algorithm to minimize wasted space.
    This algorithm maintains a skyline (lowest y-value at each x-position) and places
    images at locations that minimize the waste above the skyline.
    """
    # Track usage count for each candidate
    usage_count = [0] * len(candidates)
    placements: List[Placement] = []
    
    # Initialize skyline as a flat line at y=0
    skyline = [(0, 0, bin_width)]  # (x, y, width)
    
    if debug:
        logger.debug(f"Starting skyline packing. Bin: {bin_width:.2f}x{bin_height:.2f}, Candidates: {len(candidates)}")
    
    while True:
        best_score = float('inf')
        best_placement = None
        best_skyline_idx = -1
        
        # Find best placement location along skyline
        for sky_idx, (sky_x, sky_y, sky_width) in enumerate(skyline):
            # Skip if segment is too narrow
            if sky_width < 1:
                continue
                
            for cand_idx, candidate in enumerate(candidates):
                # Penalize repeated use of the same image with exponential penalty
                penalty = usage_count[cand_idx] ** 2 * 100.0
                
                for img_w, img_h, rotation in candidate.get_rotations(pad):
                    # Skip if image doesn't fit horizontally in skyline segment
                    if img_w > sky_width:
                        continue
                        
                    # Skip if image doesn't fit vertically in bin
                    if sky_y + img_h > bin_height:
                        continue
                    
                    # Check for collisions with existing placements
                    if check_placement_collision(sky_x, sky_y, img_w, img_h, placements, debug):
                        continue
                    
                    # Calculate wasted space score - prioritize placements that fit well
                    waste_score = 0
                    
                    # Calculate how well this image fits into current skyline segment
                    # Lower is better
                    fit_width = min(img_w, sky_width)
                    waste_width = abs(sky_width - fit_width)
                    waste_score += waste_width * 0.5
                    
                    # Give better score to images that fit perfectly or close to perfectly
                    waste_score += abs(img_w - sky_width) * 0.2
                    
                    # Add penalty to avoid same image repeatedly
                    # Add a small random factor to encourage uniform distribution
                    total_score = waste_score + penalty + random.uniform(0, 5)
                    
                    if debug and cand_idx == 0:  # Only debug first candidate to avoid log spam
                        logger.debug(f"Considering: cand={cand_idx}, rot={rotation}, "
                                    f"pos=({sky_x:.2f},{sky_y:.2f}), "
                                    f"size={img_w:.2f}x{img_h:.2f}, score={total_score:.2f}")
                    
                    if total_score < best_score:
                        best_score = total_score
                        best_placement = (cand_idx, sky_x, sky_y, img_w, img_h, rotation)
                        best_skyline_idx = sky_idx
        
        # No valid placement found, we're done
        if best_placement is None:
            if debug:
                logger.debug("No more valid placements found. Packing complete.")
            break
            
        # Place the best candidate
        cand_idx, px, py, width, height, rotation = best_placement
        usage_count[cand_idx] += 1
        
        # Add placement to results
        placement = Placement(
            img_idx=cand_idx,
            img=candidates[cand_idx].img,
            x=px,
            y=py,
            width=width,
            height=height,
            rotation=rotation
        )
        placements.append(placement)
        
        if debug:
            logger.debug(f"Placed image {cand_idx} at ({px:.2f},{py:.2f}), "
                       f"size={width:.2f}x{height:.2f}, rotation={rotation}")
        
        # Update skyline
        # Remove the used segment
        used_segment = skyline.pop(best_skyline_idx)
        sky_x, sky_y, sky_width = used_segment
        
        # If we used only part of the segment width, add back the remaining part
        if px + width < sky_x + sky_width:
            remain_x = px + width
            remain_width = sky_x + sky_width - remain_x
            skyline.insert(best_skyline_idx, (remain_x, sky_y, remain_width))
        
        # Add the new elevated segment
        new_segment = (px, py + height, width)
        skyline.insert(best_skyline_idx, new_segment)
        
        # Merge adjacent segments with the same height
        merged_skyline = []
        i = 0
        while i < len(skyline):
            seg_x, seg_y, seg_w = skyline[i]
            # Look ahead to combine adjacent segments at the same height
            while i + 1 < len(skyline) and abs(skyline[i][1] - skyline[i+1][1]) < 0.001 and skyline[i][0] + skyline[i][2] >= skyline[i+1][0]:
                next_x, next_y, next_w = skyline[i+1]
                # Extend current segment
                seg_w = max(seg_w, next_x + next_w - seg_x)
                i += 1
            merged_skyline.append((seg_x, seg_y, seg_w))
            i += 1
            
        skyline = merged_skyline
        skyline.sort(key=lambda s: s[0])  # Sort by x coordinate
        
        if debug:
            logger.debug(f"Updated skyline: {skyline}")
    
    return placements

def opportunistic_pack(candidates: List[ImageInfo],
                         placements: List[Placement],
                         page_width: float,
                         page_height: float,
                         pad: float,
                         debug: bool = False) -> List[Placement]:
    """
    Opportunistically pack additional (typically smaller) images into free spaces.
    Seed positions are gathered from the corners of existing placements.
    Candidates are tried in ascending order of area.
    """
    new_placements: List[Placement] = []
    # Gather unique seed points from existing placements
    seeds = {(p.x, p.y) for p in placements}
    for p in placements:
        for corner in p.get_corners():
            seeds.add(corner)
    # Convert to a sorted list (by x then y)
    seed_list = sorted(seeds, key=lambda s: (s[0], s[1]))
    # Process candidates from smallest area upwards
    sorted_candidates = sorted(candidates, key=lambda c: c.get_area())
    
    for sx, sy in seed_list:
        for cand_idx, candidate in enumerate(sorted_candidates):
            for img_w, img_h, rotation in candidate.get_rotations(pad):
                # Check boundaries
                if sx + img_w > page_width or sy + img_h > page_height:
                    continue
                # Check collision with existing placements
                if check_placement_collision(sx, sy, img_w, img_h, placements + new_placements, debug):
                    continue
                # Found a valid placement; add it
                new_p = Placement(
                    img_idx=cand_idx,
                    img=candidate.img,
                    x=sx,
                    y=sy,
                    width=img_w,
                    height=img_h,
                    rotation=rotation
                )
                new_placements.append(new_p)
                if debug:
                    logger.debug(f"Opportunistic placement at ({sx:.2f},{sy:.2f}) with size {img_w:.2f}x{img_h:.2f} and rot {rotation}")
                # Proceed to next seed after a successful placement
                break
            else:
                continue
            break

    return new_placements

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
    parser.add_argument("--log_level", type=str, choices=["INFO", "DEBUG"], default="INFO",
                        help="Debug logging level (INFO or DEBUG)")
    
    args = parser.parse_args()

    # Configure logger based on debug level argument
    logger.remove()
    log_level = args.log_level
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
    
    # Compute packing layout using the new skyline packing algorithm
    debug_mode = log_level == "DEBUG"
    placements = skyline_pack(candidates, page_width_pt, page_height_pt, pad_pt, debug=debug_mode)
    if not placements:
        logger.error("Unable to pack any images on the page with the given dimensions.")
        sys.exit(1)
    
    # Opportunistic second pass to fill free spaces with smaller images.
    opportunistic = opportunistic_pack(candidates, placements, page_width_pt, page_height_pt, pad_pt, debug=debug_mode)
    placements.extend(opportunistic)
    
    # Generate PDF with support for rotated images
    try:
        c = canvas.Canvas(args.output_pdf, pagesize=(page_width_pt, page_height_pt))
        
        # Draw all placements with fixed rotation rendering
        for placement in placements:
            c.saveState()
            image_w = placement.width - 2*pad_pt
            image_h = placement.height - 2*pad_pt
            reader = ImageReader(placement.img)
            if placement.rotation == 90:
                # For 90° rotation, translate from bottom-left to top-left of image bounding box
                c.translate(placement.x + pad_pt + image_w, placement.y + pad_pt)
                c.rotate(90)
                c.drawImage(reader, 0, 0, width=image_h, height=image_w)
                c.setLineWidth(0.5)
                c.rect(0, 0, image_h, image_w, stroke=1, fill=0)
            else:
                c.translate(placement.x + pad_pt, placement.y + pad_pt)
                c.drawImage(reader, 0, 0, width=image_w, height=image_h)
                c.setLineWidth(0.5)
                c.rect(0, 0, image_w, image_h, stroke=1, fill=0)
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
