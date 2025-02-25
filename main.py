#!/usr/bin/env python3
"""
Generate a single-page PDF by tightly packing images from a directory using
an advanced guillotine bin-packing algorithm that avoids overlaps and encourages
variety. Each image is preprocessed so that its largest side is scaled down to
MAX_IMAGE_CM (default 8 cm at 300 PPI). Each image's packing rectangle is its
display dimensions padded by 0.5 cm on all sides. The algorithm uses a usage-
penalty to discourage repeating the same image too often (though images may be
repeated if necessary).

Defaults (in cm):
  - Page size: A4 (21 x 29.7)
  - Maximum image dimension: 8 cm
  - Padding on each side: 0.5 cm (yielding a 1 cm gap between drawn images)

Usage:
    python script.py INPUT_DIRECTORY OUTPUT_PDF [--page_width WIDTH_CM]
                                           [--page_height HEIGHT_CM]
                                           [--max_image_cm MAX_IMAGE]
"""
import os
import sys
import argparse
from io import BytesIO

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from loguru import logger

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

# --- Image Preprocessing ---
def process_image(image_path: str, max_pixels: int):
    """
    Open and process an image.
    If its largest pixel dimension exceeds max_pixels, scale it down proportionally.
    Returns a tuple: (PIL.Image, display_width_pt, display_height_pt).
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

# --- Advanced Guillotine Packing with Usage Penalty ---
def prune_free_rectangles(free_rects):
    """Prune free rectangles that are fully contained within another."""
    pruned = []
    for i, r in enumerate(free_rects):
        rx, ry, rw, rh = r
        redundant = False
        for j, r2 in enumerate(free_rects):
            if i != j:
                r2x, r2y, r2w, r2h = r2
                if rx >= r2x and ry >= r2y and (rx + rw) <= (r2x + r2w) and (ry + rh) <= (r2y + r2h):
                    redundant = True
                    break
        if not redundant:
            pruned.append(r)
    return pruned

def pack_rectangles(candidates, bin_width: float, bin_height: float, pad: float, penalty: float = 10.0):
    """
    Pack candidate rectangles (each from a unique image) into the bin (page) using a guillotine algorithm.
    
    Each candidate is a tuple (w, h, img) with display dimensions in points.
    Its effective dimensions are (w + 2*pad, h + 2*pad).
    An unlimited copy of each candidate is available, but a usage penalty is added to discourage
    overusing any one candidate.
    
    The algorithm:
      - Maintains a list of free rectangles (starting with the full page).
      - Maintains a usage_count for each candidate (by index).
      - For every free rectangle and candidate, if the candidate's effective rectangle fits,
        computes a score = min(free_width - eff_w, free_height - eff_h) + usage_count * penalty.
      - Selects the placement with the lowest modified score.
      - Places the candidate at the free rectangle's bottom-left corner.
      - Increments its usage count.
      - Splits the free rectangle into two along the axis with the smaller leftover.
      - Repeats until no candidate fits in any free rectangle.
    
    Returns a list of placements: (img, x, y, w, h)
      where (x, y) is the bottom-left corner of the padded rectangle and the image is drawn at (x+pad, y+pad).
    """
    free_rects = [(0, 0, bin_width, bin_height)]
    placements = []
    usage_count = {i: 0 for i in range(len(candidates))}
    
    # We do not sort candidates strictly by area; variety is enforced by the usage penalty.
    while True:
        best_score = None
        best_choice = None  # (free_idx, fx, fy, candidate_index, candidate tuple)
        for free_idx, free in enumerate(free_rects):
            fx, fy, fw, fh = free
            for i, (w, h, img) in enumerate(candidates):
                eff_w = w + 2 * pad
                eff_h = h + 2 * pad
                if eff_w <= fw and eff_h <= fh:
                    leftover = min(fw - eff_w, fh - eff_h)
                    score = leftover + usage_count[i] * penalty
                    if best_score is None or score < best_score:
                        best_score = score
                        best_choice = (free_idx, fx, fy, i, w, h, img)
        if best_choice is None:
            break
        free_idx, fx, fy, cand_idx, w, h, img = best_choice
        eff_w = w + 2 * pad
        eff_h = h + 2 * pad
        # Place candidate at (fx, fy). The drawn image is at (fx+pad, fy+pad).
        placements.append((img, fx + pad, fy + pad, w, h))
        usage_count[cand_idx] += 1
        logger.info("Placed image (size: {:.2f}x{:.2f} pt) from candidate {} at ({:.2f}, {:.2f}).", w, h, cand_idx, fx, fy)
        # Remove the used free rectangle.
        used = free_rects.pop(free_idx)
        ux, uy, uw, uh = used
        new_free = []
        # Determine leftover widths and heights.
        leftover_w = uw - eff_w
        leftover_h = uh - eff_h
        # Choose split axis based on the smaller leftover.
        if leftover_w < leftover_h:
            if leftover_w > 0:
                new_free.append((fx + eff_w, uy, leftover_w, uh))
            if uh - eff_h > 0:
                new_free.append((fx, fy + eff_h, eff_w, uh - eff_h))
        else:
            if uh - eff_h > 0:
                new_free.append((ux, fy + eff_h, uw, uh - eff_h))
            if leftover_w > 0:
                new_free.append((fx + eff_w, uy, leftover_w, eff_h))
        free_rects.extend(new_free)
        free_rects = prune_free_rectangles(free_rects)
    logger.info("Total placements: {}", len(placements))
    return placements

# --- Main Routine ---
def main():
    # Default parameters (in cm)
    default_page_width_cm = 21.0      # A4 width
    default_page_height_cm = 29.7     # A4 height
    default_max_image_cm = 8.0        # Maximum image side on paper
    pad_cm = 0.5                      # Padding on each side (ensuring 1 cm gap)
    
    parser = argparse.ArgumentParser(
        description="Generate a single-page PDF by optimally packing images (with variety) "
                    "using an advanced guillotine bin-packing algorithm that avoids overlaps."
    )
    parser.add_argument("input_dir", help="Directory containing image files")
    parser.add_argument("output_pdf", help="Output PDF file path")
    parser.add_argument("--page_width", type=float, default=default_page_width_cm,
                        help=f"Page width in cm (default: {default_page_width_cm} cm)")
    parser.add_argument("--page_height", type=float, default=default_page_height_cm,
                        help=f"Page height in cm (default: {default_page_height_cm} cm)")
    parser.add_argument("--max_image_cm", type=float, default=default_max_image_cm,
                        help=f"Maximum image dimension in cm (default: {default_max_image_cm} cm)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        logger.error("Input directory '{}' does not exist.", args.input_dir)
        sys.exit(1)
    
    # Convert dimensions.
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
    
    # Preprocess each image.
    candidates = []  # List of tuples: (width_pt, height_pt, image)
    for path in image_paths:
        result = process_image(path, max_image_pixels)
        if result:
            img, w_pt, h_pt = result
            candidates.append((w_pt, h_pt, img))
    if not candidates:
        logger.error("No images processed successfully.")
        sys.exit(1)
    
    # Compute the packing layout using the advanced guillotine algorithm with usage penalty.
    placements = pack_rectangles(candidates, page_width_pt, page_height_pt, pad_pt, penalty=10.0)
    if not placements:
        logger.error("Unable to pack any images on the page with the given dimensions.")
        sys.exit(1)
    
    # Generate PDF.
    try:
        c = canvas.Canvas(args.output_pdf, pagesize=(page_width_pt, page_height_pt))
        for img, x, y, w, h in placements:
            reader = ImageReader(img)
            c.drawImage(reader, x, y, width=w, height=h)
        c.showPage()
        c.save()
    except Exception as e:
        logger.error("Error generating PDF: {}", e)
        sys.exit(1)
    
    logger.info("PDF generated successfully: {}", args.output_pdf)

if __name__ == '__main__':
    main()
