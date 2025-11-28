"""
Advanced color management system for document tags.

Provides intelligent color selection, consistency, and visual organization
for tag display and management.
"""

import hashlib
import random
from typing import Any, Dict, List, Optional, Set


class TagColorManager:
    """
    Advanced color management for document tags.

    Provides consistent color assignment, visual distinction, and
    intelligent color selection for optimal tag display.
    """

    # Professional color palette with good contrast and accessibility
    PROFESSIONAL_PALETTE = [
        "#007bff",  # Blue
        "#28a745",  # Green
        "#dc3545",  # Red
        "#ffc107",  # Yellow/Amber
        "#6f42c1",  # Purple
        "#e83e8c",  # Pink
        "#fd7e14",  # Orange
        "#20c997",  # Teal
        "#17a2b8",  # Cyan
        "#6c757d",  # Gray
        "#343a40",  # Dark Gray
        "#f8f9fa",  # Light Gray
        "#6610f2",  # Indigo
        "#d63384",  # Magenta
        "#fd7e14",  # Orange (duplicate for variety)
        "#198754",  # Dark Green
        "#0d6efd",  # Primary Blue
        "#6f42c1",  # Bootstrap Purple
        "#e3f2fd",  # Light Blue background
        "#f3e5f5",  # Light Purple background
    ]

    # Semantic color mappings for common tag categories
    SEMANTIC_COLORS = {
        "academic": "#007bff",  # Blue for academic/research
        "technical": "#28a745",  # Green for technical
        "business": "#ffc107",  # Yellow for business
        "personal": "#e83e8c",  # Pink for personal
        "medical": "#dc3545",  # Red for medical/health
        "legal": "#6c757d",  # Gray for legal
        "finance": "#20c997",  # Teal for finance
        "science": "#6610f2",  # Indigo for science
        "education": "#fd7e14",  # Orange for education
        "technology": "#17a2b8",  # Cyan for technology
        "creative": "#d63384",  # Magenta for creative
        "general": "#6c757d",  # Gray for general
    }

    # Reserved colors for system tags
    RESERVED_COLORS = {
        "important": "#dc3545",  # Red for important
        "urgent": "#dc3545",  # Red for urgent
        "review": "#ffc107",  # Yellow for review
        "draft": "#6c757d",  # Gray for draft
        "final": "#28a745",  # Green for final
        "archive": "#343a40",  # Dark gray for archive
    }

    def __init__(self):
        """Initialize the color manager."""
        self.used_colors: Set[str] = set()

    def generate_color(self, tag_name: str, category: Optional[str] = None) -> str:
        """
        Generate a consistent color for a tag based on its name.

        Uses hash-based color generation for consistency, with fallback
        to semantic colors for known categories.

        Args:
            tag_name: The tag name
            category: Optional category hint for semantic coloring

        Returns:
            Hex color code
        """
        # Check for semantic color match
        if category and category.lower() in self.SEMANTIC_COLORS:
            return self.SEMANTIC_COLORS[category.lower()]

        # Check for reserved color match
        tag_lower = tag_name.lower()
        for reserved_tag, color in self.RESERVED_COLORS.items():
            if reserved_tag in tag_lower:
                return color

        # Generate consistent color based on tag name hash
        hash_obj = hashlib.md5(tag_name.encode())
        hash_int = int(hash_obj.hexdigest(), 16)

        # Use hash to select from professional palette
        color_index = hash_int % len(self.PROFESSIONAL_PALETTE)
        return self.PROFESSIONAL_PALETTE[color_index]

    def get_similar_color(self, base_color: str, used_colors: Optional[Set[str]] = None) -> str:
        """
        Find a visually similar but distinct color.

        Useful for related tags or tag variations.

        Args:
            base_color: Base color to find similar to
            used_colors: Set of colors already in use

        Returns:
            Hex color code for similar but distinct color
        """
        used = used_colors or self.used_colors

        # Find colors in the same hue family
        base_hue = self._hex_to_hue(base_color)

        # Look for colors within 30 degrees hue difference
        similar_colors = []
        for color in self.PROFESSIONAL_PALETTE:
            if color not in used:
                hue = self._hex_to_hue(color)
                hue_diff = min(abs(base_hue - hue), 360 - abs(base_hue - hue))
                if hue_diff <= 30:  # Within 30 degrees
                    similar_colors.append((color, hue_diff))

        if similar_colors:
            # Return the most similar available color
            similar_colors.sort(key=lambda x: x[1])
            return similar_colors[0][0]

        # Fallback: return a random unused color
        available_colors = [c for c in self.PROFESSIONAL_PALETTE if c not in used]
        if available_colors:
            return random.choice(available_colors)

        # Ultimate fallback
        return self.PROFESSIONAL_PALETTE[0]

    def get_contrast_color(self, background_color: str) -> str:
        """
        Get a contrasting text color for the given background.

        Args:
            background_color: Background color hex

        Returns:
            White or black text color for contrast
        """
        # Simple luminance-based contrast calculation
        # For simplicity, return white for dark colors, black for light
        try:
            # Convert hex to RGB
            hex_color = background_color.lstrip("#")
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)

            # Calculate luminance
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

            return "#ffffff" if luminance < 0.5 else "#000000"
        except (ValueError, IndexError):
            return "#000000"  # Default to black

    def get_color_palette(self, include_backgrounds: bool = False) -> List[str]:
        """
        Get the full color palette.

        Args:
            include_backgrounds: Whether to include light background colors

        Returns:
            List of hex color codes
        """
        palette = self.PROFESSIONAL_PALETTE.copy()

        if include_backgrounds:
            # Add light background variants
            backgrounds = [
                "#e3f2fd",  # Light blue
                "#f3e5f5",  # Light purple
                "#e8f5e8",  # Light green
                "#fff3cd",  # Light yellow
                "#f8d7da",  # Light red
            ]
            palette.extend(backgrounds)

        return palette

    def validate_hex_color(self, color: str) -> bool:
        """
        Validate hex color format.

        Args:
            color: Color string to validate

        Returns:
            True if valid hex color
        """
        if not color.startswith("#"):
            return False

        hex_part = color[1:]
        if len(hex_part) not in [3, 6]:
            return False

        try:
            int(hex_part, 16)
            return True
        except ValueError:
            return False

    def get_category_colors(self) -> Dict[str, str]:
        """
        Get semantic colors for different categories.

        Returns:
            Dict mapping category names to colors
        """
        return self.SEMANTIC_COLORS.copy()

    def suggest_colors_for_tags(self, tags: List[str]) -> Dict[str, str]:
        """
        Suggest colors for a list of tags, ensuring visual distinction.

        Args:
            tags: List of tag names

        Returns:
            Dict mapping tag names to suggested colors
        """
        suggestions = {}
        used_colors = set()

        for tag in tags:
            color = self.generate_color(tag)
            # Ensure uniqueness
            while color in used_colors and len(used_colors) < len(self.PROFESSIONAL_PALETTE):
                color = self.get_similar_color(color, used_colors)

            suggestions[tag] = color
            used_colors.add(color)

        return suggestions

    def _hex_to_hue(self, hex_color: str) -> float:
        """
        Convert hex color to hue value (0-360).

        Args:
            hex_color: Hex color code

        Returns:
            Hue value in degrees
        """
        try:
            # Convert hex to RGB
            hex_color = hex_color.lstrip("#")
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0

            # Find max and min values
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            diff = max_val - min_val

            # Calculate hue
            if diff == 0:
                hue = 0
            elif max_val == r:
                hue = (60 * ((g - b) / diff) + 360) % 360
            elif max_val == g:
                hue = (60 * ((b - r) / diff) + 120) % 360
            else:
                hue = (60 * ((r - g) / diff) + 240) % 360

            return hue
        except (ValueError, IndexError):
            return 0.0

    def get_color_info(self, color: str) -> Dict[str, Any]:
        """
        Get detailed information about a color.

        Args:
            color: Hex color code

        Returns:
            Dict with color information
        """
        return {
            "hex": color,
            "contrast_text": self.get_contrast_color(color),
            "hue": self._hex_to_hue(color),
            "is_reserved": color in self.RESERVED_COLORS.values(),
            "category": None,  # Could be enhanced to detect category
        }
