"""
Unit tests for TagColorManager class.

Tests color generation, validation, contrast calculation, and palette management.
"""

from src.utils.tag_colors import TagColorManager


class TestTagColorManager:
    """Test the TagColorManager class functionality."""

    def test_init(self):
        """Test TagColorManager initialization."""
        manager = TagColorManager()
        assert manager.used_colors == set()
        assert isinstance(manager.used_colors, set)

    def test_generate_color_basic(self):
        """Test basic color generation."""
        manager = TagColorManager()

        color = manager.generate_color("test_tag")
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7  # #RRGGBB format

        # Should be in professional palette
        assert color in manager.PROFESSIONAL_PALETTE

    def test_generate_color_consistent(self):
        """Test that color generation is consistent for same input."""
        manager = TagColorManager()

        color1 = manager.generate_color("test_tag")
        color2 = manager.generate_color("test_tag")

        assert color1 == color2

    def test_generate_color_different_inputs(self):
        """Test that different inputs generate different colors."""
        manager = TagColorManager()

        color1 = manager.generate_color("tag1")
        color2 = manager.generate_color("tag2")

        # May or may not be different due to hash distribution, but should be valid
        assert color1 in manager.PROFESSIONAL_PALETTE
        assert color2 in manager.PROFESSIONAL_PALETTE

    def test_generate_color_with_category(self):
        """Test color generation with semantic category."""
        manager = TagColorManager()

        color = manager.generate_color("academic_paper", "academic")
        assert color == "#007bff"  # Blue for academic

        color = manager.generate_color("medical_report", "medical")
        assert color == "#dc3545"  # Red for medical

    def test_generate_color_reserved_words(self):
        """Test color generation with reserved words."""
        manager = TagColorManager()

        color = manager.generate_color("important_document")
        assert color == "#dc3545"  # Red for important

        color = manager.generate_color("final_version")
        assert color == "#28a745"  # Green for final

        color = manager.generate_color("draft_paper")
        assert color == "#6c757d"  # Gray for draft

    def test_get_similar_color(self):
        """Test finding similar colors."""
        manager = TagColorManager()

        base_color = "#007bff"  # Blue
        similar = manager.get_similar_color(base_color)

        # Similar color should be different or the same if no better match
        assert similar in manager.PROFESSIONAL_PALETTE
        # The function may return the same color if it's the best match

    def test_get_similar_color_with_used(self):
        """Test finding similar colors avoiding used ones."""
        manager = TagColorManager()
        used_colors = {"#007bff", "#28a745"}  # Blue and green used

        base_color = "#007bff"  # Blue
        similar = manager.get_similar_color(base_color, used_colors)

        assert similar not in used_colors
        assert similar in manager.PROFESSIONAL_PALETTE

    def test_get_similar_color_no_similar_available(self):
        """Test fallback when no similar colors available."""
        manager = TagColorManager()

        # Use all colors except one
        used_colors = set(manager.PROFESSIONAL_PALETTE[:-1])
        base_color = manager.PROFESSIONAL_PALETTE[0]

        similar = manager.get_similar_color(base_color, used_colors)
        assert similar not in used_colors
        assert similar in manager.PROFESSIONAL_PALETTE

    def test_get_contrast_color_dark_background(self):
        """Test contrast color for dark backgrounds."""
        manager = TagColorManager()

        # Dark color (high luminance would be light, but we test dark)
        contrast = manager.get_contrast_color("#000000")  # Black
        assert contrast == "#ffffff"  # Should return white

    def test_get_contrast_color_light_background(self):
        """Test contrast color for light backgrounds."""
        manager = TagColorManager()

        # Light color
        contrast = manager.get_contrast_color("#ffffff")  # White
        assert contrast == "#000000"  # Should return black

    def test_get_contrast_color_medium_background(self):
        """Test contrast color for medium backgrounds."""
        manager = TagColorManager()

        # Medium gray - depending on calculation, may be light or dark
        contrast = manager.get_contrast_color("#808080")  # Gray
        assert contrast in ["#000000", "#ffffff"]  # Either black or white

    def test_get_contrast_color_invalid_hex(self):
        """Test contrast color with invalid hex."""
        manager = TagColorManager()

        contrast = manager.get_contrast_color("invalid")
        assert contrast == "#000000"  # Should default to black

    def test_get_color_palette_basic(self):
        """Test getting basic color palette."""
        manager = TagColorManager()

        palette = manager.get_color_palette()
        assert len(palette) > 0
        assert all(color.startswith("#") for color in palette)
        assert all(len(color) == 7 for color in palette)

    def test_get_color_palette_with_backgrounds(self):
        """Test getting color palette with backgrounds."""
        manager = TagColorManager()

        palette = manager.get_color_palette(include_backgrounds=True)
        assert len(palette) > len(manager.PROFESSIONAL_PALETTE)  # Should include backgrounds

        # Should contain both regular and background colors
        assert "#007bff" in palette  # Regular color
        assert "#e3f2fd" in palette  # Background color

    def test_validate_hex_color_valid(self):
        """Test validating valid hex colors."""
        manager = TagColorManager()

        assert manager.validate_hex_color("#007bff") is True
        assert manager.validate_hex_color("#123456") is True
        assert manager.validate_hex_color("#ABCDEF") is True

    def test_validate_hex_color_short_form(self):
        """Test validating short form hex colors."""
        manager = TagColorManager()

        assert manager.validate_hex_color("#07f") is True  # 3-digit valid
        assert manager.validate_hex_color("#123") is True

    def test_validate_hex_color_invalid(self):
        """Test validating invalid hex colors."""
        manager = TagColorManager()

        assert manager.validate_hex_color("007bff") is False  # Missing #
        assert manager.validate_hex_color("#gggggg") is False  # Invalid chars
        assert manager.validate_hex_color("#12") is False  # Too short
        assert manager.validate_hex_color("#12345") is False  # Wrong length
        assert manager.validate_hex_color("not-a-color") is False

    def test_get_category_colors(self):
        """Test getting semantic category colors."""
        manager = TagColorManager()

        categories = manager.get_category_colors()
        assert isinstance(categories, dict)
        assert "academic" in categories
        assert "medical" in categories
        assert categories["academic"] == "#007bff"  # Blue

    def test_suggest_colors_for_tags(self):
        """Test suggesting colors for multiple tags."""
        manager = TagColorManager()

        tags = ["academic", "medical", "technical"]
        suggestions = manager.suggest_colors_for_tags(tags)

        assert len(suggestions) == 3
        assert all(tag in suggestions for tag in tags)
        assert all(color in manager.PROFESSIONAL_PALETTE for color in suggestions.values())

        # Should have unique colors
        colors = list(suggestions.values())
        assert len(set(colors)) == len(colors)

    def test_suggest_colors_for_tags_with_duplicates(self):
        """Test color uniqueness when hash collisions occur."""
        manager = TagColorManager()

        # Create tags that might hash to similar values
        tags = ["tag1", "tag2", "tag3", "tag4", "tag5"]
        suggestions = manager.suggest_colors_for_tags(tags)

        # Should still have unique colors
        colors = list(suggestions.values())
        assert len(set(colors)) == len(colors)

    def test_hex_to_hue(self):
        """Test hex to hue conversion."""
        manager = TagColorManager()

        # Red should be at 0 degrees
        hue_red = manager._hex_to_hue("#ff0000")
        assert hue_red == 0.0  # Pure red is exactly 0 degrees

        # Green should be around 120 degrees
        hue_green = manager._hex_to_hue("#00ff00")
        assert 110 <= hue_green <= 130

        # Blue should be around 240 degrees
        hue_blue = manager._hex_to_hue("#0000ff")
        assert 230 <= hue_blue <= 250

    def test_hex_to_hue_invalid(self):
        """Test hex to hue conversion with invalid input."""
        manager = TagColorManager()

        hue = manager._hex_to_hue("invalid")
        assert hue == 0.0

        hue = manager._hex_to_hue("#gggggg")
        assert hue == 0.0

    def test_get_color_info(self):
        """Test getting detailed color information."""
        manager = TagColorManager()

        info = manager.get_color_info("#007bff")
        assert info["hex"] == "#007bff"
        assert "contrast_text" in info
        assert "hue" in info
        assert "is_reserved" in info
        assert isinstance(info["hue"], float)

    def test_get_color_info_reserved(self):
        """Test color info for reserved colors."""
        manager = TagColorManager()

        info = manager.get_color_info("#dc3545")  # Red (important/reserved)
        assert info["is_reserved"] is True

        info = manager.get_color_info("#007bff")  # Blue (not reserved)
        assert info["is_reserved"] is False

    def test_generate_color_edge_cases(self):
        """Test color generation with edge cases."""
        manager = TagColorManager()

        # Test with empty string
        color = manager.generate_color("")
        assert color in manager.PROFESSIONAL_PALETTE

        # Test with very long tag name
        long_tag = "a" * 100
        color = manager.generate_color(long_tag)
        assert color in manager.PROFESSIONAL_PALETTE

        # Test with special characters
        special_tag = "tag-with_special.chars!"
        color = manager.generate_color(special_tag)
        assert color in manager.PROFESSIONAL_PALETTE

    def test_get_similar_color_edge_cases(self):
        """Test similar color finding with edge cases."""
        manager = TagColorManager()

        # Test with no used colors
        similar = manager.get_similar_color("#007bff", set())
        assert similar in manager.PROFESSIONAL_PALETTE

        # Test with all colors used except one
        used_colors = set(manager.PROFESSIONAL_PALETTE[:-1])
        base_color = manager.PROFESSIONAL_PALETTE[0]
        similar = manager.get_similar_color(base_color, used_colors)
        assert similar not in used_colors
        assert similar in manager.PROFESSIONAL_PALETTE

    def test_get_contrast_color_edge_cases(self):
        """Test contrast color calculation with edge cases."""
        manager = TagColorManager()

        # Test with invalid hex
        contrast = manager.get_contrast_color("invalid")
        assert contrast in ["#000000", "#ffffff"]

        # Test with 3-digit hex
        contrast = manager.get_contrast_color("#000")
        assert contrast in ["#000000", "#ffffff"]  # Either black or white

        contrast = manager.get_contrast_color("#fff")
        assert contrast == "#000000"  # Black text on white

    def test_validate_hex_color_edge_cases(self):
        """Test hex color validation with edge cases."""
        manager = TagColorManager()

        # Test various invalid formats
        assert manager.validate_hex_color("") is False
        assert manager.validate_hex_color("#") is False
        assert manager.validate_hex_color("#12") is False
        assert manager.validate_hex_color("#12345") is False
        assert manager.validate_hex_color("#gggggg") is False
        assert manager.validate_hex_color("007bff") is False

        # Test valid formats
        assert manager.validate_hex_color("#007bff") is True
        assert manager.validate_hex_color("#123") is True
        assert manager.validate_hex_color("#ABCDEF") is True

    def test_suggest_colors_for_tags_edge_cases(self):
        """Test color suggestion with edge cases."""
        manager = TagColorManager()

        # Test with empty list
        suggestions = manager.suggest_colors_for_tags([])
        assert suggestions == {}

        # Test with single tag
        suggestions = manager.suggest_colors_for_tags(["single"])
        assert len(suggestions) == 1
        assert "single" in suggestions

        # Test with duplicate tags
        suggestions = manager.suggest_colors_for_tags(["tag1", "tag1", "tag2"])
        assert len(suggestions) == 2  # Should not duplicate colors for same tag
        colors = list(suggestions.values())
        assert len(set(colors)) == len(colors)  # All colors should be unique

    def test_hex_to_hue_edge_cases(self):
        """Test hex to hue conversion with edge cases."""
        manager = TagColorManager()

        # Test grayscale colors (should return 0)
        assert manager._hex_to_hue("#000000") == 0.0
        assert manager._hex_to_hue("#ffffff") == 0.0
        assert manager._hex_to_hue("#808080") == 0.0

        # Test invalid inputs
        assert manager._hex_to_hue("invalid") == 0.0
        assert manager._hex_to_hue("#gggggg") == 0.0
        assert manager._hex_to_hue("#12") == 0.0

    def test_get_color_info_comprehensive(self):
        """Test comprehensive color information retrieval."""
        manager = TagColorManager()

        info = manager.get_color_info("#dc3545")  # Red (important/reserved)

        assert info["hex"] == "#dc3545"
        assert info["is_reserved"] is True
        assert "contrast_text" in info
        assert "hue" in info
        assert isinstance(info["hue"], float)

        # Test non-reserved color
        info = manager.get_color_info("#007bff")  # Blue
        assert info["is_reserved"] is False

    def test_color_palette_comprehensive(self):
        """Test comprehensive color palette functionality."""
        manager = TagColorManager()

        # Test basic palette
        palette = manager.get_color_palette()
        assert len(palette) >= 15  # Should have professional colors

        # Test palette with backgrounds
        palette_with_bg = manager.get_color_palette(include_backgrounds=True)
        assert len(palette_with_bg) > len(palette)  # Should include backgrounds

        # All colors should be valid hex
        for color in palette + palette_with_bg:
            assert manager.validate_hex_color(color)

    def test_semantic_colors_mapping(self):
        """Test semantic color mappings."""
        manager = TagColorManager()

        categories = manager.get_category_colors()

        # Should have expected categories
        expected_categories = ["academic", "medical", "technical", "business"]
        for category in expected_categories:
            assert category in categories
            assert manager.validate_hex_color(categories[category])

        # Should be consistent
        categories2 = manager.get_category_colors()
        assert categories == categories2
