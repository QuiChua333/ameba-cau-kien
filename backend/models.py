from typing import List, Optional
from pydantic import BaseModel, Field

class ProjectInfo(BaseModel):
    """Information about the construction project found in drawing title blocks."""
    project_name: str = Field(..., description="Project name extracted from the Japanese text (e.g., JAしまね...)")
    drawing_date: str = Field(..., description="Date of the drawing in YYYY-MM-DD format")
    drawing_scale: str = Field(..., description="Scale of the drawing (e.g., 1:200)")

class Dimensions(BaseModel):
    """Dimensions of a foundation type."""
    Lx: float = Field(..., description="Length dimension Lx in mm")
    Ly: float = Field(..., description="Length dimension Ly in mm")
    D: str = Field(..., description="Depth dimension D in mm. Can be a range like '700~300'.")

class ItemRegion(BaseModel):
    """Bounding box of a specific item on the page."""
    page: int = Field(..., description="Page number where the item was found (1-based index)")
    ymin: int = Field(..., description="Top Y coordinate (0-1000 scale)")
    xmin: int = Field(..., description="Left X coordinate (0-1000 scale)")
    ymax: int = Field(..., description="Bottom Y coordinate (0-1000 scale)")
    xmax: int = Field(..., description="Right X coordinate (0-1000 scale)")

class FoundationItem(BaseModel):
    """Data for a single foundation type."""
    type: str = Field(..., description="Foundation type identifier (e.g., F1, F2, FG1, FW1)")
    dimensions: Dimensions = Field(..., description="Physical dimensions of the foundation (Lx, Ly, D). Fill with 0 if not applicable (e.g. for Beams).")
    top_elevation: Optional[float] = Field(None, description="Top elevation of structure relative to ▽GL in mm. Negative if below GL (e.g., -200), positive if above.")
    top_elevation_alt: Optional[float] = Field(None, description="Alternate 天端 read from the 断面 (cross-section) drawing when it CONFLICTS with the chosen floor-plan top_elevation. None when they agree. Flags the cell for manual review (Excel: red cell + note).")
    rebar_x: str = Field(..., description="Rebar spec for X direction (left arrow / horizontal)")
    rebar_y: str = Field(..., description="Rebar spec for Y direction (up arrow / vertical)")
    remarks: str = Field(..., description="Remarks column content. Return empty string if blank.")
    classification: str = Field(..., description="Calculated classification: 'DD', 'D', 'TNF', or 'FW/FG' for beams.")
    region: ItemRegion | None = Field(None, description="Bounding box of the item on the drawing.")
    image_base64: str | None = Field(None, description="Base64 encoded cropped image of this specific item.")

class OvalGLItem(BaseModel):
    """Detected Oval GL marker."""
    text: str = Field(..., description="The text content of the GL marker (e.g., 'GL+100', 'GL±0').")
    region: ItemRegion = Field(..., description="Bounding box of the marker.")


class TableRegion(BaseModel):
    """Bounding box of the identified table."""
    page: int = Field(..., description="Page number where the table was found (1-based index)")
    ymin: int = Field(..., description="Top Y coordinate (0-1000 scale)")
    xmin: int = Field(..., description="Left X coordinate (0-1000 scale)")
    ymax: int = Field(..., description="Bottom Y coordinate (0-1000 scale)")
    xmax: int = Field(..., description="Right X coordinate (0-1000 scale)")

class SlabItem(BaseModel):
    """Detected Floor Slab."""
    type: str = Field(..., description="Slab type or name (e.g. S1, CS1)")
    region: ItemRegion = Field(..., description="Bounding box of the slab marker.")
    image_base64: str | None = Field(None, description="Base64 encoded cropped image of the slab item.")

class RegularFloor(BaseModel):
    """Regular floor with single elevation."""
    elevation: str = Field(..., description="Floor elevation (e.g., 'GL+45', 'GL±0', 'GL-100')")
    count: int = Field(..., description="Number of markers with same elevation (merged count)")
    regions: List[ItemRegion] = Field(default_factory=list, description="List of all bounding boxes for this elevation")
    image_base64: str | None = Field(None, description="Base64 encoded cropped image showing first occurrence.")

class SlopedFloor(BaseModel):
    """Sloped floor connecting two elevations with an arrow."""
    start_elevation: str = Field(..., description="Starting elevation (higher, e.g., 'GL+100')")
    end_elevation: str = Field(..., description="Ending elevation (lower, e.g., 'GL±0')")
    region: ItemRegion = Field(..., description="Bounding box covering both markers and arrow.")
    image_base64: str | None = Field(None, description="Base64 encoded cropped image of the sloped floor.")

class PitHoleItem(BaseModel):
    """Detected pit hole drawing (hố pit / ピット詳細図)."""
    type: str = Field(..., description="Pit name from drawing title (e.g., '側溝', 'レールのピット', 'EVピット①', 'EVピット②')")
    top_elevation: Optional[float] = Field(None, description="Distance from ▽GL to top surface of pit floor slab in mm. NEGATIVE if below GL. null if unreadable.")
    D: Optional[float] = Field(None, description="Thickness of pit bottom slab in mm. null if unreadable.")
    readable: bool = Field(True, description="True if top_elevation and D were read successfully. False if marked ※ or top surface is above/at ▽GL.")
    region: ItemRegion | None = Field(None, description="Bounding box of the pit detail drawing on the page.")
    image_base64: str | None = Field(None, description="Base64 encoded cropped image of this pit drawing.")


class ExtractionResponse(BaseModel):
    """Top-level response model for the extraction API."""
    project_info: ProjectInfo
    foundation_list: List[FoundationItem]
    slab_list: List[SlabItem] = Field(default_factory=list, description="List of detected floor slabs.")
    slab_overview_base64: str | None = Field(None, description="Base64 encoded image of the entire slab region with highlights.")
    table_region: TableRegion | None = Field(None, description="Coordinates of the foundation table")
    table_image_base64: str | None = Field(None, description="Base64 encoded cropped image of the table region")
    oval_gl_list: List[OvalGLItem] = Field(default_factory=list, description="List of detected Oval GL markers.")
    annotated_pages: List[str] = Field(default_factory=list, description="List of base64 encoded images of pages with GL markers highlighted.")
    floor_regular_list: List[RegularFloor] = Field(default_factory=list, description="List of detected regular floors (same elevation).")
    floor_sloped_list: List[SlopedFloor] = Field(default_factory=list, description="List of detected sloped floors (elevation transitions).")
    floor_overview_base64: str | None = Field(None, description="Base64 encoded image with all floor GL markers highlighted.")
    floor_plan_region: TableRegion | None = Field(None, description="Bounding box of the main floor plan drawing (平面図/基礎伏図) used for cropping.")
    pit_list: List[PitHoleItem] = Field(default_factory=list, description="List of detected pit hole drawings (ピット詳細図). Includes both readable and unreadable pits.")

