from dataclasses import dataclass

@dataclass
class Image:
    directory: str
    file_name_prefix: str


Cape_City_South_Africa = Image(
    directory="2024_Cape_City_South_Africa",
    file_name_prefix="T34HBG_20240130T082109"
)

Park_Fire_1 = Image(
    directory="2024_Park_Fire/img1",
    file_name_prefix="T10TEK_20240725T185921"
)

Park_Fire_2 = Image(
    directory="2024_Park_Fire/img2",
    file_name_prefix="T10TFK_20240727T184919"
)

Porto_Wildfire_Portugal = Image(
    directory="2024_Porto_Wildfire_Portugal",
    file_name_prefix="T29TNF_20240918T113321"
)

Cape_City_Mountain_Small_Wildfire = Image(
    directory="2025_Cape_City_Mountain_Small_Wildfire",
    file_name_prefix="T34HBH_20250223T081839"
)

Flin_Flon = Image(
    directory="2025_Flin_Flon",
    file_name_prefix="T13UFA_20250602T175931"
)

Malibu_Wildfire = Image(
    directory="2025_Malibu_Wildfire",
    file_name_prefix="T11SLT_20250107T183649"
)

Montreal_Lake = Image(
    directory="2025_Montreal_Lake",
    file_name_prefix="T13UDA_20250531T182821"
)


def get_band_paths(img: Image):
    """
    Returns the paths to the bands for a given image.
    """
    base_path = f"images/{img.directory}"
    
    # Infrared bands
    b12_path = f"{base_path}/infrared/{img.file_name_prefix}_B12_20m.jp2"
    b11_path = f"{base_path}/infrared/{img.file_name_prefix}_B11_20m.jp2"
    b8a_path = f"{base_path}/infrared/{img.file_name_prefix}_B8a_20m.jp2"
    
    # True-Color bands
    b04_path = f"{base_path}/color/{img.file_name_prefix}_B04_20m.jp2"
    b03_path = f"{base_path}/color/{img.file_name_prefix}_B03_20m.jp2"
    b02_path = f"{base_path}/color/{img.file_name_prefix}_B02_20m.jp2"

    # Cloud mask (if available)
    cm_path = f"{base_path}/MSK_CLDPRB_20m.jp2"
    
    return b12_path, b11_path, b8a_path, b04_path, b03_path, b02_path, cm_path