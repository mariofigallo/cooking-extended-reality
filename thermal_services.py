"""
thermal_services.py
-------------------
Two services for thermal temperature estimation:
  - get_properties(class_name)                        -> material properties dict
  - estimate_temp(raw_temp_c, emissivity, ambient_c)  -> corrected true temperature (°C)
  - get_temp_for_object(class_name, raw_temp_c)       -> convenience wrapper for both
"""

# ── Material Properties Table ─────────────────────────────────────────────────
# Add entries following this format:
#
# "yolo_class_name": {"emissivity": 0.00, "convective_coeff": 0.0, "notes": "description"},
#
# emissivity       : float 0.01–1.0  — how efficiently the surface radiates vs a blackbody
#                    (1.0 = perfect blackbody, ~0.05 for polished metals, ~0.95 for skin/ceramics)
# convective_coeff : float W/m²K     — natural convection coefficient (reserved for future use)
# notes            : str             — human-readable description of the material

MATERIAL_TABLE = {
    # "cup":    {"emissivity": 0.90, "convective_coeff": 8.0, "notes": "ceramic mug"},
    # "person": {"emissivity": 0.98, "convective_coeff": 5.0, "notes": "human skin"},
    # ... add your entries here

    "default": {"emissivity": 0.90, "convective_coeff": 8.0, "notes": "generic non-metallic surface"},
    "stainless steel pan": {"emissivity": 0.25, "convective_coeff": 0.19, "notes": "need to update empirically"},
    "pan": {"emissivity": 0.95, "convective_coeff": 8.0, "notes": "granite coated non-stick pan"}
}


# ── Service 1: Property Lookup ────────────────────────────────────────────────

def get_properties(class_name: str) -> dict:
    """
    Return material properties for a YOLO class name.
    Falls back to 'default' if the class isn't in the table.
    """
    key = class_name.lower().strip()
    props = MATERIAL_TABLE.get(key, MATERIAL_TABLE["default"]).copy()
    props["class_name"] = key
    props["from_table"] = key in MATERIAL_TABLE
    return props


# ── Service 2: Temperature Estimation ────────────────────────────────────────

def estimate_temp(raw_temp_c: float, emissivity: float, ambient_temp_c: float = 22.0) -> float:
    """
    Estimate true surface temperature from a thermal camera reading.

    Thermal cameras measure radiated power. A surface with emissivity < 1
    appears cooler than it really is because it also reflects ambient radiation.

    Correction (Stefan-Boltzmann):
        T_true^4 = (T_raw^4 - (1 - emissivity) * T_ambient^4) / emissivity

    All temps converted to Kelvin internally, result returned in Celsius.
    """
    if not (0.01 <= emissivity <= 1.0):
        raise ValueError(f"Emissivity must be 0.01–1.0, got {emissivity}")
       
    T_raw = raw_temp_c     + 273.15
    T_amb = ambient_temp_c + 273.15

    T_true_4 = (T_raw**4 - (1.0 - emissivity) * T_amb**4) / emissivity
    T_true_4 = max(T_true_4, 0.0)  # guard against floating point edge cases

    return round(T_true_4 ** 0.25 - 273.15, 1)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def get_temp_for_object(class_name: str, raw_temp_c: float, ambient_temp_c: float = 22.0) -> dict:
    """
    Looks up properties then estimates temp. This is what display_yolo.py calls.

    Returns:
        {
            "class_name"  : "cup",
            "true_temp_c" : 41.3,
            "emissivity"  : 0.90,
            "from_table"  : True,   # False means fallback to default was used
            "notes"       : "ceramic mug"
        }
    """
    props = get_properties(class_name)
    true_temp = estimate_temp(raw_temp_c, props["emissivity"], ambient_temp_c)
    return {
        "class_name"  : class_name,
        "true_temp_c" : true_temp,
        "emissivity"  : props["emissivity"],
        "from_table"  : props["from_table"],
        "notes"       : props["notes"],
    }
