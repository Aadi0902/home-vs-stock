"""
market_data.py — ZIP-code → local housing-market defaults.
==========================================================

Given a 5-digit U.S. ZIP code, resolve its state (via the standard 3-digit ZIP
prefix allocation) and return realistic, data-based defaults:

    * property_tax      effective property-tax rate (% of value / yr)
    * insurance         homeowners-insurance rate  (% of value / yr)
    * home_appreciation estimated home-price growth (% / yr)
    * median_home_price typical home value in that state ($)

DATA SOURCES (state-level, mid-2020s):
    * Property-tax effective rates — Tax Foundation "Property Taxes by State".
    * Median home values           — Zillow ZHVI / U.S. Census ACS.
    * Insurance rates              — NerdWallet / Bankrate state averages
                                     (higher in catastrophe-exposed states).
    * Appreciation                 — regional trend ESTIMATE anchored to the
                                     long-run Case-Shiller ~4–5% nominal, tilted
                                     by supply/demand; treat as a starting guess.

These are STATE-level figures keyed off the ZIP prefix — they are not exact
ZIP-level data. Property tax and median price vary reliably by state; home
appreciation is an estimate and should be overridden with local comps. For true
per-ZIP data you would plug in a paid API (ATTOM, RentCast, or Zillow) in
`lookup_zip`; the fallback below keeps the app fully offline and functional.
"""

# ---------------------------------------------------------------------------
# State-level defaults.  Rates are decimals (0.011 == 1.1%).
# (property_tax, insurance, home_appreciation, median_home_price)
# ---------------------------------------------------------------------------
STATE_DATA = {
    "AL": (0.0040, 0.0110, 0.045, 230_000),
    "AK": (0.0104, 0.0080, 0.035, 360_000),
    "AZ": (0.0063, 0.0075, 0.050, 430_000),
    "AR": (0.0062, 0.0130, 0.042, 200_000),
    "CA": (0.0075, 0.0055, 0.052, 760_000),
    "CO": (0.0055, 0.0100, 0.048, 540_000),
    "CT": (0.0179, 0.0075, 0.042, 400_000),
    "DE": (0.0058, 0.0060, 0.042, 370_000),
    "DC": (0.0057, 0.0060, 0.038, 610_000),
    "FL": (0.0091, 0.0150, 0.050, 400_000),
    "GA": (0.0092, 0.0090, 0.048, 330_000),
    "HI": (0.0032, 0.0035, 0.045, 840_000),
    "ID": (0.0067, 0.0060, 0.052, 450_000),
    "IL": (0.0208, 0.0075, 0.038, 260_000),
    "IN": (0.0084, 0.0070, 0.042, 240_000),
    "IA": (0.0152, 0.0085, 0.038, 220_000),
    "KS": (0.0141, 0.0130, 0.038, 230_000),
    "KY": (0.0083, 0.0080, 0.040, 210_000),
    "LA": (0.0056, 0.0170, 0.038, 210_000),
    "ME": (0.0124, 0.0060, 0.048, 390_000),
    "MD": (0.0105, 0.0065, 0.040, 430_000),
    "MA": (0.0114, 0.0065, 0.045, 600_000),
    "MI": (0.0138, 0.0075, 0.042, 250_000),
    "MN": (0.0111, 0.0090, 0.040, 340_000),
    "MS": (0.0079, 0.0140, 0.038, 180_000),
    "MO": (0.0098, 0.0110, 0.040, 250_000),
    "MT": (0.0074, 0.0090, 0.052, 460_000),
    "NE": (0.0163, 0.0130, 0.040, 270_000),
    "NV": (0.0055, 0.0060, 0.050, 460_000),
    "NH": (0.0193, 0.0055, 0.048, 480_000),
    "NJ": (0.0223, 0.0060, 0.042, 500_000),
    "NM": (0.0067, 0.0090, 0.045, 300_000),
    "NY": (0.0173, 0.0070, 0.040, 460_000),
    "NC": (0.0082, 0.0080, 0.050, 330_000),
    "ND": (0.0098, 0.0110, 0.036, 260_000),
    "OH": (0.0153, 0.0070, 0.040, 230_000),
    "OK": (0.0090, 0.0150, 0.040, 210_000),
    "OR": (0.0093, 0.0060, 0.048, 500_000),
    "PA": (0.0149, 0.0060, 0.040, 270_000),
    "RI": (0.0140, 0.0075, 0.044, 450_000),
    "SC": (0.0057, 0.0100, 0.050, 300_000),
    "SD": (0.0117, 0.0110, 0.040, 300_000),
    "TN": (0.0067, 0.0090, 0.050, 320_000),
    "TX": (0.0168, 0.0120, 0.048, 300_000),
    "UT": (0.0057, 0.0060, 0.052, 520_000),
    "VT": (0.0183, 0.0060, 0.044, 390_000),
    "VA": (0.0087, 0.0065, 0.042, 400_000),
    "WA": (0.0087, 0.0060, 0.050, 610_000),
    "WV": (0.0058, 0.0070, 0.036, 160_000),
    "WI": (0.0161, 0.0070, 0.040, 300_000),
    "WY": (0.0061, 0.0090, 0.044, 350_000),
    "PR": (0.0060, 0.0100, 0.030, 170_000),
}

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "Washington D.C.", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico",
}

# ---------------------------------------------------------------------------
# 3-digit ZIP prefix -> state.  (start, end, state) inclusive on the prefix.
# Based on the USPS/Census ZIP-prefix allocation.
# ---------------------------------------------------------------------------
_ZIP_RANGES = [
    (5, 5, "NY"), (6, 9, "PR"),
    (10, 27, "MA"), (28, 29, "RI"), (30, 38, "NH"), (39, 49, "ME"),
    (50, 54, "VT"), (56, 59, "VT"), (55, 55, "MA"),
    (60, 69, "CT"), (70, 89, "NJ"),
    (100, 149, "NY"), (150, 196, "PA"), (197, 199, "DE"),
    (200, 205, "DC"), (206, 219, "MD"), (220, 246, "VA"), (247, 268, "WV"),
    (270, 289, "NC"), (290, 299, "SC"),
    (300, 319, "GA"), (320, 349, "FL"), (350, 369, "AL"), (370, 385, "TN"),
    (386, 397, "MS"), (398, 399, "GA"),
    (400, 427, "KY"), (430, 459, "OH"), (460, 479, "IN"), (480, 499, "MI"),
    (500, 528, "IA"), (530, 549, "WI"), (550, 567, "MN"), (570, 577, "SD"),
    (580, 588, "ND"), (590, 599, "MT"),
    (600, 629, "IL"), (630, 658, "MO"), (660, 679, "KS"), (680, 693, "NE"),
    (700, 714, "LA"), (716, 729, "AR"), (730, 732, "OK"), (733, 733, "TX"),
    (734, 749, "OK"), (750, 799, "TX"),
    (800, 816, "CO"), (820, 831, "WY"), (832, 838, "ID"), (840, 847, "UT"),
    (850, 865, "AZ"), (870, 884, "NM"), (885, 885, "TX"),
    (889, 891, "NV"), (893, 898, "NV"),
    (900, 961, "CA"), (967, 968, "HI"), (969, 969, "PR"),
    (970, 979, "OR"), (980, 994, "WA"), (995, 999, "AK"),
]


def zip_to_state(zipcode):
    """Return the 2-letter state code for a 5-digit ZIP, or None if unknown."""
    z = str(zipcode).strip()
    if len(z) < 3 or not z[:3].isdigit():
        return None
    prefix = int(z[:3])
    for lo, hi, state in _ZIP_RANGES:
        if lo <= prefix <= hi:
            return state
    return None


def lookup_zip(zipcode):
    """Resolve a ZIP to a dict of local defaults, or None if unrecognized.

    Returns keys: state, state_name, property_tax, insurance,
    home_appreciation, median_home_price.
    """
    state = zip_to_state(zipcode)
    if state is None or state not in STATE_DATA:
        return None
    ptax, ins, appr, price = STATE_DATA[state]
    return {
        "state": state,
        "state_name": STATE_NAMES.get(state, state),
        "property_tax": ptax,
        "insurance": ins,
        "home_appreciation": appr,
        "median_home_price": price,
    }
