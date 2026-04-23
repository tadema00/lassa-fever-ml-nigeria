# Data Sources and Variable Descriptions

## Required Files

### 1. `Lassa_climate_features_engineered.csv`

Weekly satellite-derived and reanalysis climate features per state.

| Column | Unit | Description |
|--------|------|-------------|
| `state` | — | State name (Bauchi, Ebonyi, Edo, Ondo, Plateau, Taraba) |
| `week_number` | integer | Cumulative epidemiological week number |
| `years` | integer | Calendar year (2018–2025) |
| `weekly_ndvi` | — | Mean weekly NDVI (MOD13A2, 0–1 scale) |
| `weekly_rainfall` | mm | Total weekly precipitation (CHIRPS) |
| `weekly_soil_moisture` | m³/m³ | Volumetric soil water content (SMAP Level 3) |
| `weekly_temp` | °C | Mean weekly land surface temperature (MOD11A2) |
| `rel_humidity` | % | Mean weekly relative humidity (ERA5 reanalysis) |
| `elevation` | m | Mean elevation (SRTM 90m, static per state) |
| `rainfall_lag1` | mm | Rainfall lagged 1 week |
| `rainfall_lag2` | mm | Rainfall lagged 2 weeks |
| `rainfall_lag4` | mm | Rainfall lagged 4 weeks |
| `temp_lag1` | °C | Temperature lagged 1 week |
| `temp_lag2` | °C | Temperature lagged 2 weeks |
| `ndvi_lag1` | — | NDVI lagged 1 week |
| `ndvi_lag2` | — | NDVI lagged 2 weeks |
| `rainfall_roll4` | mm | 4-week rolling mean rainfall |
| `temp_roll4` | °C | 4-week rolling mean temperature |
| `ndvi_roll4` | — | 4-week rolling mean NDVI |
| `rainfall_change` | mm | Week-on-week change in rainfall |
| `temp_change` | °C | Week-on-week change in temperature |

---

### 2. `Cases_rainfal_data.csv`

Weekly confirmed Lassa fever case counts per state plus local rainfall observations.

| Column | Description |
|--------|-------------|
| `years` | Calendar year |
| `weeks` | Epidemiological week number (1–52) |
| `ondo_cases` | Weekly confirmed cases — Ondo State |
| `edo_cases` | Weekly confirmed cases — Edo State |
| `bauchi_cases` | Weekly confirmed cases — Bauchi State |
| `taraba_cases` | Weekly confirmed cases — Taraba State |
| `ebonyi_cases` | Weekly confirmed cases — Ebonyi State |
| `plateau_cases` | Weekly confirmed cases — Plateau State |
| `bauchi_weekly_rain_mm` | Local weekly rainfall — Bauchi (mm) |
| `edo_weekly_rain_mm` | Local weekly rainfall — Edo (mm) |
| `ondo_weekly_rain_mm` | Local weekly rainfall — Ondo (mm) |
| `taraba_weekly_rain_mm` | Local weekly rainfall — Taraba (mm) |
| `ebonyi_weekly_rain_mm` | Local weekly rainfall — Ebonyi (mm) |
| `plateau_weekly_rain_mm` | Local weekly rainfall — Plateau (mm) |

---

## Data Sources

| Dataset | Source | Access |
|---------|--------|--------|
| Lassa fever case counts | Nigeria Centre for Disease Control (NCDC) weekly situation reports | [https://ncdc.gov.ng/diseases/sitreps](https://ncdc.gov.ng/diseases/sitreps) |
| NDVI | NASA MODIS MOD13A2 (500 m, 16-day) | [https://lpdaac.usgs.gov](https://lpdaac.usgs.gov) |
| Rainfall | CHIRPS v2.0 (0.05°, weekly) | [https://www.chc.ucsb.edu/data/chirps](https://www.chc.ucsb.edu/data/chirps) |
| Soil moisture | NASA SMAP Level 3 (9 km, daily → weekly) | [https://nsidc.org/data/smap](https://nsidc.org/data/smap) |
| Temperature | NASA MODIS MOD11A2 LST (1 km, 8-day) | [https://lpdaac.usgs.gov](https://lpdaac.usgs.gov) |
| Relative humidity | ECMWF ERA5 reanalysis (0.25°, hourly → weekly) | [https://cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) |
| Elevation | NASA SRTM 90 m Digital Elevation Model | [https://srtm.csi.cgiar.org](https://srtm.csi.cgiar.org) |

---

## Study Area

Six Lassa fever-endemic states in Nigeria:

| State | Ecological Zone | ISO Region |
|-------|----------------|------------|
| Ondo | Southern Guinea Savanna / Rainforest transition | NG-ON |
| Edo | Tropical Rainforest | NG-ED |
| Bauchi | Sudan Savanna | NG-BA |
| Taraba | Guinea Savanna | NG-TA |
| Ebonyi | Derived Savanna | NG-EB |
| Plateau | Jos Plateau | NG-PL |

---

# Data Coverage

- **Period:** January 2018 — December 2025
- **Temporal resolution:** Weekly (epidemiological weeks)
- **Total observations:** 408 state-weeks per state × 6 states = 2,448 rows
- **Total confirmed cases across study period:** 6,801

---

## Ethics & Data Availability

Case count data are aggregate, anonymised counts published in official NCDC situation reports and do not contain individual patient information. No ethical approval for data access was required. All satellite data are publicly available from the sources listed above.
