# Data

Both raw files come from the World Bank's Poverty and Inequality Platform (PIP)
and are redistributed under the Creative Commons Attribution 4.0 licence
(CC BY 4.0), as stated on the World Bank Data Catalog page for dataset 0063646
(checked 2026-09-26). Cite: World Bank (2026), *Poverty and
Inequality Platform* (version 20260922_2017_01_02_PROD), pip.worldbank.org,
accessed 2026-09-26.

| File | Source | Notes |
|---|---|---|
| `raw/pip_world_100bin.csv` | PIP Percentiles dataset (World Bank Data Catalog 0063646): `https://datacatalogfiles.worldbank.org/ddh-published/0063646/DR0090251/world_100bin.csv` | Mean welfare of 100 equal-population groups per country-year, **2017 PPP** $/day |
| `raw/pip_summary_ppp2017.csv` | PIP API: `https://api.worldbank.org/pip/v1/pip?country=all&year=all&fill_gaps=false&ppp_version=2017&format=csv` | Survey table: official Gini/MLD, comparability spells, distribution type |

The API's default vintage is 2021 PPP, which does **not** match the percentile
file; `ppp_version=2017` does (bin means equal the survey means exactly).

The World Bank notes that percentile means understate within-country
inequality (inequality inside each percentile is lost) and are not a substitute
for PIP's own statistics. The paper quantifies this: indices computed on the
100 bins match PIP's official Gini within 0.004 and its MLD within 0.017.

`processed/` is rebuilt by `code/pip_panel.py` and `code/pip_disagreement.py`
(`make reproduce`).
