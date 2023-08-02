from copy import deepcopy
from hopp.simulation.technologies.sites import SiteInfo, flatirons_site

from hopp import ROOT_DIR

def create_site(data):
    # pass files so that we're not making network calls
    solar_resource_file = ROOT_DIR / "simulation" / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
    wind_resource_file = ROOT_DIR / "simulation" / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
    return SiteInfo(data, solar_resource_file=solar_resource_file, wind_resource_file=wind_resource_file)


def assert_basic(site):
    # basic assertions based on raw site data
    assert site.lat == flatirons_site["lat"]
    assert site.lon == flatirons_site["lon"]
    assert site.n_timesteps == 8760
    assert site.n_periods_per_day == 24
    assert site.interval == 60
    assert site.urdb_label == flatirons_site["urdb_label"]


def test_site_info_init():
    # Data dict
    data = deepcopy(flatirons_site)
    site = create_site(data)
    
    # Test the attributes
    assert_basic(site)
    assert site.solar_resource is not None
    assert site.wind_resource is not None
    assert site.elec_prices is not None
    assert len(site.desired_schedule) == 0


def test_site_info_no_solar():
    data = deepcopy(flatirons_site)
    data["no_solar"] = True
    site = create_site(data)
    
    # Test the attributes
    assert_basic(site)
    assert site.solar_resource is None
    assert site.wind_resource is not None
    assert site.elec_prices is not None


def test_site_info_no_wind():
    data = deepcopy(flatirons_site)
    data["no_wind"] = True
    site = create_site(data)
    
    # Test the attributes
    assert_basic(site)
    assert site.solar_resource is not None
    assert site.wind_resource is None
    assert site.elec_prices is not None