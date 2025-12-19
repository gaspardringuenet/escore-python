import os
from pathlib import Path

# First let's build a data path dictionary
BASE_DIR = Path(__file__).resolve().parent.parent

raw_data_path = BASE_DIR / "data/input/netCDF-echointegrations/"

DATA_DICT = {
    "abracos1_3pings1m_leg1" : raw_data_path / "Abracos_A1/high-res/LEG01/IRD_SOOP-BA_A_20150929T192933Z_ANTEA_FV02_ABRACOS-38-70-120-200_END-20151021T190616Z_C.nc",
    "abracos1_3pings1m_leg2" : raw_data_path / "Abracos_A1/high-res/LEG02/IRD_SOOP-BA_A_20150929T192933Z_ANTEA_FV02_ABRACOS-38-70-120-200_END-20151021T190616Z.nc",
    "abracos2_3pings1m_leg1" : raw_data_path / "Abracos_A2/high-res/LEG01/IRD_SOOP-BA_A_20170409T110633Z_ANTEA_FV02_ABRACOS-38-70-120-200_END-20170507T000620Z_C-20201002T113129Z.nc", 
    "abracos2_3pings1m_leg2" : raw_data_path / "Abracos_A2/high-res/LEG02/IRD_SOOP-BA_A_20170409T110633Z_ANTEA_FV02_ABRACOS-38-70-120-200_END-20170507T000620Z_C-20201002T160241Z.nc",
    "amazomix_3pings1m_leg1" : raw_data_path / "Amazomix/EI_3pings_1m_leg01/IRD_SOOP-BA_A_20210828T223155Z_ANTEA_FV02_AMAZOMIX2021-38-70-120-200_END-20211008T082151Z_C-20251014T104337Z.nc",
    "amazomix_3pings1m_leg2.1" : raw_data_path / "Amazomix/EI_3pings_1m_leg02/IRD_SOOP-BA_A_20210828T223155Z_ANTEA_FV02_AMAZOMIX2021-38-70-120-200_END-20211008T082151Z_C-20251014T144439Z.nc",
    "amazomix_3pings1m_leg2.2" : raw_data_path / "Amazomix/EI_3pings_1m_leg02/IRD_SOOP-BA_A_20210828T223155Z_ANTEA_FV02_AMAZOMIX2021-38-70-120-200_END-20211008T082151Z_C-20251014T145017Z.nc"
}

# Check that all data files exist in the indicated location
for key in DATA_DICT.keys():
    assert os.path.isfile(DATA_DICT.get(key)), f"{key} file not found"

# Build a survey dictionnary to group the legs
EI_DICT = {
    "abracos1_3pings1m" : ["abracos1_3pings1m_leg1", "abracos1_3pings1m_leg2"],
    "abracos2_3pings1m" : ["abracos2_3pings1m_leg1", "abracos2_3pings1m_leg2"],
    "amazomix_3pings1m" : ["amazomix_3pings1m_leg1", "amazomix_3pings1m_leg2.1", "amazomix_3pings1m_leg2.2"]
}