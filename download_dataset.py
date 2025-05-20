from datasets import load_dataset
from huggingface_hub import login

login("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")

for data_name in ['m5', 'smart_meters_in_london', 'wikipedia_web_traffic', 'global_temp', 'global_wind', 'nrel_solar_power', 'sirs', 'atec', 'google_community_mobility', \
                  'neurolib', 'rebound', 'sp500', 'china_air_quality', 'measles_england', 'largest_ca', 'largest_gba', 'largest_gla', 'wikipedia_web_traffic_2000', 'wikipedia_web_traffic_20000']:
    dict_raw = load_dataset("lingfenggold/High_Dimensional_Time_Series", data_name, cache_dir="dataset")