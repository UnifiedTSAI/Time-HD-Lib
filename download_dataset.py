from datasets import load_dataset
from huggingface_hub import login

for data_name in ['M5', 'Meter', 'Wiki-20k', 'Temp', 'Wind', 'Solar', 'SIRS', 'Atec', 'Mobility', \
                  'Neurolib', 'SP500', 'Air Quality', 'Measles', 'Traffic-CA', 'Traffic-GBA', 'Traffic-GLA']:
    dict_raw = load_dataset("Time-HD-Anonymous/High_Dimensional_Time_Series", data_name, cache_dir="dataset")