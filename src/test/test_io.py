from src.config import load_config
from src.io import load_survey_ds, print_file_infos

if __name__ == "__main__":
    
    config = load_config("scripts/config.yml")

    for survey in config["surveys"]:
        ds = load_survey_ds(survey, config)
        print()
        print_file_infos(ds)