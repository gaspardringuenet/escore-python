from pathlib import Path
import argparse

from escore.config import load_config
from escore.io import load_survey_ds
from escore.registry import ROIRegistry
from escore.apps.echotypes.app import create_app


def main(config):

    # Fetch config for paths
    interim_dir = Path(config["paths"]["interim_dir"])
    session_name = config["session"]["name"]
    ei = config["session"]["ei"]

    # Derive paths
    work_dir = interim_dir / ei / session_name
    registry_path = work_dir / "roi_registry.db"

    # Fetch one ROI from registry
    with ROIRegistry(db_path=registry_path, root_path=HERE) as registry:
        roi_ids = registry.list_ids()

    # Load sv
    ds = load_survey_ds(survey=config["session"]["ei"], config=config)
    sv = ds["Sv"]

    # Create and run Dash app   
    app = create_app(sv, registry_path, HERE, roi_ids)
    app.run(debug=True)


if __name__ == '__main__':

    HERE = Path(__file__).resolve().parent.parent

    # Parse config argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/config.yml", help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)

    # Execute main
    main(config)
