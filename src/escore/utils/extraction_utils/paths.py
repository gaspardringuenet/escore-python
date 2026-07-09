from pathlib import Path


class ResultsPathManager:
    """Manage EchotypeWorkflow paths."""

    def __init__(self, results_dir: str):
        self.root = Path(results_dir).resolve()
        self.root.mkdir(exist_ok=True, parents=True)

        self.workflow_state: Path = self.root / ".workflow_state.yaml"

    def region_results(self, region_id: int) -> Path:
        return self.root / f"region_{region_id:03}"

    def segmenter(self, region_id: int) -> Path:
        return self.region_results(region_id) / f"region_{region_id:03}_segmenter.pkl"

    def recipe(self, region_id: int) -> Path:
        return self.region_results(region_id) / f"region_{region_id:03}_recipe.yaml"

    def feature_data(self, region_id: int) -> Path:
        return self.region_results(region_id) / f"region_{region_id:03}_echotype.nc"
