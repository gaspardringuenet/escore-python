from pathlib import Path


class EchotypeWorkflow:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir).resolve()

        self.results_dir.mkdir(exist_ok=True, parents=True)

    def load_state(self): ...
