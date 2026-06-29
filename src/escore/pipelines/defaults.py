from sklearn.cluster import AgglomerativeClustering
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from escore.pipelines.transformers import SvDifferenceExtractor


def make_echotypes_clustering_pipeline(n_classes):

    classifier = AgglomerativeClustering(linkage="ward", n_clusters=n_classes)

    clustering_pipeline = Pipeline(
        [
            ("features", SvDifferenceExtractor()),
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )

    return clustering_pipeline
