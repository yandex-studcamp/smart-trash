from __future__ import annotations

from .base_spotter import BaseSpotter, SpotterImageInput


class EnsembleSpotter(BaseSpotter):
    """Combine two spotters with an OR voting rule."""

    def __init__(
        self,
        anomalib_spotter: BaseSpotter,
        autoencoder_spotter: BaseSpotter,
    ) -> None:
        if not isinstance(anomalib_spotter, BaseSpotter):
            raise TypeError(
                "anomalib_spotter must be an instance of BaseSpotter."
            )
        if not isinstance(autoencoder_spotter, BaseSpotter):
            raise TypeError(
                "autoencoder_spotter must be an instance of BaseSpotter."
            )

        self.anomalib_spotter = anomalib_spotter
        self.autoencoder_spotter = autoencoder_spotter

    def predict(self, image: SpotterImageInput) -> bool:
        votes = self.predict_votes(image)
        return votes["is_anomaly"]

    def predict_votes(self, image: SpotterImageInput) -> dict[str, bool]:
        anomalib_vote = bool(self.anomalib_spotter.predict(image))
        autoencoder_vote = bool(self.autoencoder_spotter.predict(image))
        return {
            "anomalib_vote": anomalib_vote,
            "autoencoder_vote": autoencoder_vote,
            "is_anomaly": anomalib_vote or autoencoder_vote,
        }
