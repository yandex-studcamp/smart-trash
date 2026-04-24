from fastapi import Request

from app.core.config import templates
from app.services.image_service import StoredImage, image_storage
from app.services.pipeline_service import pipeline_service


def map_pipeline_result_to_ui_prediction(command: int, label: str | None) -> dict[str, str | int]:
    normalized_label = (label or "").strip().lower()
    if command < 0 or normalized_label in {"none", "nothing"}:
        return {"class_id": 3, "label": "nothing"}
    if normalized_label == "paper":
        return {"class_id": 1, "label": "paper"}
    if normalized_label == "plastic":
        return {"class_id": 2, "label": "plastic"}
    if normalized_label == "other":
        return {"class_id": 0, "label": "other"}
    if command == 2:
        return {"class_id": 1, "label": "paper"}
    if command == 1:
        return {"class_id": 2, "label": "plastic"}
    if command == 0:
        return {"class_id": 0, "label": "other"}
    return {"class_id": 3, "label": "nothing"}


def render_home_page(
    request: Request,
    *,
    uploaded_image: StoredImage | None = None,
    prediction: dict[str, str | int] | None = None,
    error_message: str | None = None,
    status_code: int = 200,
):
    current_image = uploaded_image or image_storage.get_latest_image()
    current_prediction = prediction
    if current_prediction is None:
        latest_result = pipeline_service.last_result()
        if latest_result is not None:
            current_prediction = map_pipeline_result_to_ui_prediction(latest_result.command, latest_result.label)

    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={
            "uploaded_image": current_image,
            "prediction": current_prediction,
            "error_message": error_message,
        },
        status_code=status_code,
    )
