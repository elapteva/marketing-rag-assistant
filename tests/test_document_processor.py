from io import BytesIO

import pandas as pd
import pytest

from src.document_processor import DocumentProcessingError, extract_text


def create_excel_file_bytes() -> bytes:
    dataframe = pd.DataFrame(
        {
            "Campaign": ["Email Launch", "Social Promotion"],
            "Clicks": [1250, 2100],
            "ROI": [3.4, 4.8],
        }
    )

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, sheet_name="Campaign Results", index=False)

    return output.getvalue()


def test_extract_text_from_excel_file():
    file_bytes = create_excel_file_bytes()

    result = extract_text("campaign_results.xlsx", file_bytes)

    assert "[Worksheet: Campaign Results]" in result
    assert "Email Launch" in result
    assert "Social Promotion" in result
    assert "Clicks" in result
    assert "ROI" in result


def test_extract_text_preserves_numeric_excel_values():
    file_bytes = create_excel_file_bytes()

    result = extract_text("campaign_results.xlsx", file_bytes)

    assert "1250" in result
    assert "2100" in result
    assert "3.4" in result
    assert "4.8" in result


def test_extract_text_rejects_unsupported_file_type():
    with pytest.raises(
        DocumentProcessingError,
        match="Unsupported file type"
    ):
        extract_text("campaign_results.txt", b"Sample marketing information")


def test_extract_text_reports_invalid_excel_file():
    invalid_excel_bytes = b"This is not a valid Excel workbook."

    with pytest.raises(
        DocumentProcessingError,
        match="Could not process"
    ):
        extract_text("invalid_report.xlsx", invalid_excel_bytes)