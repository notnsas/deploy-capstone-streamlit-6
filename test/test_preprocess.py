import pytest
from utils import clean_text_advanced
import setting


def test_clean_text_advanced():
    text_list = [
        "aplikasinya bagus tapi uinya jelek",
        "lagunya bagus tapi musiknya bagus",
        "i dont like the music but the ui is good",
    ]
    assert (
        clean_text_advanced(
            ASPECT_KEYWORDS=setting.ASPECT_KEYWORDS,
            text=text_list[0],
            lang="id",
            use_stemming=True,
        )
        == "aplikasi bagus tapi ui buruk"
    )
    assert (
        clean_text_advanced(
            ASPECT_KEYWORDS=setting.ASPECT_KEYWORDS,
            text=text_list[1],
            lang="id",
            use_stemming=True,
        )
        == "lagu bagus tapi musik bagus"
    )
    assert (
        clean_text_advanced(
            ASPECT_KEYWORDS=setting.ASPECT_KEYWORDS,
            text=text_list[2],
            lang="id",
            use_stemming=True,
        )
        == "i dont like the musik but the ui is bagus"
    )
