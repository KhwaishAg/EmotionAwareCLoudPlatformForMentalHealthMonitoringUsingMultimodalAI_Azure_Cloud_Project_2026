"""Design rules for the web interface, enforced so they cannot creep back in.

The interface must not use: purple gradients (or any gradients), pill-shaped buttons, invented metrics, vague
marketing copy, emoji icons, em dashes, scroll/entrance animation, or any "made with AI" tag.
"""
import colorsys
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.ai_model.fusion import explanations
from src.backend.assessment import api as api_module

FRONTEND = Path(__file__).resolve().parents[1] / "src" / "frontend"
FILES = {name: (FRONTEND / name).read_text(encoding="utf-8") for name in ("index.html", "style.css", "app.js")}
CSS = FILES["style.css"]
HTML = FILES["index.html"]
JS = FILES["app.js"]
client = TestClient(api_module.app)

EMOJI = re.compile("[\u2190-\u21ff\u2300-\u23ff\u25a0-\u27bf\u2b00-\u2bff\U0001F000-\U0001FAFF\uFE0F]")


def visible_text(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html))


class TestNoGradientsOrPurple:
    def test_no_gradients_anywhere(self):
        assert "gradient(" not in CSS.lower()
        assert "gradient" not in HTML.lower() and "background-clip" not in CSS

    def test_no_purple_or_magenta_colours(self):
        offenders = []
        for match in re.finditer(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b", CSS):
            code = match.group(1)
            code = "".join(c * 2 for c in code) if len(code) == 3 else code
            r, g, b = (int(code[i:i + 2], 16) / 255 for i in (0, 2, 4))
            h, lightness, s = colorsys.rgb_to_hls(r, g, b)
            if 245 <= h * 360 <= 330 and s > 0.15:
                offenders.append(match.group(0))
        assert offenders == []


class TestSquareButtonsAndNoOverRounding:
    def test_radius_tokens_are_small(self):
        for token in re.findall(r"--radius[\w-]*:\s*(\d+)px", CSS):
            assert int(token) <= 8

    def test_no_pill_or_large_rounding(self):
        for value in re.findall(r"border-radius:\s*([^;]+);", CSS):
            for number in re.findall(r"(\d+)px", value):
                assert int(number) <= 8, value
            assert "999" not in value

    def test_buttons_use_the_small_radius(self):
        block = re.search(r"\.btn\s*\{[^}]*\}", CSS).group(0)
        assert "var(--radius-sm)" in block


class TestNoMotion:
    @pytest.mark.parametrize("token", ["@keyframes", "animation", "transition", "scroll-behavior", "backdrop-filter", "@property"])
    def test_no_css_motion(self, token):
        assert token not in CSS

    def test_no_css_transforms(self):
        assert not re.search(r"(?<!text-)transform\s*:", CSS)  # text-transform (capitalisation) is not motion

    def test_no_scripted_motion_or_smooth_scrolling(self):
        # (a short setTimeout is allowed: it waits for the optional history service, it does not animate anything)
        assert "smooth" not in JS and "requestAnimationFrame" not in JS
        assert "animate(" not in JS and "classList.add(\"reveal\")" not in JS


class TestNoEmojiOrEmDashes:
    @pytest.mark.parametrize("name", list(FILES))
    def test_no_emoji_or_pictographs(self, name):
        assert EMOJI.findall(FILES[name]) == []
        assert not re.search(r"&#(1[2-9]\d{4}|\d{5,6});|&#x1[0-9a-f]{4,5};", FILES[name], re.I)

    @pytest.mark.parametrize("name", list(FILES))
    def test_no_em_dashes(self, name):
        text = FILES[name]
        assert "\u2014" not in text and "&mdash;" not in text.lower()

    def test_the_only_em_dash_reference_is_the_display_conversion(self):
        # The questionnaire's own answer labels contain one; the page converts it for display only, building the
        # character from its code point so that no em dash appears in the source.
        assert JS.count("fromCharCode(8212)") == 1
        assert "fromCharCode(8212)" in next(line for line in JS.splitlines() if "function plainOption" in line)

    def test_displayed_answer_label_conversion_keeps_the_sent_value(self):
        assert 'value: option' in JS and "plainOption(option)" in JS

    def test_server_text_shown_to_users_has_no_em_dashes(self):
        texts = list(explanations.ADVICE.values()) + [explanations.MASKED_ADVICE, explanations.SELF_REPORT_HIGHER_ADVICE,
                                                      explanations.LOW_FALLBACK, explanations.NO_SPECIFIC_FALLBACK, api_module.DISCLAIMER]
        texts += list(explanations.SUPPORT.values()) + [label for label, _ in explanations._BASE.values()]
        assert [t for t in texts if "\u2014" in t] == []
        body = client.get("/evaluation").json()
        assert "\u2014" not in str(body)


class TestNoFakeMetricsOrVagueCopy:
    def test_no_illustrative_numbers_or_made_up_claims(self):
        text = visible_text(HTML)
        assert not re.search(r"\d+\s?%", text), "no percentages should be hard-coded in the page"
        for phrase in ("Illustration", "Risk index", "About 5 minutes", "not real data", "10,000", "trusted by", "users"):
            assert phrase.lower() not in text.lower(), phrase

    def test_no_marketing_filler(self):
        text = visible_text(HTML).lower()
        for phrase in ("every angle", "honest picture", "smart fusion", "tailored ideas", "plain-language reasons", "seamless", "powerful",
                       "cutting-edge", "next-gen", "revolution", "unlock", "supercharge", "game-changing", "ai-powered"):
            assert phrase not in text, phrase

    def test_the_intro_states_concrete_facts(self):
        text = visible_text(HTML)
        for fact in ("20-question survey", "five questions about your daily habits", "low, moderate or high", "not a medical test"):
            assert fact in text, fact

    def test_figures_shown_in_the_result_come_from_the_response(self):
        assert "toFixed(2)" in JS and "data.assessment" in JS or "a.risk_score" in JS


PRODUCT_NAMES = re.compile(r"Azure AI (Speech|Language)")  # real Microsoft product names used by the optional cloud features


class TestNoAiTag:
    @pytest.mark.parametrize("name", list(FILES))
    def test_no_made_with_ai_tag(self, name):
        text = PRODUCT_NAMES.sub("Azure", FILES[name])
        assert not re.search(r"(made|built|created|generated|powered|written)\s+(with|by)\b", text, re.I)
        assert not re.search(r"\bAI\b|ChatGPT|Claude|OpenAI|\bGPT\b|\bLLM\b|copilot", text)

    def test_served_page_has_no_ai_tag(self):
        page = PRODUCT_NAMES.sub("Azure", client.get("/app/").text)
        assert not re.search(r"made with|generated by|powered by|\bAI\b", page, re.I)
