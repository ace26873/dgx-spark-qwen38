"""Each version's name in the cockpit is its official one (decisions 1180 and 1185 of Hasan, 2026-11-07): the model,
its precision and who publishes it, read off the repository its installer pins, "abliterated" where those builds'
makers say so, and never "stock", "uncensored" nor "Flash 176B". A pin that moves to another publisher, another
precision or another model fails here until the name follows it. The targets' ids, which commands take, do not change."""
import pathlib
import re
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
BASE = (REPO / "dashboard/static/js/base.js").read_text()

# target -> (installer, variable): the repository each version loads
PINS = {"stock": ("install.sh", "STOCK_REPO"), "uncensored": ("install.sh", "UNC_REPO"),
        "fp8": ("install.sh", "FP8_REPO"), "uncensored-fp8": ("install.sh", "UNCFP8_REPO"),
        "arete": ("install.sh", "ARETE_REPO"),
        "daisyrose": ("install.sh", "DAISYROSE_REPO"),
        "flash": ("install.sh", "FLASH_REPO"), "flash-uncensored": ("install.sh", "FLASH_UNC_REPO"),
        "flash-nvda": ("install.sh", "FLASH_NVDA_REPO"), "flash-local": ("install-flash-local.sh", "FLASH_LOCAL_REPO"),
        "kolibri": ("install-kolibri.sh", "KOLIBRI_FP8_REPO"), "kolibri-nvfp4": ("install-kolibri.sh", "KOLIBRI_NVFP4_REPO"),
        "nemotron": ("install-nemotron.sh", "NEMOTRON_BOLT_REPO"),
        "nemotron-super": ("install-nemotron-super.sh", "NEMOTRON_SUPER_REPO"),
        "ornith-122b": ("install-ornith-122b.sh", "ORNITH_122B_REPO"),
        "flash-unsloth": ("install-flash-unsloth.sh", "FLASH_UNSLOTH_REPO"),
        "flash-unsloth-iq3": ("install-flash-unsloth.sh", "FLASH_UNSLOTH_REPO"),
        "image": ("install-image.sh", "IMAGE_MODEL_PIN"), "image-turbo": ("install-image.sh", "IMAGE_TURBO_PIN"),
        "video": ("install-video.sh", "VIDEO_MODEL_PIN"), "image-h3": ("install-image-h3.sh", "IMAGE_H3_MODEL_PIN"),
        "transcribe": ("install-transcribe.sh", "TRANSCRIBE_REPO"), "lfm25": ("install-lfm25.sh", "LFM25_REPO"),
        "lfm3": ("install-lfm3.sh", "LFM3_REPO"), "inkling": ("install-inkling.sh", "INKLING_REPO")}
# a repository's owner on Hugging Face, written as its own pages write its name
PUBLISHER = {"RadixArk": "RadixArk", "edp1096": "edp1096", "Qwen": "Qwen", "nvidia": "NVIDIA", "dealignai": "dealignai",
             "Aleph-Alpha": "Aleph Alpha", "Quantinine": "Quantinine", "ornith-ai": "Ornith AI", "MiniMaxAI": "MiniMax",
             "unsloth": "Unsloth", "atakans": "atakans", "mradermacher": "mradermacher", "LiquidAI": "Liquid AI",
             "thinkingmachines": "Thinking Machines"}
# a precision the repository's name does not say: Aleph Alpha's own release of Kolibri-1 is its FP8 build, and
# Unsloth's GGUF repository holds every quantization of the model, the lane serving its UD-Q4_K_XL folder or its
# UD-IQ3_XXS one, each read off the folder of the part its installer pins first (PRECISION below); mradermacher's GGUF
# repository of NVIDIA's speech recognizer holds every quantization too, the lane serving its Q8_0 (v1.51)
UNSAID = {"kolibri": {"fp8"}, "flash-unsloth": {"ud", "q4", "k", "xl"}, "flash-unsloth-iq3": {"ud", "iq3", "xxs"},
          "transcribe": {"q8", "0"}}
PRECISION = {"flash-unsloth": "FLASH_UNSLOTH_PART1", "flash-unsloth-iq3": "FLASH_UNSLOTH_IQ3_PART1"}


def table(name):
    """A table of base.js keyed by target: {target: text}."""
    body = re.search(rf"const {name} = \{{(.*?)\}};", BASE, re.S).group(1)
    return {k.strip("'"): a or b for k, a, b in re.findall(r"""('[\w-]+'|\w+): (?:'([^']*)'|"([^"]*)")""", body)}


def pinned(script, var):
    m = re.search(rf'(?m)^{var}="([^"$]+)"', (REPO / script).read_text())
    return m.group(1) if m else None


def words(text):
    return re.sub(r"[-_/]", " ", text.lower()).split()


class EachVersionHasItsOfficialName(unittest.TestCase):
    def setUp(self):
        self.name, self.short = table("TARGET_NAME"), table("TARGET_SHORT_NAME")

    def test_every_version_has_one(self):
        self.assertEqual(set(self.name), set(PINS))

    def test_the_name_is_read_off_the_pinned_repository(self):
        for target, (script, var) in PINS.items():
            with self.subTest(target=target):
                repo = pinned(script, var)
                self.assertIsNotNone(repo, f"{script} pins no {var}")
                owner, model = repo.split("/")
                name = self.name[target]
                # who publishes it, last
                self.assertTrue(name.endswith(" · " + PUBLISHER[owner]), (name, repo))
                # each word before it is one the repository's name says, or a precision it leaves unsaid
                said = " ".join(words(model))
                for w in words(name.rsplit(" · ", 1)[0]):
                    self.assertTrue(w in said or w in UNSAID.get(target, ()), (target, w, repo))
                self.assertEqual("abliterated" in model.lower(), "abliterated" in name, (name, repo))

    def test_a_versions_button_says_its_precision_and_its_publisher_where_its_lane_has_several(self):
        for target in ("stock", "uncensored", "fp8", "uncensored-fp8", "arete", "daisyrose", "flash", "flash-uncensored",
                       "flash-nvda", "kolibri", "kolibri-nvfp4", "flash-unsloth", "flash-unsloth-iq3"):
            with self.subTest(target=target):
                pub = self.name[target].rsplit(" · ", 1)[1]
                self.assertTrue(self.short[target].endswith(" · " + pub), self.short[target])
                # its words in the order of the version's name: "NVFP4 abliterated" read the name's
                # "abliterated NVFP4" backwards (code review of v1.45.0)
                said, name = words(self.short[target].rsplit(" · ", 1)[0]), words(self.name[target].rsplit(" · ", 1)[0])
                at = [name.index(w) for w in said]
                self.assertEqual(at, sorted(at), (self.short[target], self.name[target]))

    def test_a_gguf_versions_precision_is_the_folder_its_parts_are_pinned_in(self):
        # Unsloth's repository holds every quantization of the model: the precision a version's name and its button say
        # is the folder of the first part its installer pins, two versions of one repository since v1.49
        text = (REPO / "install-flash-unsloth.sh").read_text()
        for target, var in PRECISION.items():
            with self.subTest(target=target):
                folder = re.search(rf'^{var}="([^/"]+)/', text, re.M).group(1)
                self.assertEqual(set(words(folder)), UNSAID[target])
                self.assertIn(f" {folder} ", self.name[target])
                self.assertTrue(self.short[target].startswith(folder + " · "), self.short[target])

    def test_the_nvidia_nemotron_transcribe_lanes_precision_is_the_file_its_installer_pins(self):
        # its repository holds every quantization: the name says the one the lane loads, read off its pinned file
        text = (REPO / "install-transcribe.sh").read_text()
        file = re.search(r'^TRANSCRIBE_FILE="([^"=]+)=', text, re.M).group(1)
        quant = re.search(r"\.(Q\d+_\d+)\.gguf$", file).group(1)
        self.assertEqual(set(words(quant)), UNSAID["transcribe"])
        self.assertIn(f" {quant} ", self.name["transcribe"])
        self.assertEqual(self.short["transcribe"], quant)

    def test_no_name_says_stock_uncensored_or_flash_176b(self):
        shown = list(self.name.values()) + list(self.short.values()) + list(table("TARGET_LINE").values())
        shown += re.findall(r"\[\w+\]: '([^']*)'", re.search(r"const LANE_NAME = \{(.*?)\};", BASE, re.S).group(1))
        shown += re.findall(r"title: '([^']*)'", BASE)
        for text in shown:
            with self.subTest(text=text):
                self.assertNotRegex(text.lower(), r"\bstock\b|uncensored|176b")


    def test_no_text_of_the_page_names_a_lane_the_old_way(self):
        # the Decide view's button read "Load flash 176B" and "Load 27B stock", outside the tables (found while
        # renaming, 2026-11-08): every text of the page, the targets' ids apart ('uncensored', 'flash-uncensored')
        static = REPO / "dashboard/static"
        for f in sorted(list((static / "js").glob("*.js")) + [static / "index.html"]):
            with self.subTest(file=f.name):
                text = f.read_text()
                self.assertNotRegex(text, r"(?i)flash 176b|27b stock|[a-z0-9] uncensored\b|>stock<|flash fp4-local")

    def test_no_text_the_cockpit_sends_names_a_lane_the_old_way(self):
        # its own sentences, in its strings: "the Flash FP4-Local lane has no cache flush route" (code review of
        # v1.45.0); comments may still say how a lane was named once
        text = (REPO / "dashboard/cockpit.py").read_text()
        said = re.findall(r"""["'][^"'\n]*(?i:flash fp4-local|flash 176b|27b stock)[^"'\n]*["']""", text)
        self.assertEqual(said, [])

if __name__ == "__main__":
    unittest.main()
