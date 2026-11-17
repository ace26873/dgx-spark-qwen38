#!/usr/bin/env python3
"""Library's Delete of a whole repository, on the page as it runs (decisions 1402 and 1404 of Hasan, 2026-11-16): a
Delete on each repository's first row and on each other model; its click asks the server's plan first, and opens the
sheet with it (what it frees on which disk, what it keeps and why, the lanes whose next Load downloads it again), its
button "Delete it", or "Delete it anyway" (lose=yes) with what the Hub would not give back; a repository read now gets
no sheet, only why. Same harness as test_page_behaviour.py."""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_page_behaviour import run  # noqa: E402

REG = {"images": [], "caches": ["/c"], "targets": [],
       "models": [{"repo_id": "RadixArk/Qwen3.8-27B-DSpark", "cache": "/c", "managed": True, "disk_bytes": 3_000_000_000,
                   "revisions": [{"rev": "a" * 40, "bytes": 3_000_000_000, "status": "pinned", "pin": "DRAFT_REV"},
                                 {"rev": "b" * 40, "bytes": 1_000_000_000, "status": "stray", "delete": {"bytes": 10**9}}]}],
       "other_models": [{"repo_id": "x/y", "cache": "/c", "disk_bytes": 75_000_000_000}]}
PLAN = {"repo": "RadixArk/Qwen3.8-27B-DSpark", "cache": "/c", "bytes": 3 * 2**30, "mount": "/mnt/ssd", "kept_bytes": 0,
        "kept_why": [], "readers": [], "reloads": ["stock"], "identical": True,
        "revisions": [{"rev": "a" * 40, "identical": True, "same": 3, "rewritten": [], "differs": [], "absent": [],
                       "unverified": []}]}


def click_delete(test, plan, table="reg-models"):
    return run(test, r"""
    __fetch = async (url, init) => url.startsWith('/api/recipes') ? __response(200, {builtin: [], custom: [], custom_dir: ''})
      : url.startsWith('/api/registry/plan') ? __response(%d, %s)
      : url.startsWith('/api/registry') ? __response(200, %s) : new Promise(() => {});
    feed({config: CONFIG, units: UNITS, lifecycle: life({'qwen38-sglang.service': eng('ready')})});
    showView('library'); await __settle(); await __settle();
    const buttons = [...$('%s').querySelectorAll('button')].filter(b => b.textContent === 'Delete');
    buttons[0].click(); await __settle(); await __settle();
    report({buttons: buttons.length, open: !$('scrim').hidden, title: txt('sh-title'), lede: txt('sh-lede'),
            warns: txt('sh-warns'), go: txt('sh-go'), toast: toastText(),
            asked: __fetches.map(f => f.url).filter(u => u.startsWith('/api/registry/plan'))});
    """ % (409 if "refused" in plan else 200, json.dumps(plan), json.dumps(REG), table))


class LibraryDeletesARepository(unittest.TestCase):
    def test_one_delete_per_repository_and_the_plan_in_the_sheet(self):
        r = click_delete(self, PLAN)
        self.assertEqual(r["buttons"], 1, "one Delete on the repository's first row, none on its other revisions")
        self.assertEqual(r["asked"], ["/api/registry/plan?repo=RadixArk%2FQwen3.8-27B-DSpark&cache=%2Fc"])
        self.assertTrue(r["open"], r)
        self.assertEqual((r["title"], r["go"]), ("Delete Qwen3.8-27B-DSpark", "Delete it"))
        self.assertIn("3.0 GiB freed on /mnt/ssd", r["lede"])
        self.assertIn("a download gets it back", r["lede"])
        self.assertIn("downloads it again", r["warns"])

    def test_a_loss_the_hub_would_not_give_back_is_said_and_confirmed_as_such(self):
        plan = dict(PLAN, identical=False, revisions=[dict(PLAN["revisions"][0], identical=False,
                                                           differs=["w.safetensors"])])
        r = click_delete(self, plan)
        self.assertEqual(r["go"], "Delete it anyway")
        self.assertIn("w.safetensors served otherwise", r["warns"])
        self.assertIn("lose it for good", r["warns"])
        self.assertNotIn("a download gets it back", r["lede"])

    def test_what_reads_it_now_gets_no_sheet(self):
        r = click_delete(self, dict(PLAN, readers=["the container gta-vlm"]))
        self.assertFalse(r["open"])
        self.assertIn("read now by the container gta-vlm", r["toast"])

    def test_a_refusal_of_the_server_gets_no_sheet(self):
        r = click_delete(self, {"refused": "a download is writing to it now"})
        self.assertFalse(r["open"])
        self.assertIn("a download is writing to it now", r["toast"])

    def test_other_models_have_their_delete_too(self):
        r = click_delete(self, dict(PLAN, repo="x/y"), table="reg-other")
        self.assertEqual(r["buttons"], 1)
        self.assertEqual(r["asked"], ["/api/registry/plan?repo=x%2Fy&cache=%2Fc"])
        self.assertTrue(r["open"])


IMAGES = [{"ref": "eugr/spark-vllm-b12x@sha256:" + "03" * 32, "size": "18.2GB", "id": "a7f76ae9f18c", "engine": True},
          {"ref": "lmsysorg/sglang@sha256:" + "e3" * 32, "size": "36.5GB", "id": "fd8439732b16", "engine": True},
          {"ref": "lmsysorg/sglang:v0.5.22", "size": "36.5GB", "id": "fd8439732b16", "engine": True}]
IPLAN = {"image": "a7f76ae9f18c", "names": ["qwen38-pinned:flash-local-03a956544e", IMAGES[0]["ref"]],
         "bytes": 17_770_000_000, "containers": [], "units": [], "reloads": ["qwen38-flash-local.service"],
         "registry": [{"name": IMAGES[0]["ref"], "served": True}], "obtainable": True, "mount": "/"}


def click_image(test, plan, row=0):
    reg = dict(REG, images=IMAGES)
    return run(test, r"""
    __fetch = async (url, init) => url.startsWith('/api/recipes') ? __response(200, {builtin: [], custom: [], custom_dir: ''})
      : url.startsWith('/api/registry/image-plan') ? __response(%d, %s)
      : url.startsWith('/api/registry') ? __response(200, %s) : new Promise(() => {});
    feed({config: CONFIG, units: UNITS, lifecycle: life({'qwen38-sglang.service': eng('ready')})});
    showView('library'); await __settle(); await __settle();
    const rows = [...$('reg-images').tBodies[0].rows].map(r => r.querySelectorAll('button').length);
    $('reg-images').tBodies[0].rows[%d].querySelector('button').click(); await __settle(); await __settle();
    report({rows, open: !$('scrim').hidden, title: txt('sh-title'), lede: txt('sh-lede'), warns: txt('sh-warns'),
            go: txt('sh-go'), toast: toastText(),
            asked: __fetches.map(f => f.url).filter(u => u.startsWith('/api/registry/image-plan'))});
    """ % (409 if "refused" in plan else 200, json.dumps(plan), json.dumps(reg), row))


class LibraryDeletesAnEngineImage(unittest.TestCase):
    def test_one_delete_per_image_and_its_plan_in_the_sheet(self):
        r = click_image(self, IPLAN)
        self.assertEqual(r["rows"], [1, 1, 0], "one Delete per image, on the first of its names")
        self.assertEqual(r["asked"], ["/api/registry/image-plan?id=a7f76ae9f18c"])
        self.assertTrue(r["open"], r)
        self.assertEqual((r["title"], r["go"]), ("Delete the image a7f76ae9f18c", "Delete it"))
        self.assertIn("16.5 GiB freed on /", r["lede"])
        self.assertIn("a pull gets it back", r["lede"])
        self.assertIn("pulls or builds it again", r["warns"])

    def test_a_container_or_a_serving_lane_gets_no_sheet(self):
        r = click_image(self, dict(IPLAN, containers=["gta-vlm (running)"]))
        self.assertFalse(r["open"])
        self.assertIn("the container gta-vlm (running)", r["toast"])

    def test_an_image_no_registry_serves_is_deleted_anyway_only_said_so(self):
        r = click_image(self, dict(IPLAN, obtainable=False, registry=[{"name": IMAGES[0]["ref"], "served": False}]))
        self.assertEqual(r["go"], "Delete it anyway")
        self.assertIn("lose it for good", r["warns"])



class LibrarySaysEachDisk(unittest.TestCase):
    def test_each_disk_its_free_space_and_what_the_stack_holds_there(self):
        GiB = 2**30
        reg = dict(REG, disks=[{"mount": "/", "total": 915 * GiB, "free": 354 * GiB, "models": 62 * GiB,
                                "docker": 110 * GiB, "runtimes": 54 * GiB},
                               {"mount": "/mnt/ssd", "total": 1877 * GiB, "free": 1181 * GiB, "models": 303 * GiB}],
                   runtimes=[{"item": "image-h3", "bytes": 16 * GiB, "mount": "/", "paths": ["/h/venv"]}])
        r = run(self, r"""
        __fetch = async (url, init) => url.startsWith('/api/recipes') ? __response(200, {builtin: [], custom: [], custom_dir: ''})
          : url.startsWith('/api/registry') ? __response(200, %s) : new Promise(() => {});
        feed({config: CONFIG, units: UNITS, lifecycle: life({'qwen38-sglang.service': eng('ready')})});
        showView('library'); await __settle(); await __settle();
        report({disks: txt('reg-disks'), runtimes: [...$('reg-runtimes').tBodies[0].rows].map(r => [...r.cells].map(c => c.textContent))});
        """ % json.dumps(reg))
        self.assertIn("/: 354 GiB free of 915 GiB (models 62.0 GiB, Docker's images (every project's) 110.0 GiB, "
                      "runtimes and caches 54.0 GiB)", r["disks"])
        self.assertIn("/mnt/ssd: 1181 GiB free of 1877 GiB (models 303.0 GiB)", r["disks"])
        self.assertEqual(r["runtimes"], [["MiniMax-H3-Image runtime on /", "16.0 GiB", "Delete"]])


class LibraryShowsWhatAnotherProjectKeeps(unittest.TestCase):
    def test_a_kept_repository_and_image_say_who_keeps_them_and_offer_no_delete(self):
        reg = dict(REG, models=[dict(REG["models"][0], keep="GTA, brouillon de gta-vlm")],
                   other_models=[dict(REG["other_models"][0], keep=None)],
                   images=[dict(IMAGES[1], keep="GTA, image de gta-vlm"), dict(IMAGES[0], keep=None)])
        r = run(self, r"""
        __fetch = async (url, init) => url.startsWith('/api/recipes') ? __response(200, {builtin: [], custom: [], custom_dir: ''})
          : url.startsWith('/api/registry') ? __response(200, %s) : new Promise(() => {});
        feed({config: CONFIG, units: UNITS, lifecycle: life({'qwen38-sglang.service': eng('ready')})});
        showView('library'); await __settle(); await __settle();
        const rows = id => [...$(id).tBodies[0].rows].map(r => ({text: r.textContent, buttons: r.querySelectorAll('button').length}));
        report({models: rows('reg-models'), other: rows('reg-other'), images: rows('reg-images')});
        """ % json.dumps(reg))
        self.assertIn("kept for GTA, brouillon de gta-vlm", r["models"][0]["text"])
        self.assertEqual([x["buttons"] for x in r["models"]], [0, 0])
        self.assertEqual(r["other"][0]["buttons"], 1)
        self.assertIn("kept for GTA, image de gta-vlm", r["images"][0]["text"])
        self.assertEqual([x["buttons"] for x in r["images"]], [0, 1])

if __name__ == "__main__":
    unittest.main()
