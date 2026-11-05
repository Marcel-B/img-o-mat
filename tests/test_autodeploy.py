"""deploy/autodeploy.sh against a throwaway origin: it rolls out each new main once, waits while the service paints,
skips a checkout with local changes and does not retry a commit whose tests fail. The deploy itself is a stand-in
that only notes the commit; to-proxmox.sh needs the real Proxmox host.

    python3 -m unittest discover tests
"""

import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "deploy", "autodeploy.sh")

GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
}


class FakeService(http.server.BaseHTTPRequestHandler):
    status = {"painter": {"running": None, "queue": []}, "downloads": {}}

    def do_GET(self):
        body = json.dumps(FakeService.status).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class AutodeployTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeService)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeService.status = {"painter": {"running": None, "queue": []}, "downloads": {}}
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.origin = os.path.join(self.tmp, "origin.git")
        self.upstream = os.path.join(self.tmp, "upstream")
        self.checkout = os.path.join(self.tmp, "checkout")
        self.deployed = os.path.join(self.tmp, "deployed.txt")
        self.git(self.tmp, "init", "--quiet", "--bare", "-b", "main", self.origin)
        self.git(self.tmp, "clone", "--quiet", self.origin, self.upstream)
        self.git(self.upstream, "checkout", "--quiet", "-b", "main")
        os.makedirs(os.path.join(self.upstream, "deploy"))
        os.makedirs(os.path.join(self.upstream, "tests"))
        shutil.copy(SCRIPT, os.path.join(self.upstream, "deploy", "autodeploy.sh"))
        self.write_test(passes=True)
        self.commit("Anfang")
        self.git(self.tmp, "clone", "--quiet", self.origin, self.checkout)

        fake_deploy = os.path.join(self.tmp, "fake-deploy.sh")
        with open(fake_deploy, "w") as f:
            f.write('#!/bin/sh\necho "$(git rev-parse HEAD) PROXMOX=${PROXMOX:-}" >> "%s"\n' % self.deployed)
        os.chmod(fake_deploy, 0o755)
        self.settings = os.path.join(self.tmp, "autodeploy.env")
        self.env = dict(os.environ, **GIT_ENV,
                        AUTODEPLOY_STATE=os.path.join(self.tmp, "state"),
                        AUTODEPLOY_LOG=os.path.join(self.tmp, "autodeploy.log"),
                        AUTODEPLOY_ENV=self.settings,
                        AUTODEPLOY_DEPLOY=fake_deploy,
                        AUTODEPLOY_PYTHON=sys.executable,
                        IMG_O_MAT_URL="http://127.0.0.1:%d" % self.server.server_address[1])

    def git(self, cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, env=dict(os.environ, **GIT_ENV), check=True,
                              capture_output=True, text=True).stdout.strip()

    def write_test(self, passes):
        with open(os.path.join(self.upstream, "tests", "test_x.py"), "w") as f:
            f.write("import unittest\n\nclass X(unittest.TestCase):\n    def test(self):\n"
                    "        self.assertTrue(%s)\n" % passes)

    def commit(self, message):
        self.git(self.upstream, "add", "-A")
        self.git(self.upstream, "commit", "--quiet", "--allow-empty", "-m", message)
        self.git(self.upstream, "push", "--quiet", "origin", "main")
        return self.git(self.upstream, "rev-parse", "HEAD")

    def round(self):
        script = os.path.join(self.checkout, "deploy", "autodeploy.sh")
        return subprocess.run(["sh", script], cwd=self.tmp, env=self.env, capture_output=True, text=True)

    def rollouts(self):
        if not os.path.exists(self.deployed):
            return []
        with open(self.deployed) as f:
            return [line.split()[0] for line in f.read().splitlines()]

    def log(self):
        with open(self.env["AUTODEPLOY_LOG"]) as f:
            return f.read()

    def test_rolls_out_each_new_main_once(self):
        first = self.git(self.upstream, "rev-parse", "HEAD")
        self.assertEqual(self.round().returncode, 0)
        self.assertEqual(self.rollouts(), [first])
        self.round()
        self.assertEqual(self.rollouts(), [first])

        second = self.commit("Neu")
        self.assertEqual(self.round().returncode, 0, self.log())
        self.assertEqual(self.rollouts(), [first, second])
        self.assertEqual(self.git(self.checkout, "rev-parse", "HEAD"), second)
        self.assertIn("ist ausgerollt", self.log())

    def test_settings_reach_the_deploy(self):
        with open(self.settings, "w") as f:
            f.write("PROXMOX=pve\n")
        self.round()
        with open(self.deployed) as f:
            self.assertIn("PROXMOX=pve", f.read())

    def test_waits_while_painting_or_downloading(self):
        FakeService.status["painter"]["running"] = {"app": "tonwerk-ai", "model": "flux2-klein-4b"}
        self.round()
        self.assertEqual(self.rollouts(), [])
        self.assertIn("wartet", self.log())

        FakeService.status = {"painter": {"running": None, "queue": []},
                              "downloads": {"krea2-turbo": {"state": "running"}}}
        self.round()
        self.assertEqual(self.rollouts(), [])

        FakeService.status["downloads"]["krea2-turbo"]["state"] = "done"
        self.round()
        self.assertEqual(len(self.rollouts()), 1)

    def test_deploys_when_the_service_does_not_answer(self):
        self.env["IMG_O_MAT_URL"] = "http://127.0.0.1:9"
        self.round()
        self.assertEqual(len(self.rollouts()), 1)

    def test_local_changes_stop_it(self):
        with open(os.path.join(self.checkout, "tests", "test_x.py"), "a") as f:
            f.write("# local\n")
        self.round()
        self.assertEqual(self.rollouts(), [])
        self.assertIn("Lokale Änderungen", self.log())

    def test_failing_tests_are_not_retried(self):
        self.round()
        self.write_test(passes=False)
        broken = self.commit("Kaputt")
        self.assertNotEqual(self.round().returncode, 0)
        self.assertEqual(len(self.rollouts()), 1)
        self.assertIn("Tests scheitern", self.log())
        with open(os.path.join(self.env["AUTODEPLOY_STATE"], "failed-tests")) as f:
            self.assertEqual(f.read().strip(), broken)
        self.assertEqual(self.round().returncode, 0)

        self.write_test(passes=True)
        fixed = self.commit("Repariert")
        self.round()
        self.assertEqual(self.rollouts()[-1], fixed)

    def test_failed_deploy_is_retried(self):
        with open(self.env["AUTODEPLOY_DEPLOY"], "w") as f:
            f.write("#!/bin/sh\nexit 1\n")
        self.assertNotEqual(self.round().returncode, 0)
        self.assertNotEqual(self.round().returncode, 0)
        self.assertEqual(self.log().count("Ausrollen scheiterte"), 2)


if __name__ == "__main__":
    unittest.main()
