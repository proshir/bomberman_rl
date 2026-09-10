import os
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from time import time

from main import main


class MainTestCase(unittest.TestCase):
    def test_play(self):
        start_time = time()
        main(["play", "--n-rounds", "1", "--no-gui"])
        # Assert that log exists
        self.assertTrue(os.path.isfile("logs/game.log"))
        # Assert that game log way actually written
        self.assertGreater(os.path.getmtime("logs/game.log"), start_time)

    def test_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'benchmark'
            result = subprocess.run([
                sys.executable, 'run_benchmark.py',
                '--agents', 'peaceful_agent', 'coin_collector_agent',
                '--seeds', '10', '--seats', '0', '--max-steps', '20',
                '--output', str(output),
            ])
            self.assertEqual(result.returncode, 0)
            with open(output / 'summary.json') as file:
                summary = json.load(file)
            self.assertEqual(summary['metric'], 'coins')
            self.assertEqual(len(summary['agents']), 2)


if __name__ == '__main__':
    unittest.main()
