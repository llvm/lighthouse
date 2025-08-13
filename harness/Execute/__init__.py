# -*- coding: utf-8 -*-
"""
 Wrapper around `subprocess`.

 Execute Commands, return out/err, log command and error.

 Usage:
  result = Execute(logger).run(['myapp', '-flag', 'etc'])
"""

import logging
import subprocess

from Logger import Logger

class Execute(object):
    """Executes commands, returns out/err"""

    def __init__(self, logger):
        self.logger = logger.clone("execute")

    def run(self, program, input=""):
        """Execute Commands, return out/err"""

        if not program:
            raise ValueError("Need program arguments to execute")
        if not isinstance(program, list):
            raise TypeError("Program needs to be a list of arguments")

        self.logger.debug(f"Executing: {' '.join(program)}")

        # Call the program, capturing stdout/stderr
        result = subprocess.run(
            program,
            input=input if input else None,
            capture_output=True,
            encoding="utf-8",
        )

        # Collect stdout, stderr as UTF-8 strings
        result.stdout = str(result.stdout)
        result.stderr = str(result.stderr)

        if result.returncode != 0:
            self.logger.debug(f"Error: {result.stderr.strip()}")

        # Return
        return result
