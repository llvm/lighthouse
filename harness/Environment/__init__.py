# -*- coding: utf-8 -*-
"""
    Target Information
"""

import shlex
import os
import re
import shutil

import Logger
import Execute

class CPUInfo(object):
    def __init__(self, logger):
        self.logger = logger.clone("cpuinfo")

        with Execute(self.logger).run(["uname", "-m"]) as r:
            self.arch = r.stdout.strip()

        with open("/proc/cpuinfo", "r") as f:
            self.cpuinfo = f.read()

        self.flags = None
        for keyword in [ "flags", "Features" ]:
            with re.match(keyword + r"\s*:\s+(.*)", self.cpuinfo, re.MULTILINE) as match:
                if match:
                    self.flags = shlex.split(match.group(1))
                    self.logger.debug(f"CPU flags: {self.flags}")
                    return

        self.cpu = None
        for keyword in [ "model name", "Processor" ]:
            with re.match(keyword + r"\s*:\s+(.*)", self.cpuinfo, re.MULTILINE) as match:
                if match:
                    self.cpu = shlex.split(match.group(1))
                    self.logger.debug(f"CPU: {self.cpu}")
                    return

    def hasFlag(self, flag):
        if not self.flags:
            return False
        for ext in self.flags:
            if re.fullmatch(flag, ext):
                return True
        return False

    def matchCPU(self, name):
        if not self.cpu:
            return False
        for ext in self.cpu:
            if re.fullmatch(name, ext):
                return True
        return False

    def getArch(self): 
        return self.arch


class Environment(object):
    def __init__(self, args, logger):
        self.logger = logger.clone("environment")
        self.base_dir = os.path.realpath(os.path.dirname(__file__))
        self.root_dir = self.findGitRoot(self.base_dir)
        self.build_dir = args.build
        if not self.build_dir:
            self.build_dir = self.root_dir

        self.programs = {}
        for prog in [ "mlir-opt", "mlir-runner" ]:
            self.programs[prog] = self.findProgram(prog, self.build_dir)
        for _, path in self.programs.items():
            if os.path.exists(path):
                self.bin_dir = os.path.realpath(os.path.dirname(path))
                parent = os.path.join(self.bin_dir, os.path.pardir)
                self.build_dir = os.path.realpath(parent)
                self.lib_dir = os.path.join(self.build_dir, "lib")
                break
        assert self.build_dir != self.root_dir
        self.bench_dir = os.path.join(self.root_dir, "benchmarks")
        self.harness = os.path.join(self.bench_dir, "harness", "controller.py")
        self.test_dir = os.path.join(self.bench_dir, "mlir")

        # Pass arguments down to benchmarks, if known
        self.extra_args = list()
        if args.verbose > 0:
            for v in range(args.verbose - args.quiet):
                self.extra_args.append("-v")

        # Set environment variables for dynamic loading (Linux and Mac)
        for path in ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"]:
            environ = [os.getenv(path)] if os.getenv(path) else []
            environ.insert(0, self.lib_dir)  # prepend
            os.environ[path] = ":".join(environ)

        # Check if taskset is available
        # (every CPU we care about has at least 4 cores)
        self.cpu_pinning = None
        taskset = shutil.which("taskset")
        if taskset:
            self.cpu_pinning = [taskset, "-c", "3"]

    def findGitRoot(self, path):
        """Find the git root directory, if any, or return the input"""
        temp = path
        while temp:
            if os.path.exists(os.path.join(temp, ".git")):
                return temp
            temp = os.path.abspath(os.path.join(temp, os.pardir))
        return path

    def findProgram(self, program, baseDir):
        """Find a program in the specified base directory"""
        for root, dirs, files in os.walk(baseDir, followlinks=True):
            if program in files:
                full_path = os.path.join(root, program)
                self.logger.debug(f"{program}: {full_path}")
                break
        return full_path


    def pin_task(self, command):
        """Adds taskset if not forced through other means"""
        if not self.cpu_pinning:
            return
        if "KMP_AFFINITY" in os.environ:
            return
        command.extend(self.cpu_pinning)
