import logging
import csv
import os


class CSVLogger:
    def __init__(self, filename: str = None):
        self.filename = filename
        self.header_written = False
        self.fieldnames = None
        self.logger = logging.getLogger("csv_logger")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        if self.filename is not None:
            assert not os.path.exists(self.filename), (
                f"CSV file '{self.filename}' already exists"
            )

    def log(self, data: dict):
        if self.fieldnames is None:
            self.fieldnames = list(data.keys())
        write_header = not os.path.exists(self.filename) or not self.header_written
        if write_header:
            self.logger.info(",".join(self.fieldnames))
        self.logger.info(",".join(str(data[k]) for k in self.fieldnames))
        if self.filename is None:
            return
        with open(self.filename, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
                self.header_written = True
            writer.writerow(data)
