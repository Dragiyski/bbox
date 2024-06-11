import sys

class NullLogger:
    def log(self, message):
        pass

class StderrLogger:
    def log(self, message):
        print(message, file=sys.stderr)