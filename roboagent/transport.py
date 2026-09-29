"""Line-delimited JSON backend over a local command (or an explicit ssh argv)."""
import json
import queue
import subprocess
import threading


class Backend:
    def __init__(self, command, stderr_path, timeout=360):
        if not command or not all(isinstance(x, str) for x in command):
            raise ValueError('Backend command must be an argv list')
        self.timeout = timeout
        self.errors = open(stderr_path, 'w')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.errors, text=True, bufsize=1)
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def receive(self):
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError('Backend response deadline exceeded')
        if line is None:
            raise RuntimeError('Backend exited before a response')
        value = json.loads(line)
        if not isinstance(value, dict) or 'error' in value:
            raise RuntimeError('Invalid backend response: ' + str(value))
        return value

    def request(self, value):
        self.process.stdin.write(json.dumps(value)+'\n')
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join(timeout=2)
        self.process.stdin.close()
        self.process.stdout.close()
        self.errors.close()
