import threading
import time
import logging

class LogInterceptor(logging.Handler):
    def __init__(self, api_manager):
        super().__init__()
        self.api_manager = api_manager

    def emit(self, record):
        log_entry = self.format(record)
        if 'RECEIVED' in log_entry:
            self.api_manager.last_received_time = time.time()

class APICallManager:
    def __init__(self, func, *args, timeout=15, retry_attempts=3, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.attempts = 0
        self.result = None
        self.last_received_time = None
        self.running = True

        # Setup the logging
        self.logger = logging.getLogger('nnsight_remote')
        self.logger.setLevel(logging.INFO)
        self.interceptor = LogInterceptor(self)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.interceptor.setFormatter(formatter)
        self.logger.addHandler(self.interceptor)

    def __call__(self):
        self.running = True
        self.result = None
        self.attempts = 0
        self.last_received_time = None

        timeout_checker = threading.Thread(target=self.check_timeout)
        timeout_checker.start()
        self.API_call()  # Initially trigger the API call

        while self.running:
            time.sleep(0.5)  # Main thread can wait and check for status updates

        timeout_checker.join()
        return self.result

    def check_timeout(self):
        while self.running:
            if self.last_received_time and (time.time() - self.last_received_time > self.timeout):
                if self.attempts < self.retry_attempts:
                    print("Timeout since last RECEIVED status. Taking action.")
                    self.handle_timeout()
                self.last_received_time = None
            time.sleep(1)

    def handle_timeout(self):
        print(f"Attempt {self.attempts + 1} failed; retrying...")
        self.attempts += 1
        self.API_call()

    def API_call(self):
        # Here you would call the actual API function
        self.result = self.func(*self.args, **self.kwargs)
        if self.attempts == 0:  # Assuming successful initial call
            self.running = False