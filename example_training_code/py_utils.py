class RunningAverageLogger():
    def __init__(self, initial_value=0):
        self.initial_value = initial_value
        self.reset()
    
    def reset(self):
        self._count = 0
        self._running_sum = self.initial_value
    
    def get_avg(self):
        if self._count == 0:
            return self._running_sum
        return float(self._running_sum) / float(self._count)

    def add_value(self, value):
        self._running_sum += float(value)
        self._count += 1
    
    def count(self):
        return self._count
    
class InplacePrinter():
    CURSOR_UP = '\033[F'
    CLEAR_TILL_LINE_END = '\033[K'
    def __init__(self, max_lines, auto_clear=True):
        self.max_lines = max_lines
        self.auto_clear = True
        self._curr_num_lines = 0
    def reset(self):
        self._curr_num_lines = 0
    def clear(self):
        for i in range(self._curr_num_lines):
            print(self.CURSOR_UP+'\r'+self.CLEAR_TILL_LINE_END, end="", flush=True)
        self._curr_num_lines = 0
    
    def print(self, str:str):
        if self.max_lines != -1 and self._curr_num_lines == self.max_lines:
            if self.auto_clear:
                self.clear()
            else:
                return

        print(str, flush=True)
        self._curr_num_lines += 1