class bbox:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

    def set_values(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_values(self):
        return [self.x, self.y, self.w, self.h]