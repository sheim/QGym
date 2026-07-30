class RLController:
    def __init__(self):
        pass

    def initialize_actor(self):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError
