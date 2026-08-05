from go2_deploy.utility.thread import RecurrentThread


class KeyboardHandler:
    def __init__(self, controller):
        self.controller = controller
        self.keyboard_thread = RecurrentThread(
            interval=0.005, target=self._process_input
        )
        self.keyboard_thread.Start()

    def _process_input(self):
        key = input()
        if key == "":
            self.controller.emergency_stop()
        elif key == "r":
            self.controller.switch_to_recovery()
        elif key == "c":
            self.controller.switch_to_custom_controller()
        elif key == "d":
            self.controller.switch_to_default_controller
        else:
            print("Invalid keyboard input!")
