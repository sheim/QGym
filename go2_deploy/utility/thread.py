import sys
import os
import errno
import ctypes

# import struct
import threading

from unitree_sdk2py.utils.future import Future
from unitree_sdk2py.utils.timerfd import itimerspec, timerfd_create, timerfd_settime


class Thread(Future):
    def __init__(self, target=None, name=None, args=(), kwargs=None):
        super().__init__()
        self.__target = target
        self.__args = args
        self.__kwargs = {} if kwargs is None else kwargs
        self.__thread = threading.Thread(
            target=self.__ThreadFunc, name=name, daemon=True
        )

    def Start(self):
        return self.__thread.start()

    def IsAlive(self):
        return self.__thread.is_alive()

    def GetId(self):
        """NOTE: not a run discriminator -- Linux recycles the tid of an exited
        thread, so a renewed Thread often reports the same ident."""
        return self.__thread.ident

    def GetNativeId(self):
        return self.__thread.native_id

    def __ThreadFunc(self):
        value = None
        try:
            value = self.__target(*self.__args, **self.__kwargs)
            self.Ready(value)
        except:
            info = sys.exc_info()
            self.Fail(
                "[Thread] target func raise exception:"
                + f"name={info[0].__name__}, args={str(info[1].args)}"
            )


class RecurrentThread(Thread):
    def __init__(
        self, interval: float = 1.0, target=None, name=None, args=(), kwargs=None
    ):
        self.__quit = False
        self.__inter = interval
        self.__loopTarget = target
        self.__loopArgs = args
        self.__loopKwargs = {} if kwargs is None else kwargs
        self.__lock = threading.Lock()

        super().__init__(target=self.__SelectLoopFunc(), name=name)

    def __SelectLoopFunc(self):
        if self.__inter is None or self.__inter <= 0.0:
            return self.__LoopFunc_0
        return self.__LoopFunc

    def Wait(self, timeout: float = None):
        with self.__lock:
            self.__quit = True
        # Propagate the result: False means the wait timed out and the loop may
        # still be running, which is exactly what Restart() needs to know.
        return super().Wait(timeout)

    def __LoopFunc(self):
        # clock type CLOCK_MONOTONIC = 1
        tfd = timerfd_create(1, 0)
        # finally (no except): the fd must be released even when the read below
        # raises, otherwise every crash-and-restart cycle leaks a timer fd.
        try:
            spec = itimerspec.from_seconds(self.__inter, self.__inter)
            timerfd_settime(tfd, 0, ctypes.byref(spec), None)

            while not self.__quit:
                try:
                    self.__loopTarget(*self.__loopArgs, **self.__loopKwargs)
                except:
                    info = sys.exc_info()
                    print(
                        "[RecurrentThread] target func raise exception:"
                        + f"name={info[0].__name__}, args={str(info[1].args)}"
                    )

                try:
                    os.read(tfd, 8)  # buf = ...
                    # print(struct.unpack("Q", buf)[0])
                except OSError as e:
                    if e.errno != errno.EAGAIN:
                        raise e
        finally:
            os.close(tfd)

    def __LoopFunc_0(self):
        while not self.__quit:
            try:
                self.__loopTarget(*self.__loopArgs, **self.__loopKwargs)
            except:
                info = sys.exc_info()
                print(
                    "[RecurrentThread] target func raise exception:"
                    + f"name={info[0].__name__}, args={str(info[1].args)}"
                )
