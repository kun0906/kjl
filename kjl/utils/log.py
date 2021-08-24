class MyLog:
    def __init__(self, level='info'):
        """ Need to be modified

        Parameters
        ----------
        level
        """
        self.level = level

    def debug(self, s):
        # print(s, flush=True)
        pass    # don't print s

    def info(self, s):
        print(s, flush=True)

    def warning(self, s):
        print(s, flush=True)

    def error(self, s):
        print(s, flush=True)

    def critical(self, s):
        print(s, flush=True)


def get_log(level='info'):
    # # lg = ''
    # # if level == 'debug':
    # #     return lg.debug
    # # elif level =='info':
    # #     return lg.info
    # # elif level =='warning':
    # #     return lg.warning
    # # else:   # default
    # #     return lg.info
    #
    # def lg(s, level= level):
    #     print(s, flush=True)
    #
    # return lg

    return MyLog(level=level)
