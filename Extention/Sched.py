# -*- conding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

class Schedule:
    """
        schedule specify the time to run the strategy to get the next action ,this is used in real trading
        sched enter(delay,priority,func,augment or kwargs) ; enterabs(time,priority ,func,augment or kwargs)
        queue ( accumlate the scheduled task)
        import sched,time

        def crontab():
            pass

        if __name__=='__main__':

            while True:
                s = sched.scheduler(time.time, time.sleep)
                task = s.enter(0,1,crontab,argument = ())
                s.queue.append(task)
                s.run()
    """
    pass