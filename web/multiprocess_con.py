# -*- coding: utf-8  -*-
#进程通讯
import multiprocessing

def consumer(input_q):
    while True:
        item = input_q.get()
        print(item)
        # task_done发出信号，表示get返回的项已经被处理 ,防止返回queue.empty错误
        input_q.task_done()


def producer(sequence, out_put_q):
    for item in sequence:
        # 将item放入队列，如果队列已经满，阻塞至有空间用为止
        out_put_q.put(item)

if __name__ == '__main__':
    # 创建共享进程列队，queue对象
    q = multiprocessing.JoinableQueue()
    cons_p = multiprocessing.Process(target=consumer, args=(q,))
    cons_p.daemon = True
    cons_p.start()

    sequence = [1, 2, 3, 4]
    producer(sequence, q)
    # 生产进行阻塞，直到队列所有项均被处理
    q.join()
