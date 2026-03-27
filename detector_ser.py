# 使用python3.4 才可以执行
import DexelaPy
import sys
from concurrent.futures import ThreadPoolExecutor
import concurrent
from threading import Thread, Lock
import queue
from multiprocessing.connection import Client

import numpy as np


class CBData:
    def __init__(self, start_count, filename, number):
        self.start_count = start_count
        self.filename = filename
        self.number = number


class Detector:
    def __init__(self):
        try:
            scanner = DexelaPy.BusScannerPy()
            count = scanner.EnumerateDevices()

            info = scanner.GetDevice(0)
            self.detector = DexelaPy.DexelaDetectorPy(info)
            self.pool = ThreadPoolExecutor(max_workers=10)
            self.finished = True
            self.client = Client(r"\\.\pipe\detectResult", authkey=b"ctRestruct")
            self.client_lock = Lock()
        except DexelaPy.DexelaExceptionPy as e:
            # 初始化错误
            sys.stdout.write("ERROR01\n")
            sys.stdout.flush()
            sys.exit(1)

    def set_seq_mode(
        self, exposureTime, exit_queue=None, gap_time=0,scan_number=0, filename=None,
    ):
        self.exposureTime = exposureTime
        self.gap_time = gap_time
        expMode = DexelaPy.ExposureModes.Frame_Rate_exposure
        binFmt = DexelaPy.bins.x11
        wellMode = DexelaPy.FullWellModes.High
        trigger = DexelaPy.ExposureTriggerSource.Internal_Software
        p = self.detector.OpenBoard()
        self.w = self.detector.GetBufferXdim()
        self.h = self.detector.GetBufferYdim()
        self.detector.SetFullWellMode(wellMode)
        self.detector.SetExposureTime(exposureTime)
        self.detector.SetBinningMode(binFmt)
        self.detector.SetTriggerSource(trigger)
        self.detector.SetExposureMode(expMode)
        self.detector.SetGapTime(gap_time)
        self.scan_number = scan_number
        self.exit_queue = exit_queue
        if scan_number > 0:
            self.detector.SetNumOfExposures(scan_number)

    # def set_snap_mode(self, exposureTime):
    #    self.exposureTime = exposureTime
    #    expMode = DexelaPy.ExposureModes.Expose_and_read
    #    binFmt = DexelaPy.bins.x11
    #    wellMode = DexelaPy.FullWellModes.High
    #    trigger = DexelaPy.ExposureTriggerSource.Internal_Software
    #    p = self.detector.OpenBoard()
    #    self.w = self.detector.GetBufferXdim()
    #    self.h = self.detector.GetBufferYdim()
    #    self.detector.SetFullWellMode(wellMode)
    #    self.detector.SetExposureTime(exposureTime)
    #    self.detector.SetBinningMode(binFmt)
    #    self.detector.SetTriggerSource(trigger)
    #    self.detector.SetExposureMode(expMode)
    #    model = self.detector.GetModelNumber()

    # def snap(self,filename=None):
    #    img = DexelaPy.DexImagePy()
    #    try:
    #        self.detector.Snap(1, self.exposureTime+500)
    #        self.detector.ReadBuffer(1,img)
    #    except DexelaPy.DexelaExceptionPy as e:
    #        return None
    #    if filename is None:
    #        filename = 'Image_%dx%d.tif' % (img.GetImageXdim(),img.GetImageYdim())
    #    fut = self.pool.submit(self.trans_img, img, filename)
    #    return fut

    # def trans_img(self, img, filename):
    #    img.UnscrambleImage()
    #    buf = img.GetPlaneData()
    #    ar = np.array(buf, dtype=np.uint16)
    #    buf = ar.tobytes()
    #    return buf

    def can_exit(self):
        if self.exit_queue is None:
            return True
        try:
            cmd = self.exit_queue.get_nowait()
        except queue.Empty:
            return False
        return True

    def seq_save(self, count, img, client):
        img.UnscrambleImage()
        #sys.stderr.write("send {}\r\n".format(count))
        buf = img.GetPlaneData()
        ar = np.array(buf, dtype=np.uint16)
        buf = ar.tobytes()
        self.client_lock.acquire()
        client.send((count, buf))
        self.client_lock.release()

    def seq_start(self):
        self.detector.GoLiveSeq()
        startCount = self.detector.GetFieldCount()
        self.detector.CheckForLiveError()
        self.detector.SoftwareTrigger()
        self.finished = False
        sys.stderr.write("seq start! {}".format(self.scan_number))
        sys.stderr.flush()
        imCnt = 0
        fut_list = []
        while imCnt < self.scan_number:
            try:
                self.detector.WaitImage(self.exposureTime + self.gap_time)
                now_count = self.detector.GetFieldCount()
                count = now_count -  startCount -1
            except DexelaPy.DexelaExceptionPy as e:
                sys.stderr.write("error for receive image {} {}\r\n".format(imCnt, e))
                imCnt += 1
                continue
            capBuf = self.detector.GetCapturedBuffer()
            img = DexelaPy.DexImagePy()
            self.detector.ReadBuffer(capBuf, img)
            sys.stderr.write("receive image {} {}\r\n".format(imCnt, capBuf))
            fut = self.pool.submit(self.seq_save, imCnt, img, self.client)
            imCnt += 1
            fut_list.append(fut)

        for fut in fut_list:
            fut.result()
        if self.detector.IsLive() == True:
            self.detector.GoUnLive()
        sys.stderr.write("end!")
        sys.stderr.flush()

    def __del__(self):
        self.detector.CloseBoard()
        return


#def fin_loop(fut_queue, exit_queue):
#    client = Client(r"\\.\pipe\detectResult", authkey=b"ctRestruct")
#    exited = False
#    while True:
#        try:
#            if exited:
#                if fut_queue.empty():
#                    break
#            exit_cmd = exit_queue.get_nowait()
#            if exit_cmd:
#                exited = True
#                if fut_queue.empty():
#                    break
#        except queue.Empty:
#            pass
#        try:
#            filename, fut = fut_queue.get(timeout=1)
#            if fut:
#                buf = fut.result(timeout=10)
#                client.send((filename, buf))
#                fut_queue.task_done()
#        except queue.Empty:
#            pass
#        except concurrent.futures._base.TimeoutError:
#            pass
#        except Exception as e:
#            raise e


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 参数错误
        sys.stdout.write("ERROR02\n")
        sys.stdout.flush()
    progress = sys.argv[1]
    # if progress == 'snap':
    #    exposeTime = 200
    #    if len(sys.argv) == 3:
    #        exposeTime = int(sys.argv[2])
    #    detector = Detector()
    #    detector.set_snap_mode(exposeTime)
    #    fut_queue = queue.Queue()
    #    exit_queue = queue.Queue()
    #    fin_thread = Thread(target=fin_loop, args=(fut_queue,exit_queue))
    #    fin_thread.start()
    #    while True:
    #        cmd = sys.stdin.readline()
    #        if cmd.startswith('snap'):
    #            _, filename = cmd.split(' ')
    #            filename=filename.strip()
    #            fut = detector.snap(filename)
    #            sys.stdout.write('ok\n')
    #            sys.stdout.flush()
    #            fut_queue.put((filename, fut))
    #        elif cmd.startswith('exit'):
    #            exit_queue.put(True)
    #            break
    #        else:
    #            break

    if progress == "seq":
        exposeTime = int(sys.argv[2])
        gapTime = int(sys.argv[3])
        number = int(sys.argv[4])
        detector = Detector()
        exit_queue = queue.Queue()
        sys.stderr.write("exposeTim {} gapTime{} number {}\r\n".format(exposeTime, gapTime, number))
        detector.set_seq_mode(exposeTime, exit_queue,  gapTime, number+1)

        sys.stdout.write("READY\n")
        sys.stdout.flush()

        start_cmd = sys.stdin.readline()
        if not start_cmd.startswith("start"):
            sys.stdout.write("ERROR4\n")
            sys.stdout.flush()
            sys.exit(1)
        fin_thread = Thread(target=detector.seq_start, daemon=True)
        fin_thread.start()
        sys.stderr.write("start6666")
        sys.stderr.flush()
        # exit_or_wait_cmd = sys.stdin.readline()
        # sys.stderr.write(exit_or_wait_cmd)
        # sys.stderr.flush()
        fin_thread.join()
        sys.stdout.write("EXIT\n")
        sys.stdout.flush()
