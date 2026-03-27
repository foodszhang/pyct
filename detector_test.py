import detector
det = detector.Detector()
fut = det.snap('test.tif')
print(fut.result())
